"""
Hybrid K-means + NSGA-II approach for Home Health Care Routing and Scheduling
This implements the novel hybridization approach from the research paper
"""
import numpy as np
from typing import List, Dict, Tuple
import copy
from ..models.problem import Solution, Customer, Depot, Vehicle
from .nsga2 import NSGAII
from .kmeans_clustering import CustomerClustering
import matplotlib.pyplot as plt

class HybridKmeansNSGAII:
    """
    Hybrid approach combining K-means clustering with NSGA-II
    Main innovation: Divide customers into clusters, optimize each cluster separately,
    then combine results to form global Pareto front
    """
    
    def __init__(self,
                 n_clusters: int = None,
                 population_size_per_cluster: int = 50,
                 max_generations: int = 300,
                 crossover_probability: float = 0.9,
                 mutation_probability: float = 0.1,
                 random_seed: int = 42):
        
        self.n_clusters = n_clusters
        self.population_size_per_cluster = population_size_per_cluster
        self.max_generations = max_generations
        self.crossover_prob = crossover_probability
        self.mutation_prob = mutation_probability
        self.random_seed = random_seed
        
        # Components
        self.clusterer = None
        self.cluster_optimizers: Dict[int, NSGAII] = {}
        
        # Results
        self.cluster_solutions: Dict[int, List[Solution]] = {}
        self.global_pareto_front: List[Solution] = []
        self.clustering_stats = {}
        
    def solve(self, depot: Depot, customers: List[Customer], vehicles: List[Vehicle]) -> List[Solution]:
        """
        Main hybrid algorithm:
        1. Cluster customers using K-means
        2. Run NSGA-II on each cluster
        3. Combine sub-solutions to form global Pareto front
        """
        print(f"\n🔬 Starting Hybrid K-means + NSGA-II Algorithm")
        print(f"Customers: {len(customers)}, Vehicles: {len(vehicles)}")
        
        # Determine number of clusters
        if self.n_clusters is None:
            self.n_clusters = min(len(vehicles), max(2, len(customers) // 10))
        
        print(f"Number of clusters: {self.n_clusters}")
        
        # Step 1: K-means clustering
        print(f"\n📊 Step 1: K-means Clustering")
        clusters = self._cluster_customers(customers, depot)
        
        # Step 2: Optimize each cluster with NSGA-II
        print(f"\n🧬 Step 2: NSGA-II Optimization per Cluster")
        cluster_pareto_fronts = self._optimize_clusters(clusters, depot, vehicles)
        
        # Step 3: Combine results to global Pareto front
        print(f"\n🌐 Step 3: Global Pareto Front Construction")
        global_pareto_front = self._construct_global_pareto_front(cluster_pareto_fronts, depot, vehicles)
        
        self.global_pareto_front = global_pareto_front
        
        print(f"✅ Hybrid Algorithm Completed!")
        print(f"Global Pareto front size: {len(global_pareto_front)}")
        
        return global_pareto_front
    
    def _cluster_customers(self, customers: List[Customer], depot: Depot) -> Dict[int, List[Customer]]:
        """Step 1: Cluster customers using K-means"""
        self.clusterer = CustomerClustering(
            n_clusters=self.n_clusters,
            random_state=self.random_seed
        )
        
        clusters = self.clusterer.fit_customers(customers, depot)
        self.clustering_stats = self.clusterer.get_cluster_statistics()
        
        print(f"Clustering completed:")
        for cluster_id, cluster_customers in clusters.items():
            if cluster_customers:
                avg_distance_to_depot = np.mean([depot.distance_to(c) for c in cluster_customers])
                total_demand = sum(c.demand for c in cluster_customers)
                print(f"  Cluster {cluster_id}: {len(cluster_customers)} customers, "
                      f"avg distance to depot: {avg_distance_to_depot:.2f}, "
                      f"total demand: {total_demand}")
        
        return clusters
    
    def _optimize_clusters(self, clusters: Dict[int, List[Customer]], 
                          depot: Depot, vehicles: List[Vehicle]) -> Dict[int, List[Solution]]:
        """Step 2: Run NSGA-II optimization on each cluster"""
        cluster_pareto_fronts = {}
        
        for cluster_id, cluster_customers in clusters.items():
            if not cluster_customers:
                cluster_pareto_fronts[cluster_id] = []
                continue
            
            print(f"\n  Optimizing Cluster {cluster_id} ({len(cluster_customers)} customers)")
            
            # Determine vehicles for this cluster
            vehicles_per_cluster = max(1, len(vehicles) // self.n_clusters)
            if cluster_id == self.n_clusters - 1:  # Last cluster gets remaining vehicles
                vehicles_per_cluster = len(vehicles) - (self.n_clusters - 1) * vehicles_per_cluster
            
            cluster_vehicles = vehicles[:vehicles_per_cluster]
            
            # Create NSGA-II optimizer for this cluster
            optimizer = NSGAII(
                population_size=self.population_size_per_cluster,
                max_generations=self.max_generations,
                crossover_probability=self.crossover_prob,
                mutation_probability=self.mutation_prob,
                random_seed=self.random_seed + cluster_id
            )
            
            # Optimize cluster
            pareto_front = optimizer.solve(depot, cluster_customers, cluster_vehicles)
            
            cluster_pareto_fronts[cluster_id] = pareto_front
            self.cluster_optimizers[cluster_id] = optimizer
            
            print(f"    Cluster {cluster_id} Pareto front: {len(pareto_front)} solutions")
        
        self.cluster_solutions = cluster_pareto_fronts
        return cluster_pareto_fronts
    
    def _construct_global_pareto_front(self, cluster_pareto_fronts: Dict[int, List[Solution]],
                                     depot: Depot, vehicles: List[Vehicle]) -> List[Solution]:
        """Step 3: Combine cluster solutions to form global Pareto front"""
        
        # Generate all possible combinations of cluster solutions
        global_solutions = []
        
        # Method 1: Direct combination of cluster solutions
        print("    Combining cluster solutions...")
        combined_solutions = self._combine_cluster_solutions(cluster_pareto_fronts, depot, vehicles)
        global_solutions.extend(combined_solutions)
        
        # Method 2: Iterative construction
        print("    Iterative construction...")
        iterative_solutions = self._iterative_construction(cluster_pareto_fronts, depot, vehicles)
        global_solutions.extend(iterative_solutions)
        
        # Remove duplicates and invalid solutions
        valid_solutions = []
        seen_objectives = set()
        
        for solution in global_solutions:
            if solution.is_feasible:
                obj_tuple = tuple(solution.objectives)
                if obj_tuple not in seen_objectives:
                    valid_solutions.append(solution)
                    seen_objectives.add(obj_tuple)
        
        # Extract global Pareto front
        global_pareto_front = self._extract_pareto_front(valid_solutions)
        
        print(f"    Combined {len(global_solutions)} solutions")
        print(f"    Valid solutions: {len(valid_solutions)}")
        print(f"    Global Pareto front: {len(global_pareto_front)}")
        
        return global_pareto_front
    
    def _combine_cluster_solutions(self, cluster_pareto_fronts: Dict[int, List[Solution]],
                                  depot: Depot, vehicles: List[Vehicle]) -> List[Solution]:
        """Directly combine solutions from different clusters"""
        combined_solutions = []
        
        # Get all non-empty clusters
        non_empty_clusters = {cid: solutions for cid, solutions in cluster_pareto_fronts.items() 
                             if solutions}
        
        if not non_empty_clusters:
            return []
        
        # Try different combination strategies
        max_combinations = min(100, max(10, len(vehicles) * 5))  # Limit combinations
        combinations_created = 0
        
        for i in range(max_combinations):
            # Select one solution from each cluster
            global_solution = Solution(depot, vehicles)
            used_vehicles = 0
            
            for cluster_id in sorted(non_empty_clusters.keys()):
                cluster_solutions = non_empty_clusters[cluster_id]
                if not cluster_solutions:
                    continue
                
                # Select a solution (round-robin or random)
                solution_idx = i % len(cluster_solutions)
                cluster_solution = cluster_solutions[solution_idx]
                
                # Add routes from cluster solution to global solution
                for route in cluster_solution.routes:
                    if route.customers and used_vehicles < len(vehicles):
                        # Copy route to global solution
                        global_route = global_solution.routes[used_vehicles]
                        for customer in route.customers:
                            global_route.add_customer(customer)
                        used_vehicles += 1
            
            if global_solution.get_all_customers():
                combined_solutions.append(global_solution)
                combinations_created += 1
        
        return combined_solutions
    
    def _iterative_construction(self, cluster_pareto_fronts: Dict[int, List[Solution]],
                              depot: Depot, vehicles: List[Vehicle]) -> List[Solution]:
        """Iteratively construct global solutions by merging clusters"""
        iterative_solutions = []
        
        # Get best solutions from each cluster
        cluster_best_solutions = {}
        for cluster_id, solutions in cluster_pareto_fronts.items():
            if solutions:
                # Select diverse solutions from Pareto front
                if len(solutions) >= 3:
                    # Best in obj1, best in obj2, and compromise solution
                    best_obj1 = min(solutions, key=lambda s: s.objectives[0])
                    best_obj2 = min(solutions, key=lambda s: s.objectives[1])
                    # Compromise: solution closest to ideal point (0,0)
                    compromise = min(solutions, key=lambda s: s.objectives[0] + s.objectives[1])
                    cluster_best_solutions[cluster_id] = [best_obj1, best_obj2, compromise]
                else:
                    cluster_best_solutions[cluster_id] = solutions
        
        # Create solutions by combining best solutions
        max_iterations = min(20, len(vehicles) * 2)
        
        for iteration in range(max_iterations):
            global_solution = Solution(depot, vehicles)
            vehicle_idx = 0
            
            # Iterate through clusters and assign routes
            for cluster_id in sorted(cluster_best_solutions.keys()):
                if vehicle_idx >= len(vehicles):
                    break
                
                solutions = cluster_best_solutions[cluster_id]
                if not solutions:
                    continue
                
                # Select solution based on iteration
                solution_idx = iteration % len(solutions)
                cluster_solution = solutions[solution_idx]
                
                # Add one route from this cluster solution
                for route in cluster_solution.routes:
                    if route.customers and vehicle_idx < len(vehicles):
                        global_route = global_solution.routes[vehicle_idx]
                        for customer in route.customers:
                            global_route.add_customer(customer)
                        vehicle_idx += 1
                        break  # Only take one route per cluster per iteration
            
            if global_solution.get_all_customers():
                iterative_solutions.append(global_solution)
        
        return iterative_solutions
    
    def _extract_pareto_front(self, solutions: List[Solution]) -> List[Solution]:
        """Extract Pareto front from a set of solutions"""
        if not solutions:
            return []
        
        pareto_front = []
        
        for candidate in solutions:
            is_dominated = False
            
            # Check if candidate is dominated by any other solution
            for other in solutions:
                if other != candidate and other.dominates(candidate):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(candidate)
        
        return pareto_front
    
    def visualize_clustering(self, save_path: str = None, show: bool = True):
        """Visualize the customer clustering"""
        if self.clusterer is None:
            print("No clustering performed yet")
            return
        
        self.clusterer.visualize_clusters(
            depot=self.clusterer.clusters.get('depot', None),  # This needs to be fixed
            save_path=save_path,
            show=show
        )
    
    def get_detailed_results(self) -> Dict:
        """Get detailed results including clustering and optimization statistics"""
        results = {
            'clustering_stats': self.clustering_stats,
            'n_clusters': self.n_clusters,
            'cluster_pareto_front_sizes': {},
            'global_pareto_front_size': len(self.global_pareto_front),
            'global_objectives': []
        }
        
        # Cluster-specific results
        for cluster_id, solutions in self.cluster_solutions.items():
            results['cluster_pareto_front_sizes'][cluster_id] = len(solutions)
        
        # Global Pareto front objectives
        if self.global_pareto_front:
            results['global_objectives'] = [sol.objectives for sol in self.global_pareto_front]
            
            # Calculate hypervolume and spread metrics
            obj1_values = [sol.objectives[0] for sol in self.global_pareto_front]
            obj2_values = [sol.objectives[1] for sol in self.global_pareto_front]
            
            results['objective_ranges'] = {
                'obj1_min': min(obj1_values),
                'obj1_max': max(obj1_values),
                'obj2_min': min(obj2_values),
                'obj2_max': max(obj2_values)
            }
        
        return results
    
    def compare_with_standard_nsga2(self, depot: Depot, customers: List[Customer], 
                                   vehicles: List[Vehicle]) -> Dict:
        """Compare hybrid approach with standard NSGA-II"""
        print(f"\n📊 Comparison: Hybrid vs Standard NSGA-II")
        
        # Run standard NSGA-II
        print("Running standard NSGA-II...")
        standard_nsga2 = NSGAII(
            population_size=self.population_size_per_cluster * self.n_clusters,
            max_generations=self.max_generations,
            crossover_probability=self.crossover_prob,
            mutation_probability=self.mutation_prob,
            random_seed=self.random_seed
        )
        
        standard_pareto_front = standard_nsga2.solve(depot, customers, vehicles)
        
        # Compare results
        comparison = {
            'hybrid': {
                'pareto_front_size': len(self.global_pareto_front),
                'objectives': [sol.objectives for sol in self.global_pareto_front] if self.global_pareto_front else []
            },
            'standard': {
                'pareto_front_size': len(standard_pareto_front),
                'objectives': [sol.objectives for sol in standard_pareto_front] if standard_pareto_front else []
            }
        }
        
        print(f"Hybrid Pareto front size: {comparison['hybrid']['pareto_front_size']}")
        print(f"Standard Pareto front size: {comparison['standard']['pareto_front_size']}")
        
        return comparison