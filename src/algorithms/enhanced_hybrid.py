"""
Enhanced Hybrid K-means + NSGA-II approach for Home Health Care Routing and Scheduling
This implementation fixes objective calculation issues and improves the hybridization strategy
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
import copy
import random
from ..models.problem import Solution, Customer, Depot, Vehicle
from .nsga2 import NSGAII
from .kmeans_clustering import CustomerClustering
import matplotlib.pyplot as plt

class EnhancedHybridKmeansNSGAII:
    """
    Enhanced hybrid approach with multiple improvements:
    1. Fixed objective calculation and validation
    2. Improved cluster combination strategies  
    3. Smart vehicle allocation per cluster
    4. Enhanced solution construction methods
    5. Better diversity preservation
    """
    
    def __init__(self,
                 n_clusters: int = None,
                 population_size_per_cluster: int = 30,
                 max_generations: int = 150,
                 crossover_probability: float = 0.85,
                 mutation_probability: float = 0.15,
                 combination_strategies: List[str] = None,
                 random_seed: int = 42):
        
        self.n_clusters = n_clusters
        self.population_size_per_cluster = population_size_per_cluster
        self.max_generations = max_generations
        self.crossover_prob = crossover_probability
        self.mutation_prob = mutation_probability
        self.random_seed = random_seed
        
        # Enhanced combination strategies
        self.combination_strategies = combination_strategies or [
            'direct_merge', 'iterative_construction', 'greedy_merge', 'best_combination'
        ]
        
        # Components
        self.clusterer = None
        self.cluster_optimizers: Dict[int, NSGAII] = {}
        
        # Enhanced results tracking
        self.cluster_solutions: Dict[int, List[Solution]] = {}
        self.global_pareto_front: List[Solution] = []
        self.clustering_stats = {}
        self.combination_stats = {}
        self.performance_metrics = {}
        
    def solve(self, depot: Depot, customers: List[Customer], vehicles: List[Vehicle]) -> List[Solution]:
        """
        Enhanced hybrid algorithm with multiple improvements
        """
        print(f"\n🔬 Starting Enhanced Hybrid K-means + NSGA-II Algorithm")
        print(f"Customers: {len(customers)}, Vehicles: {len(vehicles)}")
        
        # Step 0: Validate inputs and setup
        self._validate_inputs(depot, customers, vehicles)
        
        # Auto-determine optimal number of clusters if not specified
        if self.n_clusters is None:
            self.n_clusters = self._determine_optimal_clusters(customers, vehicles)
        
        print(f"Number of clusters: {self.n_clusters}")
        
        # Step 1: Enhanced K-means clustering
        clusters = self._enhanced_clustering(customers, depot)
        
        # Step 2: Optimized cluster optimization
        cluster_pareto_fronts = self._optimize_clusters_enhanced(clusters, depot, vehicles)
        
        # Step 3: Multi-strategy global solution construction
        global_solutions = self._construct_global_solutions_multi_strategy(
            cluster_pareto_fronts, depot, vehicles
        )
        
        # Step 4: Enhanced Pareto front filtering and validation
        self.global_pareto_front = self._filter_and_validate_solutions(global_solutions)
        
        # Step 5: Performance analysis and metrics
        self._calculate_performance_metrics()
        
        print(f"✅ Enhanced Hybrid Algorithm Completed!")
        print(f"Global Pareto front size: {len(self.global_pareto_front)}")
        
        return self.global_pareto_front
    
    def _validate_inputs(self, depot: Depot, customers: List[Customer], vehicles: List[Vehicle]):
        """Validate input data and fix common issues"""
        if not customers:
            raise ValueError("No customers provided")
        if not vehicles:
            raise ValueError("No vehicles provided")
        
        # Validate customer data
        for customer in customers:
            if customer.preferred_time is None:
                # Auto-set preferred time as middle of time window
                customer.preferred_time = customer.ready_time + (customer.due_date - customer.ready_time) // 2
            
            # Ensure reasonable time windows
            if customer.due_date <= customer.ready_time:
                customer.due_date = customer.ready_time + 100  # Add reasonable buffer
    
    def _determine_optimal_clusters(self, customers: List[Customer], vehicles: List[Vehicle]) -> int:
        """Automatically determine optimal number of clusters"""
        n_customers = len(customers)
        n_vehicles = len(vehicles)
        
        # Use heuristic: customers per vehicle ratio with bounds
        customers_per_cluster = max(3, min(8, n_customers // n_vehicles))
        optimal_clusters = max(2, min(n_vehicles, n_customers // customers_per_cluster))
        
        return optimal_clusters
    
    def _enhanced_clustering(self, customers: List[Customer], depot: Depot) -> Dict[int, List[Customer]]:
        """Enhanced K-means clustering with better initialization"""
        print(f"\n📊 Step 1: Enhanced K-means Clustering")
        
        self.clusterer = CustomerClustering(n_clusters=self.n_clusters, random_state=self.random_seed)
        clusters = self.clusterer.fit_customers(customers, depot)
        
        # Calculate enhanced clustering statistics
        cluster_sizes = [len(cluster_customers) for cluster_customers in clusters.values()]
        cluster_distances = {}
        cluster_demands = {}
        
        for cluster_id, cluster_customers in clusters.items():
            if cluster_customers:
                avg_distance = np.mean([depot.distance_to(c) for c in cluster_customers])
                total_demand = sum(c.demand for c in cluster_customers)
                cluster_distances[cluster_id] = avg_distance
                cluster_demands[cluster_id] = total_demand
                
                print(f"  Cluster {cluster_id}: {len(cluster_customers)} customers, "
                      f"avg distance: {avg_distance:.2f}, total demand: {total_demand}")
        
        self.clustering_stats = {
            'n_clusters': self.n_clusters,
            'cluster_sizes': cluster_sizes,
            'avg_cluster_size': np.mean(cluster_sizes),
            'std_cluster_size': np.std(cluster_sizes),
            'cluster_distances': cluster_distances,
            'cluster_demands': cluster_demands,
            'balance_ratio': min(cluster_sizes) / max(cluster_sizes) if max(cluster_sizes) > 0 else 0
        }
        
        print(f"Clustering completed:")
        for cluster_id, cluster_customers in clusters.items():
            if cluster_customers:
                avg_dist = cluster_distances.get(cluster_id, 0)
                demand = cluster_demands.get(cluster_id, 0)
                print(f"  Cluster {cluster_id}: {len(cluster_customers)} customers, "
                      f"avg distance to depot: {avg_dist:.2f}, total demand: {demand}")
        
        return clusters
    
    def _optimize_clusters_enhanced(self, clusters: Dict[int, List[Customer]], 
                                  depot: Depot, vehicles: List[Vehicle]) -> Dict[int, List[Solution]]:
        """Enhanced cluster optimization with smart vehicle allocation"""
        print(f"\n🧬 Step 2: Enhanced NSGA-II Optimization per Cluster")
        
        cluster_pareto_fronts = {}
        
        # Smart vehicle allocation based on cluster characteristics
        vehicle_allocation = self._allocate_vehicles_smartly(clusters, vehicles)
        
        for cluster_id, cluster_customers in clusters.items():
            if not cluster_customers:
                cluster_pareto_fronts[cluster_id] = []
                continue
                
            print(f"\n  Optimizing Cluster {cluster_id} ({len(cluster_customers)} customers)")
            
            # Get allocated vehicles for this cluster
            cluster_vehicles = vehicle_allocation[cluster_id]
            
            # Create enhanced NSGA-II optimizer
            optimizer = NSGAII(
                population_size=self.population_size_per_cluster,
                max_generations=self.max_generations,
                crossover_probability=self.crossover_prob,
                mutation_probability=self.mutation_prob,
                random_seed=self.random_seed + cluster_id
            )
            
            # Optimize cluster with validation
            pareto_front = optimizer.solve(depot, cluster_customers, cluster_vehicles)
            
            # Validate and filter cluster solutions
            valid_solutions = []
            for solution in pareto_front:
                self._validate_solution_objectives(solution)
                if solution.objectives[0] > 0 or solution.objectives[1] > 0:  # Non-zero objectives
                    valid_solutions.append(solution)
            
            cluster_pareto_fronts[cluster_id] = valid_solutions
            self.cluster_optimizers[cluster_id] = optimizer
            
            print(f"    Cluster {cluster_id} valid solutions: {len(valid_solutions)}/{len(pareto_front)}")
            if valid_solutions:
                obj1_range = [min(s.objectives[0] for s in valid_solutions), 
                             max(s.objectives[0] for s in valid_solutions)]
                obj2_range = [min(s.objectives[1] for s in valid_solutions), 
                             max(s.objectives[1] for s in valid_solutions)]
                print(f"    Service time range: [{obj1_range[0]:.2f}, {obj1_range[1]:.2f}]")
                print(f"    Tardiness range: [{obj2_range[0]:.2f}, {obj2_range[1]:.2f}]")
        
        self.cluster_solutions = cluster_pareto_fronts
        return cluster_pareto_fronts
    
    def _allocate_vehicles_smartly(self, clusters: Dict[int, List[Customer]], 
                                 vehicles: List[Vehicle]) -> Dict[int, List[Vehicle]]:
        """Smart vehicle allocation based on cluster demand and distance"""
        vehicle_allocation = {}
        
        # Calculate cluster priorities based on demand and size
        cluster_priorities = {}
        total_demand = 0
        
        for cluster_id, cluster_customers in clusters.items():
            if cluster_customers:
                demand = sum(c.demand for c in cluster_customers)
                size = len(cluster_customers)
                priority = demand * size  # Combined metric
                cluster_priorities[cluster_id] = priority
                total_demand += demand
        
        # Allocate vehicles proportionally to priority
        remaining_vehicles = vehicles.copy()
        
        for cluster_id in sorted(cluster_priorities.keys(), key=lambda x: cluster_priorities[x], reverse=True):
            if not remaining_vehicles:
                vehicle_allocation[cluster_id] = []
                continue
                
            cluster_customers = clusters[cluster_id]
            if not cluster_customers:
                vehicle_allocation[cluster_id] = []
                continue
            
            # Calculate needed vehicles for this cluster
            cluster_demand = sum(c.demand for c in cluster_customers)
            min_vehicles_needed = max(1, int(np.ceil(cluster_demand / vehicles[0].capacity)))
            
            # Allocate vehicles (at least 1, at most what's needed and available)
            vehicles_to_allocate = min(min_vehicles_needed, len(remaining_vehicles))
            vehicles_to_allocate = max(1, vehicles_to_allocate)
            
            allocated_vehicles = remaining_vehicles[:vehicles_to_allocate]
            vehicle_allocation[cluster_id] = allocated_vehicles
            remaining_vehicles = remaining_vehicles[vehicles_to_allocate:]
        
        return vehicle_allocation
    
    def _construct_global_solutions_multi_strategy(self, cluster_pareto_fronts: Dict[int, List[Solution]],
                                                 depot: Depot, vehicles: List[Vehicle]) -> List[Solution]:
        """Multi-strategy approach to construct global solutions"""
        print(f"\n🌐 Step 3: Multi-Strategy Global Solution Construction")
        
        all_global_solutions = []
        strategy_results = {}
        
        for strategy in self.combination_strategies:
            print(f"  Applying strategy: {strategy}")
            
            if strategy == 'direct_merge':
                solutions = self._direct_merge_strategy(cluster_pareto_fronts, depot, vehicles)
            elif strategy == 'iterative_construction':
                solutions = self._iterative_construction_strategy(cluster_pareto_fronts, depot, vehicles)
            elif strategy == 'greedy_merge':
                solutions = self._greedy_merge_strategy(cluster_pareto_fronts, depot, vehicles)
            elif strategy == 'best_combination':
                solutions = self._best_combination_strategy(cluster_pareto_fronts, depot, vehicles)
            else:
                continue
            
            strategy_results[strategy] = len(solutions)
            all_global_solutions.extend(solutions)
            print(f"    {strategy}: {len(solutions)} solutions generated")
        
        self.combination_stats = strategy_results
        return all_global_solutions
    
    def _direct_merge_strategy(self, cluster_pareto_fronts: Dict[int, List[Solution]],
                             depot: Depot, vehicles: List[Vehicle]) -> List[Solution]:
        """Direct merge: Combine routes from different cluster solutions"""
        merged_solutions = []
        
        # Get valid cluster solutions
        valid_clusters = {k: v for k, v in cluster_pareto_fronts.items() if v}
        
        if not valid_clusters:
            return []
        
        max_combinations = min(50, np.prod([len(solutions) for solutions in valid_clusters.values()]))
        
        for _ in range(max_combinations):
            global_solution = Solution(depot, vehicles)
            vehicle_idx = 0
            
            for cluster_id, cluster_solutions in valid_clusters.items():
                if not cluster_solutions or vehicle_idx >= len(vehicles):
                    continue
                
                # Select a random solution from this cluster
                cluster_solution = random.choice(cluster_solutions)
                
                # Add routes from cluster to global solution
                for route in cluster_solution.routes:
                    if route.customers and vehicle_idx < len(vehicles):
                        global_route = global_solution.routes[vehicle_idx]
                        for customer in route.customers:
                            global_route.add_customer(customer)
                        vehicle_idx += 1
            
            # Validate and add if customers were assigned
            if global_solution.get_all_customers():
                self._validate_solution_objectives(global_solution)
                merged_solutions.append(global_solution)
        
        return merged_solutions
    
    def _iterative_construction_strategy(self, cluster_pareto_fronts: Dict[int, List[Solution]],
                                       depot: Depot, vehicles: List[Vehicle]) -> List[Solution]:
        """Iterative construction with best solutions from each cluster"""
        iterative_solutions = []
        
        # Get best solutions from each cluster (diverse selection)
        cluster_best_solutions = {}
        for cluster_id, solutions in cluster_pareto_fronts.items():
            if not solutions:
                continue
                
            # Select diverse solutions: best obj1, best obj2, compromise, random
            best_solutions = []
            if len(solutions) >= 1:
                best_obj1 = min(solutions, key=lambda s: s.objectives[0])
                best_solutions.append(best_obj1)
            if len(solutions) >= 2:
                best_obj2 = min(solutions, key=lambda s: s.objectives[1])
                best_solutions.append(best_obj2)
            if len(solutions) >= 3:
                compromise = min(solutions, key=lambda s: sum(s.objectives))
                best_solutions.append(compromise)
            if len(solutions) >= 4:
                random_solution = random.choice(solutions)
                best_solutions.append(random_solution)
                
            cluster_best_solutions[cluster_id] = best_solutions
        
        # Create combinations iteratively
        max_iterations = 20
        for iteration in range(max_iterations):
            global_solution = Solution(depot, vehicles)
            vehicle_idx = 0
            
            for cluster_id in sorted(cluster_best_solutions.keys()):
                if vehicle_idx >= len(vehicles):
                    break
                    
                cluster_solutions = cluster_best_solutions[cluster_id]
                if not cluster_solutions:
                    continue
                
                # Select solution based on iteration (round-robin style)
                solution_idx = iteration % len(cluster_solutions)
                cluster_solution = cluster_solutions[solution_idx]
                
                # Add best route from this cluster solution
                best_route = None
                min_cost = float('inf')
                
                for route in cluster_solution.routes:
                    if route.customers:
                        route_cost = route.total_time + route.total_tardiness
                        if route_cost < min_cost:
                            min_cost = route_cost
                            best_route = route
                
                if best_route and vehicle_idx < len(vehicles):
                    global_route = global_solution.routes[vehicle_idx]
                    for customer in best_route.customers:
                        global_route.add_customer(customer)
                    vehicle_idx += 1
            
            if global_solution.get_all_customers():
                self._validate_solution_objectives(global_solution)
                iterative_solutions.append(global_solution)
        
        return iterative_solutions
    
    def _greedy_merge_strategy(self, cluster_pareto_fronts: Dict[int, List[Solution]],
                             depot: Depot, vehicles: List[Vehicle]) -> List[Solution]:
        """Greedy merge based on solution quality"""
        greedy_solutions = []
        
        # Collect all cluster solutions with quality scores
        all_cluster_solutions = []
        for cluster_id, solutions in cluster_pareto_fronts.items():
            for solution in solutions:
                quality_score = 1.0 / (sum(solution.objectives) + 1)  # Higher is better
                all_cluster_solutions.append((solution, quality_score, cluster_id))
        
        # Sort by quality
        all_cluster_solutions.sort(key=lambda x: x[1], reverse=True)
        
        # Create greedy combinations
        max_solutions = min(30, len(all_cluster_solutions))
        
        for start_idx in range(0, max_solutions, 3):
            global_solution = Solution(depot, vehicles)
            vehicle_idx = 0
            used_clusters = set()
            
            # Greedily select best solutions from different clusters
            for i in range(start_idx, min(start_idx + len(vehicles), len(all_cluster_solutions))):
                solution, quality, cluster_id = all_cluster_solutions[i]
                
                # Skip if cluster already used (for diversity)
                if cluster_id in used_clusters:
                    continue
                used_clusters.add(cluster_id)
                
                # Add best route from this solution
                if vehicle_idx < len(vehicles):
                    best_route = max(solution.routes, key=lambda r: len(r.customers) if r.customers else 0)
                    if best_route.customers:
                        global_route = global_solution.routes[vehicle_idx]
                        for customer in best_route.customers:
                            global_route.add_customer(customer)
                        vehicle_idx += 1
            
            if global_solution.get_all_customers():
                self._validate_solution_objectives(global_solution)
                greedy_solutions.append(global_solution)
        
        return greedy_solutions
    
    def _best_combination_strategy(self, cluster_pareto_fronts: Dict[int, List[Solution]],
                                 depot: Depot, vehicles: List[Vehicle]) -> List[Solution]:
        """Best combination strategy focusing on optimal solutions"""
        best_solutions = []
        
        # For each cluster, get the single best solution
        cluster_champions = {}
        for cluster_id, solutions in cluster_pareto_fronts.items():
            if solutions:
                # Champion is the solution with minimum combined objectives
                champion = min(solutions, key=lambda s: sum(s.objectives))
                cluster_champions[cluster_id] = champion
        
        if not cluster_champions:
            return []
        
        # Create the ultimate combination
        global_solution = Solution(depot, vehicles)
        vehicle_idx = 0
        
        # Add routes from all cluster champions
        for cluster_id, champion_solution in cluster_champions.items():
            for route in champion_solution.routes:
                if route.customers and vehicle_idx < len(vehicles):
                    global_route = global_solution.routes[vehicle_idx]
                    for customer in route.customers:
                        global_route.add_customer(customer)
                    vehicle_idx += 1
        
        if global_solution.get_all_customers():
            self._validate_solution_objectives(global_solution)
            best_solutions.append(global_solution)
        
        return best_solutions
    
    def _filter_and_validate_solutions(self, solutions: List[Solution]) -> List[Solution]:
        """Filter and validate global solutions to create final Pareto front"""
        print(f"    Filtering {len(solutions)} global solutions...")
        
        # Step 1: Validate all solutions
        valid_solutions = []
        for solution in solutions:
            self._validate_solution_objectives(solution)
            if self._is_valid_solution(solution):
                valid_solutions.append(solution)
        
        print(f"    Valid solutions: {len(valid_solutions)}")
        
        if not valid_solutions:
            return []
        
        # Step 2: Remove duplicates (same objective values)
        unique_solutions = []
        seen_objectives = set()
        
        for solution in valid_solutions:
            obj_tuple = tuple(round(obj, 2) for obj in solution.objectives)
            if obj_tuple not in seen_objectives:
                seen_objectives.add(obj_tuple)
                unique_solutions.append(solution)
        
        print(f"    Unique solutions: {len(unique_solutions)}")
        
        # Step 3: Apply Pareto dominance filtering
        pareto_front = self._extract_pareto_front(unique_solutions)
        
        print(f"    Final Pareto front: {len(pareto_front)} solutions")
        
        return pareto_front
    
    def _validate_solution_objectives(self, solution: Solution):
        """Ensure solution has correct objectives calculated"""
        solution._calculate_objectives()  # Recalculate to be sure
        
        # Additional validation: manually verify objectives
        total_service_time = 0.0
        total_tardiness = 0.0
        
        for route in solution.routes:
            if route.customers:
                route._calculate_route_metrics()  # Ensure route metrics are calculated
                total_service_time += route.total_time
                total_tardiness += route.total_tardiness
        
        # Update with verified values
        solution.objectives = [total_service_time, total_tardiness]
    
    def _is_valid_solution(self, solution: Solution) -> bool:
        """Check if solution is valid for inclusion in results"""
        # Must have customers
        if not solution.get_all_customers():
            return False
            
        # Must have reasonable objectives (not all zero unless truly optimal)
        if solution.objectives[0] == 0.0 and solution.objectives[1] == 0.0:
            # Only valid if it's actually a zero-cost solution
            total_customers = len(solution.get_all_customers())
            return total_customers > 0
        
        # Must have positive service time if customers exist
        if len(solution.get_all_customers()) > 0 and solution.objectives[0] <= 0:
            return False
        
        return True
    
    def _extract_pareto_front(self, solutions: List[Solution]) -> List[Solution]:
        """Extract non-dominated solutions (Pareto front)"""
        if not solutions:
            return []
        
        pareto_front = []
        
        for i, solution_i in enumerate(solutions):
            is_dominated = False
            
            for j, solution_j in enumerate(solutions):
                if i != j and solution_j.dominates(solution_i):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(solution_i)
        
        return pareto_front
    
    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.global_pareto_front:
            self.performance_metrics = {'pareto_front_size': 0}
            return
        
        # Basic metrics
        pareto_size = len(self.global_pareto_front)
        
        # Objective statistics
        obj1_values = [sol.objectives[0] for sol in self.global_pareto_front]
        obj2_values = [sol.objectives[1] for sol in self.global_pareto_front]
        
        # Solution quality metrics
        feasible_solutions = [sol for sol in self.global_pareto_front if sol.is_feasible]
        
        self.performance_metrics = {
            'pareto_front_size': pareto_size,
            'feasible_solutions': len(feasible_solutions),
            'feasibility_rate': len(feasible_solutions) / pareto_size if pareto_size > 0 else 0,
            'obj1_range': [min(obj1_values), max(obj1_values)] if obj1_values else [0, 0],
            'obj2_range': [min(obj2_values), max(obj2_values)] if obj2_values else [0, 0],
            'obj1_mean': np.mean(obj1_values) if obj1_values else 0,
            'obj2_mean': np.mean(obj2_values) if obj2_values else 0,
            'best_combined': min(sum(sol.objectives) for sol in self.global_pareto_front) if self.global_pareto_front else 0,
        }
    
    def get_detailed_results(self) -> Dict:
        """Get comprehensive results including all metrics and statistics"""
        return {
            'clustering_stats': self.clustering_stats,
            'combination_stats': self.combination_stats,
            'performance_metrics': self.performance_metrics,
            'n_clusters': self.n_clusters,
            'cluster_optimizers': len(self.cluster_optimizers),
            'strategies_used': self.combination_strategies
        }