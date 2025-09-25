"""
NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation
for Home Health Care Multi-Objective Vehicle Routing Problem
"""
import numpy as np
import random
from typing import List, Tuple, Dict
import copy
from collections import defaultdict
from ..models.problem import Solution, Customer, Depot, Vehicle, Route

class NSGAII:
    """NSGA-II algorithm implementation for HHC-MOVRPTW"""
    
    def __init__(self, 
                 population_size: int = 100,
                 max_generations: int = 500,
                 crossover_probability: float = 0.9,
                 mutation_probability: float = 0.1,
                 tournament_size: int = 2,
                 random_seed: int = 42):
        
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_prob = crossover_probability
        self.mutation_prob = mutation_probability
        self.tournament_size = tournament_size
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Algorithm state
        self.population: List[Solution] = []
        self.best_solutions: List[Solution] = []
        self.convergence_data = []
        self.generation_stats = []
        
    def solve(self, depot: Depot, customers: List[Customer], vehicles: List[Vehicle]) -> List[Solution]:
        """
        Main NSGA-II algorithm
        Returns: List of non-dominated solutions (Pareto front)
        """
        print(f"\n🚀 Starting NSGA-II Algorithm")
        print(f"Population size: {self.population_size}")
        print(f"Max generations: {self.max_generations}")
        print(f"Customers: {len(customers)}, Vehicles: {len(vehicles)}")
        
        # Initialize population
        self.population = self._initialize_population(depot, customers, vehicles)
        print(f"✓ Initial population created")
        
        # Evaluate initial population
        self._evaluate_population()
        print(f"✓ Initial population evaluated")
        
        # Debug: Check initial population objectives
        sample_objectives = []
        feasible_count = 0
        for solution in self.population[:5]:  # Check first 5 solutions
            sample_objectives.append(solution.objectives)
            if solution.is_feasible:
                feasible_count += 1
        
        print(f"✓ Sample objectives: {sample_objectives[:3]}")
        print(f"✓ Feasible solutions: {feasible_count}/{len(self.population)}")
        
        # Evolution loop
        for generation in range(self.max_generations):
            # Create offspring population
            offspring = self._create_offspring(depot, customers, vehicles)
            
            # Evaluate offspring
            for child in offspring:
                child._calculate_objectives()
            
            # Combine parent and offspring populations
            combined_population = self.population + offspring
            
            # Non-dominated sorting and crowding distance
            fronts = self._fast_non_dominated_sort(combined_population)
            
            # Select next generation
            self.population = self._select_next_generation(fronts, combined_population)
            
            # Track statistics
            self._track_generation_stats(generation)
            
            # Print progress
            if generation % 50 == 0 or generation == self.max_generations - 1:
                stats = self.generation_stats[-1] if self.generation_stats else {}
                print(f"Generation {generation:3d}: "
                      f"Pareto front size: {stats.get('pareto_front_size', 0):2d}, "
                      f"Avg obj1: {stats.get('avg_obj1', 0):.2f}, "
                      f"Avg obj2: {stats.get('avg_obj2', 0):.2f}")
        
        # Extract final Pareto front - prefer feasible solutions, but include best infeasible if none feasible
        final_fronts = self._fast_non_dominated_sort(self.population)
        if final_fronts:
            pareto_front = [self.population[i] for i in final_fronts[0]]
            
            # If no feasible solutions, at least return the best infeasible ones
            feasible_pareto = [sol for sol in pareto_front if sol.is_feasible]
            if not feasible_pareto and pareto_front:
                print(f"⚠️ No feasible solutions found, returning best infeasible solutions")
                # Sort by combined objectives to get the "best" infeasible solutions
                pareto_front = sorted(pareto_front, key=lambda sol: sum(sol.objectives))[:min(10, len(pareto_front))]
            elif feasible_pareto:
                pareto_front = feasible_pareto
        else:
            pareto_front = []
        
        print(f"✅ NSGA-II completed!")
        print(f"Final Pareto front size: {len(pareto_front)}")
        
        return pareto_front
    
    def _initialize_population(self, depot: Depot, customers: List[Customer], 
                             vehicles: List[Vehicle]) -> List[Solution]:
        """Initialize population with diverse solutions"""
        population = []
        
        for _ in range(self.population_size):
            solution = self._create_random_solution(depot, customers, vehicles)
            population.append(solution)
        
        return population
    
    def _create_random_solution(self, depot: Depot, customers: List[Customer], 
                              vehicles: List[Vehicle]) -> Solution:
        """Create a random feasible solution"""
        solution = Solution(depot, vehicles)
        available_customers = customers.copy()
        random.shuffle(available_customers)
        
        # Assign customers to routes using nearest insertion heuristic
        for customer in available_customers:
            best_route_idx = 0
            best_cost = float('inf')
            
            # Find best route to insert customer
            for route_idx, route in enumerate(solution.routes):
                # Check capacity constraint
                if route.total_demand + customer.demand <= vehicles[route_idx].capacity:
                    # Calculate insertion cost
                    if not route.customers:
                        cost = depot.distance_to(customer) * 2  # To and from depot
                    else:
                        # Try inserting at the end
                        last_customer = route.customers[-1]
                        cost = (last_customer.distance_to(customer) + 
                               customer.distance_to(depot) - 
                               last_customer.distance_to(depot))
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_route_idx = route_idx
            
            # Assign to best route
            solution.add_customer_to_route(customer, best_route_idx)
        
        return solution
    
    def _evaluate_population(self):
        """Evaluate fitness for all solutions in population"""
        for solution in self.population:
            solution._calculate_objectives()
    
    def _create_offspring(self, depot: Depot, customers: List[Customer], 
                         vehicles: List[Vehicle]) -> List[Solution]:
        """Create offspring population through selection, crossover, and mutation"""
        offspring = []
        
        while len(offspring) < self.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Ensure parents are different
            if parent1 == parent2:
                continue
            
            # Crossover
            if random.random() < self.crossover_prob:
                child1, child2 = self._crossover(parent1, parent2, depot, vehicles)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < self.mutation_prob:
                child1 = self._mutate(child1, depot, customers, vehicles)
            if random.random() < self.mutation_prob:
                child2 = self._mutate(child2, depot, customers, vehicles)
            
            offspring.extend([child1, child2])
        
        return offspring[:self.population_size]
    
    def _tournament_selection(self) -> Solution:
        """Tournament selection"""
        tournament = random.sample(self.population, self.tournament_size)
        
        # Select best solution in tournament (lowest rank, highest crowding distance)
        best = tournament[0]
        for candidate in tournament[1:]:
            if (candidate.dominates_count < best.dominates_count or
                (candidate.dominates_count == best.dominates_count and
                 candidate.crowding_distance > best.crowding_distance)):
                best = candidate
        
        return best
    
    def _crossover(self, parent1: Solution, parent2: Solution, 
                  depot: Depot, vehicles: List[Vehicle]) -> Tuple[Solution, Solution]:
        """Order crossover (OX) adapted for VRP"""
        # Get all customers from both parents
        customers1 = parent1.get_all_customers()
        customers2 = parent2.get_all_customers()
        
        if not customers1 or not customers2:
            return parent1.copy(), parent2.copy()
        
        # Create children
        child1 = Solution(depot, vehicles)
        child2 = Solution(depot, vehicles)
        
        # Simple route exchange crossover
        try:
            # Choose random routes to exchange
            route1_idx = random.randint(0, len(parent1.routes) - 1)
            route2_idx = random.randint(0, len(parent2.routes) - 1)
            
            # Copy parent structures
            child1 = parent1.copy()
            child2 = parent2.copy()
            
            # Exchange selected routes
            if (route1_idx < len(child1.routes) and route2_idx < len(child2.routes) and
                route1_idx < len(parent2.routes) and route2_idx < len(parent1.routes)):
                
                child1.routes[route1_idx] = parent2.routes[route1_idx].copy()
                child1.routes[route1_idx].vehicle_id = route1_idx
                
                child2.routes[route2_idx] = parent1.routes[route2_idx].copy()
                child2.routes[route2_idx].vehicle_id = route2_idx
        
        except Exception:
            # Fallback to parent solutions if crossover fails
            child1 = parent1.copy()
            child2 = parent2.copy()
        
        return child1, child2
    
    def _mutate(self, solution: Solution, depot: Depot, customers: List[Customer], 
               vehicles: List[Vehicle]) -> Solution:
        """Mutation operator: relocate, swap, or 2-opt"""
        mutated = solution.copy()
        
        mutation_type = random.choice(['relocate', 'swap', '2opt', 'route_swap'])
        
        try:
            if mutation_type == 'relocate':
                self._relocate_mutation(mutated)
            elif mutation_type == 'swap':
                self._swap_mutation(mutated)
            elif mutation_type == '2opt':
                self._two_opt_mutation(mutated)
            elif mutation_type == 'route_swap':
                self._route_swap_mutation(mutated)
        except Exception:
            # Return original if mutation fails
            pass
        
        return mutated
    
    def _relocate_mutation(self, solution: Solution):
        """Relocate a customer to a different position"""
        all_customers = solution.get_all_customers()
        if len(all_customers) < 2:
            return
        
        # Select random customer
        customer = random.choice(all_customers)
        route_pos = solution.get_route_for_customer(customer.id)
        
        if route_pos is None:
            return
        
        route_idx, pos = route_pos
        
        # Remove customer
        solution.routes[route_idx].remove_customer(customer.id)
        
        # Find new position
        new_route_idx = random.randint(0, len(solution.routes) - 1)
        new_pos = random.randint(0, max(0, len(solution.routes[new_route_idx].customers)))
        
        # Check capacity constraint
        vehicle_capacity = solution.vehicles[new_route_idx].capacity
        if (solution.routes[new_route_idx].total_demand + customer.demand <= vehicle_capacity):
            solution.routes[new_route_idx].add_customer(customer, new_pos)
        else:
            # Reinsert to original position
            solution.routes[route_idx].add_customer(customer, pos)
    
    def _swap_mutation(self, solution: Solution):
        """Swap two customers"""
        all_customers = solution.get_all_customers()
        if len(all_customers) < 2:
            return
        
        # Select two random customers
        customer1, customer2 = random.sample(all_customers, 2)
        
        route1_pos = solution.get_route_for_customer(customer1.id)
        route2_pos = solution.get_route_for_customer(customer2.id)
        
        if route1_pos is None or route2_pos is None:
            return
        
        route1_idx, pos1 = route1_pos
        route2_idx, pos2 = route2_pos
        
        # Remove both customers
        solution.routes[route1_idx].remove_customer(customer1.id)
        solution.routes[route2_idx].remove_customer(customer2.id)
        
        # Check capacity constraints and swap
        vehicle1_capacity = solution.vehicles[route1_idx].capacity
        vehicle2_capacity = solution.vehicles[route2_idx].capacity
        
        demand_diff1 = customer2.demand - customer1.demand
        demand_diff2 = customer1.demand - customer2.demand
        
        if (solution.routes[route1_idx].total_demand + demand_diff1 <= vehicle1_capacity and
            solution.routes[route2_idx].total_demand + demand_diff2 <= vehicle2_capacity):
            
            # Perform swap
            solution.routes[route1_idx].add_customer(customer2, pos1)
            solution.routes[route2_idx].add_customer(customer1, pos2)
        else:
            # Revert if capacity violated
            solution.routes[route1_idx].add_customer(customer1, pos1)
            solution.routes[route2_idx].add_customer(customer2, pos2)
    
    def _two_opt_mutation(self, solution: Solution):
        """2-opt improvement within a single route"""
        non_empty_routes = [route for route in solution.routes if len(route.customers) > 3]
        
        if not non_empty_routes:
            return
        
        route = random.choice(non_empty_routes)
        customers = route.customers
        
        if len(customers) < 4:
            return
        
        # Select two edges to swap
        i = random.randint(0, len(customers) - 3)
        j = random.randint(i + 2, len(customers) - 1)
        
        # Reverse segment between i+1 and j
        route.customers[i+1:j+1] = reversed(route.customers[i+1:j+1])
        route._calculate_route_metrics()
    
    def _route_swap_mutation(self, solution: Solution):
        """Swap entire routes between vehicles"""
        if len(solution.routes) < 2:
            return
        
        route1_idx, route2_idx = random.sample(range(len(solution.routes)), 2)
        
        # Swap routes
        temp_customers = solution.routes[route1_idx].customers.copy()
        solution.routes[route1_idx].customers = solution.routes[route2_idx].customers.copy()
        solution.routes[route2_idx].customers = temp_customers
        
        # Recalculate metrics
        solution.routes[route1_idx]._calculate_route_metrics()
        solution.routes[route2_idx]._calculate_route_metrics()
    
    def _fast_non_dominated_sort(self, population: List[Solution]) -> List[List[int]]:
        """Fast non-dominated sorting algorithm"""
        # Initialize domination counts and dominated solutions
        for solution in population:
            solution.dominates_count = 0
            solution.dominated_by = set()
        
        # Calculate domination relationships
        for i, solution_i in enumerate(population):
            for j, solution_j in enumerate(population):
                if i != j:
                    if solution_i.dominates(solution_j):
                        solution_i.dominated_by.add(j)
                    elif solution_j.dominates(solution_i):
                        solution_i.dominates_count += 1
        
        # Create fronts
        fronts = []
        current_front = []
        
        # First front (non-dominated solutions)
        for i, solution in enumerate(population):
            if solution.dominates_count == 0:
                current_front.append(i)
        
        fronts.append(current_front)
        
        # Subsequent fronts
        while current_front:
            next_front = []
            for i in current_front:
                for j in population[i].dominated_by:
                    population[j].dominates_count -= 1
                    if population[j].dominates_count == 0:
                        next_front.append(j)
            
            if next_front:
                fronts.append(next_front)
            current_front = next_front
        
        return fronts
    
    def _calculate_crowding_distance(self, front_indices: List[int], population: List[Solution]):
        """Calculate crowding distance for solutions in a front"""
        if len(front_indices) <= 2:
            for idx in front_indices:
                population[idx].crowding_distance = float('inf')
            return
        
        # Initialize distances
        for idx in front_indices:
            population[idx].crowding_distance = 0.0
        
        # Calculate for each objective
        for obj_idx in range(2):  # Two objectives
            # Sort by objective value
            front_indices.sort(key=lambda idx: population[idx].objectives[obj_idx])
            
            # Set boundary solutions to infinite distance
            population[front_indices[0]].crowding_distance = float('inf')
            population[front_indices[-1]].crowding_distance = float('inf')
            
            # Calculate objective range
            obj_min = population[front_indices[0]].objectives[obj_idx]
            obj_max = population[front_indices[-1]].objectives[obj_idx]
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
            
            # Calculate crowding distance for intermediate solutions
            for i in range(1, len(front_indices) - 1):
                if population[front_indices[i]].crowding_distance != float('inf'):
                    distance = ((population[front_indices[i+1]].objectives[obj_idx] - 
                               population[front_indices[i-1]].objectives[obj_idx]) / obj_range)
                    population[front_indices[i]].crowding_distance += distance
    
    def _select_next_generation(self, fronts: List[List[int]], 
                              combined_population: List[Solution]) -> List[Solution]:
        """Select next generation using elitism and crowding distance"""
        next_generation = []
        
        for front in fronts:
            if len(next_generation) + len(front) <= self.population_size:
                # Add entire front
                for idx in front:
                    next_generation.append(combined_population[idx])
            else:
                # Add part of front based on crowding distance
                self._calculate_crowding_distance(front, combined_population)
                
                # Sort by crowding distance (descending)
                front.sort(key=lambda idx: combined_population[idx].crowding_distance, reverse=True)
                
                # Add solutions until population is full
                remaining_slots = self.population_size - len(next_generation)
                for i in range(remaining_slots):
                    next_generation.append(combined_population[front[i]])
                break
        
        return next_generation
    
    def _track_generation_stats(self, generation: int):
        """Track statistics for current generation"""
        if not self.population:
            return
        
        # Calculate objectives statistics for ALL solutions (feasible and infeasible)
        all_obj1 = [sol.objectives[0] for sol in self.population]
        all_obj2 = [sol.objectives[1] for sol in self.population]
        
        # Calculate objectives statistics for feasible solutions only
        feasible_obj1 = [sol.objectives[0] for sol in self.population if sol.is_feasible]
        feasible_obj2 = [sol.objectives[1] for sol in self.population if sol.is_feasible]
        
        # Get current Pareto front
        fronts = self._fast_non_dominated_sort(self.population)
        pareto_front_size = len(fronts[0]) if fronts else 0
        
        stats = {
            'generation': generation,
            'pareto_front_size': pareto_front_size,
            'feasible_solutions': len(feasible_obj1),
            'total_solutions': len(self.population),
            'avg_obj1': np.mean(feasible_obj1) if feasible_obj1 else np.mean(all_obj1),
            'min_obj1': np.min(feasible_obj1) if feasible_obj1 else np.min(all_obj1),
            'avg_obj2': np.mean(feasible_obj2) if feasible_obj2 else np.mean(all_obj2),
            'min_obj2': np.min(feasible_obj2) if feasible_obj2 else np.min(all_obj2),
            'avg_all_obj1': np.mean(all_obj1),
            'avg_all_obj2': np.mean(all_obj2),
        }
        
        self.generation_stats.append(stats)
    
    def get_convergence_data(self) -> List[Dict]:
        """Get convergence data for analysis"""
        return self.generation_stats.copy()