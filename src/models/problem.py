"""
Core data models for Home Health Care Routing and Scheduling Problem
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import copy

@dataclass
class Customer:
    """Represents a patient/customer in the HHC problem"""
    id: int
    x: float
    y: float
    demand: int
    ready_time: int
    due_date: int
    service_time: int
    preferred_time: Optional[int] = None  # Patient's preferred visit time
    
    def distance_to(self, other: 'Customer') -> float:
        """Calculate Euclidean distance to another customer"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __repr__(self):
        return f"Customer(id={self.id}, pos=({self.x:.1f},{self.y:.1f}), tw=[{self.ready_time},{self.due_date}])"

@dataclass
class Depot:
    """Represents the depot (home base) for vehicles/caregivers"""
    id: int = 0
    x: float = 0.0
    y: float = 0.0
    ready_time: int = 0
    due_date: int = 1000
    
    def distance_to(self, customer: Customer) -> float:
        """Calculate distance to a customer"""
        return np.sqrt((self.x - customer.x)**2 + (self.y - customer.y)**2)

@dataclass
class Vehicle:
    """Represents a vehicle/caregiver in the system"""
    id: int
    capacity: int
    max_route_time: int = 1000
    
class Route:
    """Represents a route for a single vehicle/caregiver"""
    def __init__(self, vehicle_id: int, depot: Depot, vehicle_capacity: int = 200):
        self.vehicle_id = vehicle_id
        self.depot = depot
        self.vehicle_capacity = vehicle_capacity
        self.customers: List[Customer] = []
        self.arrival_times: List[int] = []
        self.departure_times: List[int] = []
        self.total_distance = 0.0
        self.total_time = 0
        self.total_demand = 0
        self.total_tardiness = 0.0
        self.is_feasible = True
        
    def add_customer(self, customer: Customer, position: int = -1):
        """Add a customer to the route at specified position"""
        if position == -1:
            self.customers.append(customer)
        else:
            self.customers.insert(position, customer)
        self._calculate_route_metrics()
    
    def remove_customer(self, customer_id: int) -> bool:
        """Remove customer from route"""
        for i, customer in enumerate(self.customers):
            if customer.id == customer_id:
                self.customers.pop(i)
                self._calculate_route_metrics()
                return True
        return False
    
    def _calculate_route_metrics(self):
        """Calculate all route metrics including objectives"""
        if not self.customers:
            self.total_distance = 0.0
            self.total_time = 0
            self.total_demand = 0
            self.total_tardiness = 0.0
            self.is_feasible = True
            return
        
        # Reset metrics
        self.total_distance = 0.0
        self.total_time = 0
        self.total_demand = 0
        self.total_tardiness = 0.0
        self.arrival_times = []
        self.departure_times = []
        
        # Calculate from depot to first customer
        current_time = self.depot.ready_time
        current_location = self.depot
        
        for i, customer in enumerate(self.customers):
            # Travel time to customer
            travel_time = current_location.distance_to(customer)
            self.total_distance += travel_time
            current_time += travel_time
            
            # Arrival time
            arrival_time = max(current_time, customer.ready_time)
            self.arrival_times.append(arrival_time)
            
            # Calculate tardiness (delay from preferred time)
            if customer.preferred_time is not None:
                tardiness = max(0, arrival_time - customer.preferred_time)
                self.total_tardiness += tardiness
            
            # Service time
            departure_time = arrival_time + customer.service_time
            self.departure_times.append(departure_time)
            current_time = departure_time
            
            # Update demand
            self.total_demand += customer.demand
            current_location = customer
        
        # Return to depot
        if self.customers:
            return_distance = self.customers[-1].distance_to(self.depot)
            self.total_distance += return_distance
            current_time += return_distance
        
        self.total_time = current_time - self.depot.ready_time
        
        # Check feasibility
        self._check_feasibility()
    
    def _check_feasibility(self):
        """Check if route satisfies all constraints"""
        self.is_feasible = True
        
        # Check capacity constraint
        if self.total_demand > self.vehicle_capacity:
            self.is_feasible = False
            return
        
        # Check time window constraints
        for i, customer in enumerate(self.customers):
            if i < len(self.arrival_times):
                if self.arrival_times[i] > customer.due_date:
                    self.is_feasible = False
                    return
    
    def copy(self) -> 'Route':
        """Create a deep copy of the route"""
        new_route = Route(self.vehicle_id, self.depot, self.vehicle_capacity)
        new_route.customers = copy.deepcopy(self.customers)
        new_route._calculate_route_metrics()
        return new_route
    
    def __len__(self):
        return len(self.customers)
    
    def __repr__(self):
        customer_ids = [c.id for c in self.customers]
        return f"Route(vehicle={self.vehicle_id}, customers={customer_ids}, dist={self.total_distance:.2f}, time={self.total_time}, tardiness={self.total_tardiness:.2f})"

class Solution:
    """Represents a complete solution to the HHC problem"""
    def __init__(self, depot: Depot, vehicles: List[Vehicle]):
        self.depot = depot
        self.vehicles = vehicles
        self.routes: List[Route] = []
        self.objectives = [0.0, 0.0]  # [total_service_time, total_tardiness]
        self.is_feasible = True
        self.dominates_count = 0
        self.dominated_by = set()
        self.crowding_distance = 0.0
        
        # Initialize empty routes
        for vehicle in vehicles:
            self.routes.append(Route(vehicle.id, depot, vehicle.capacity))
    
    def add_customer_to_route(self, customer: Customer, route_index: int, position: int = -1):
        """Add customer to specified route"""
        if 0 <= route_index < len(self.routes):
            self.routes[route_index].add_customer(customer, position)
            self._calculate_objectives()
    
    def remove_customer(self, customer_id: int) -> bool:
        """Remove customer from solution"""
        for route in self.routes:
            if route.remove_customer(customer_id):
                self._calculate_objectives()
                return True
        return False
    
    def _calculate_objectives(self):
        """Calculate the two objectives: total service time and total tardiness"""
        total_service_time = 0.0
        total_tardiness = 0.0
        self.is_feasible = True
        
        for route in self.routes:
            if route.customers:  # Only count non-empty routes
                total_service_time += route.total_time
                total_tardiness += route.total_tardiness
                if not route.is_feasible:
                    self.is_feasible = False
        
        self.objectives = [total_service_time, total_tardiness]
    
    def dominates(self, other: 'Solution') -> bool:
        """Check if this solution dominates another solution"""
        if not self.is_feasible and other.is_feasible:
            return False
        if self.is_feasible and not other.is_feasible:
            return True
            
        # For minimization: solution A dominates B if A is better or equal in all objectives
        # and strictly better in at least one objective
        better_in_any = False
        for i in range(len(self.objectives)):
            if self.objectives[i] > other.objectives[i]:
                return False
            if self.objectives[i] < other.objectives[i]:
                better_in_any = True
        
        return better_in_any
    
    def copy(self) -> 'Solution':
        """Create a deep copy of the solution"""
        new_solution = Solution(self.depot, self.vehicles)
        new_solution.routes = [route.copy() for route in self.routes]
        new_solution._calculate_objectives()
        return new_solution
    
    def get_all_customers(self) -> List[Customer]:
        """Get all customers in the solution"""
        all_customers = []
        for route in self.routes:
            all_customers.extend(route.customers)
        return all_customers
    
    def get_route_for_customer(self, customer_id: int) -> Optional[Tuple[int, int]]:
        """Get route index and position for a customer"""
        for route_idx, route in enumerate(self.routes):
            for pos, customer in enumerate(route.customers):
                if customer.id == customer_id:
                    return route_idx, pos
        return None
    
    def __repr__(self):
        return f"Solution(objectives={[f'{obj:.2f}' for obj in self.objectives]}, feasible={self.is_feasible}, routes={len([r for r in self.routes if r.customers])})"