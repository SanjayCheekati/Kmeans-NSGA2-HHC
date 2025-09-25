"""
Solomon dataset parser for VRPTW instances
"""
import numpy as np
from typing import List, Tuple
from ..models.problem import Customer, Depot, Vehicle
import os

class SolomonParser:
    """Parser for Solomon VRPTW benchmark instances"""
    
    @staticmethod
    def parse_instance(file_path: str) -> Tuple[Depot, List[Customer], List[Vehicle]]:
        """
        Parse a Solomon instance file
        Returns: (depot, customers, vehicles)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Instance file not found: {file_path}")
        
        print(f"📁 Parsing Solomon instance: {os.path.basename(file_path)}")
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Remove empty lines and strip whitespace
        lines = [line.strip() for line in lines if line.strip()]
        
        # Parse header information
        instance_name = lines[0] if lines else "Unknown"
        
        # Find vehicle information
        vehicle_count = 25  # Default
        vehicle_capacity = 200  # Default
        
        for i, line in enumerate(lines):
            if 'VEHICLE' in line:
                # Look for the numbers in subsequent lines
                for j in range(i+1, min(i+10, len(lines))):
                    parts = lines[j].split()
                    if len(parts) >= 2:
                        try:
                            # Try to parse as numbers
                            vc = int(parts[0])
                            cap = int(parts[1])
                            if vc > 0 and cap > 0:  # Valid values
                                vehicle_count = vc
                                vehicle_capacity = cap
                                break
                        except ValueError:
                            continue
                break
        
        # Find customer data section
        customers = []
        depot = None
        customer_section_started = False
        
        for i, line in enumerate(lines):
            # Look for customer section indicators
            if any(keyword in line.upper() for keyword in ['CUSTOMER', 'CUST NO.', 'XCOORD']):
                customer_section_started = True
                continue
                
            if customer_section_started and line:
                # Try to parse as customer data
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        cust_id = int(parts[0])
                        x_coord = float(parts[1])
                        y_coord = float(parts[2])
                        demand = int(parts[3])
                        ready_time = int(parts[4])
                        due_date = int(parts[5])
                        service_time = int(parts[6])
                        
                        if cust_id == 0:  # Depot
                            depot = Depot(
                                id=cust_id,
                                x=x_coord,
                                y=y_coord,
                                ready_time=ready_time,
                                due_date=due_date
                            )
                        else:  # Customer
                            # Set preferred time as middle of time window
                            preferred_time = ready_time + (due_date - ready_time) // 2
                            
                            customer = Customer(
                                id=cust_id,
                                x=x_coord,
                                y=y_coord,
                                demand=demand,
                                ready_time=ready_time,
                                due_date=due_date,
                                service_time=service_time,
                                preferred_time=preferred_time
                            )
                            customers.append(customer)
                    
                    except (ValueError, IndexError):
                        continue
        
        if depot is None:
            raise ValueError("Depot not found in instance file")
        
        # Create vehicles
        vehicles = []
        for i in range(vehicle_count):
            vehicles.append(Vehicle(id=i, capacity=vehicle_capacity))
        
        print(f"✅ Parsed Solomon instance '{instance_name}':")
        print(f"   Depot: ({depot.x}, {depot.y}), Time: [{depot.ready_time}-{depot.due_date}]")
        print(f"   Customers: {len(customers)}")
        print(f"   Vehicles: {vehicle_count} (capacity: {vehicle_capacity})")
        
        return depot, customers, vehicles
    
    @staticmethod
    def get_instance_characteristics(file_path: str) -> dict:
        """Get characteristics of a Solomon instance without full parsing"""
        try:
            depot, customers, vehicles = SolomonParser.parse_instance(file_path)
            
            # Calculate statistics
            demands = [c.demand for c in customers]
            x_coords = [c.x for c in customers]
            y_coords = [c.y for c in customers]
            time_windows = [(c.due_date - c.ready_time) for c in customers]
            
            return {
                'name': os.path.basename(file_path).replace('.txt', ''),
                'customers': len(customers),
                'vehicles': len(vehicles),
                'vehicle_capacity': vehicles[0].capacity if vehicles else 0,
                'depot_location': (depot.x, depot.y),
                'total_demand': sum(demands),
                'avg_demand': sum(demands) / len(demands) if demands else 0,
                'spatial_spread': {
                    'x_range': (min(x_coords), max(x_coords)),
                    'y_range': (min(y_coords), max(y_coords))
                },
                'time_horizon': depot.due_date - depot.ready_time,
                'avg_time_window': sum(time_windows) / len(time_windows) if time_windows else 0
            }
        except Exception as e:
            return {'error': str(e)}