"""
Visualization utilities for Home Health Care Routing and Scheduling results
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from ..models.problem import Solution, Customer, Depot
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

class HHCVisualizer:
    """Comprehensive visualization for HHC optimization results"""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """Initialize visualizer with plotting style"""
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Color palettes
        self.route_colors = plt.cm.Set3(np.linspace(0, 1, 20))
        self.cluster_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
    def plot_solution_routes(self, solution: Solution, depot: Depot, 
                           title: str = "Vehicle Routes", 
                           save_path: str = None, show: bool = True,
                           figsize: Tuple[int, int] = (14, 10)):
        """Plot vehicle routes for a single solution"""
        
        plt.figure(figsize=figsize)
        
        # Plot depot
        plt.scatter(depot.x, depot.y, c='red', s=300, marker='s', 
                   label='Depot', edgecolors='black', linewidth=2, zorder=5)
        plt.annotate('DEPOT', (depot.x, depot.y), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=12, fontweight='bold', color='red')
        
        # Plot routes
        route_info = []
        for route_idx, route in enumerate(solution.routes):
            if not route.customers:
                continue
            
            color = self.route_colors[route_idx % len(self.route_colors)]
            
            # Plot route path
            x_coords = [depot.x]
            y_coords = [depot.y]
            
            for customer in route.customers:
                x_coords.append(customer.x)
                y_coords.append(customer.y)
            
            x_coords.append(depot.x)  # Return to depot
            y_coords.append(depot.y)
            
            # Draw route lines
            plt.plot(x_coords, y_coords, color=color, linewidth=2, alpha=0.7)
            
            # Plot customers
            customer_x = [c.x for c in route.customers]
            customer_y = [c.y for c in route.customers]
            
            plt.scatter(customer_x, customer_y, c=[color], s=100, 
                       label=f'Route {route_idx+1} ({len(route.customers)} customers)',
                       edgecolors='black', linewidth=1, alpha=0.8, zorder=3)
            
            # Add customer IDs
            for customer in route.customers:
                plt.annotate(str(customer.id), (customer.x, customer.y),
                           xytext=(2, 2), textcoords='offset points',
                           fontsize=8, fontweight='bold')
            
            # Collect route information
            route_info.append({
                'Route': route_idx + 1,
                'Customers': len(route.customers),
                'Distance': f"{route.total_distance:.2f}",
                'Time': route.total_time,
                'Tardiness': f"{route.total_tardiness:.2f}",
                'Demand': route.total_demand
            })
        
        # Formatting
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.title(f'{title}\nObjectives: Service Time={solution.objectives[0]:.2f}, Tardiness={solution.objectives[1]:.2f}', 
                 fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Add route information as text
        if route_info:
            info_text = "Route Summary:\n"
            for info in route_info:
                info_text += f"R{info['Route']}: {info['Customers']}c, D={info['Distance']}, T={info['Tardiness']}\n"
            
            plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Route visualization saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_pareto_front(self, solutions: List[Solution], 
                         title: str = "Pareto Front",
                         save_path: str = None, show: bool = True,
                         figsize: Tuple[int, int] = (12, 8)):
        """Plot Pareto front of solutions"""
        
        if not solutions:
            print("No solutions to plot")
            return
        
        plt.figure(figsize=figsize)
        
        # Extract objectives
        obj1_values = [sol.objectives[0] for sol in solutions]
        obj2_values = [sol.objectives[1] for sol in solutions]
        
        # Plot Pareto front
        plt.scatter(obj1_values, obj2_values, c='blue', s=100, alpha=0.7, 
                   edgecolors='black', linewidth=1)
        
        # Connect points to show Pareto front
        sorted_indices = np.argsort(obj1_values)
        sorted_obj1 = [obj1_values[i] for i in sorted_indices]
        sorted_obj2 = [obj2_values[i] for i in sorted_indices]
        
        plt.plot(sorted_obj1, sorted_obj2, 'r--', alpha=0.5, linewidth=1)
        
        # Add solution indices
        for i, (obj1, obj2) in enumerate(zip(obj1_values, obj2_values)):
            plt.annotate(str(i+1), (obj1, obj2), xytext=(3, 3), 
                        textcoords='offset points', fontsize=8)
        
        # Formatting
        plt.xlabel('Total Service Time', fontsize=12)
        plt.ylabel('Total Tardiness', fontsize=12)
        plt.title(f'{title} ({len(solutions)} solutions)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f"Statistics:\n"
        stats_text += f"Solutions: {len(solutions)}\n"
        stats_text += f"Service Time: [{min(obj1_values):.2f}, {max(obj1_values):.2f}]\n"
        stats_text += f"Tardiness: [{min(obj2_values):.2f}, {max(obj2_values):.2f}]\n"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Pareto front plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_convergence(self, convergence_data: List[Dict],
                        title: str = "Algorithm Convergence",
                        save_path: str = None, show: bool = True,
                        figsize: Tuple[int, int] = (14, 10)):
        """Plot algorithm convergence over generations"""
        
        if not convergence_data:
            print("No convergence data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        generations = [data['generation'] for data in convergence_data]
        
        # Plot 1: Pareto front size evolution
        pareto_sizes = [data.get('pareto_front_size', 0) for data in convergence_data]
        axes[0, 0].plot(generations, pareto_sizes, 'b-o', markersize=4)
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Pareto Front Size')
        axes[0, 0].set_title('Pareto Front Size Evolution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Average objectives evolution
        avg_obj1 = [data.get('avg_obj1', 0) for data in convergence_data]
        avg_obj2 = [data.get('avg_obj2', 0) for data in convergence_data]
        
        ax2 = axes[0, 1]
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(generations, avg_obj1, 'g-o', markersize=3, label='Avg Service Time')
        line2 = ax2_twin.plot(generations, avg_obj2, 'r-s', markersize=3, label='Avg Tardiness')
        
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Average Service Time', color='g')
        ax2_twin.set_ylabel('Average Tardiness', color='r')
        ax2.set_title('Average Objectives Evolution')
        ax2.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper right')
        
        # Plot 3: Best objectives evolution
        min_obj1 = [data.get('min_obj1', 0) for data in convergence_data]
        min_obj2 = [data.get('min_obj2', 0) for data in convergence_data]
        
        ax3 = axes[1, 0]
        ax3_twin = ax3.twinx()
        
        line3 = ax3.plot(generations, min_obj1, 'g-o', markersize=3, label='Best Service Time')
        line4 = ax3_twin.plot(generations, min_obj2, 'r-s', markersize=3, label='Best Tardiness')
        
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Best Service Time', color='g')
        ax3_twin.set_ylabel('Best Tardiness', color='r')
        ax3.set_title('Best Objectives Evolution')
        ax3.grid(True, alpha=0.3)
        
        lines2 = line3 + line4
        labels2 = [l.get_label() for l in lines2]
        ax3.legend(lines2, labels2, loc='upper right')
        
        # Plot 4: Feasible solutions
        feasible_solutions = [data.get('feasible_solutions', 0) for data in convergence_data]
        axes[1, 1].plot(generations, feasible_solutions, 'm-o', markersize=4)
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('Feasible Solutions')
        axes[1, 1].set_title('Feasible Solutions Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_comparison_pareto_fronts(self, comparison_data: Dict,
                                    title: str = "Algorithm Comparison",
                                    save_path: str = None, show: bool = True,
                                    figsize: Tuple[int, int] = (12, 8)):
        """Compare Pareto fronts from different algorithms"""
        
        plt.figure(figsize=figsize)
        
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        markers = ['o', 's', '^', 'D', 'v']
        
        for i, (algorithm_name, data) in enumerate(comparison_data.items()):
            objectives = data.get('objectives', [])
            if not objectives:
                continue
            
            obj1_values = [obj[0] for obj in objectives]
            obj2_values = [obj[1] for obj in objectives]
            
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            plt.scatter(obj1_values, obj2_values, c=color, s=80, alpha=0.7,
                       marker=marker, label=f'{algorithm_name} ({len(objectives)})',
                       edgecolors='black', linewidth=0.5)
            
            # Connect points for Pareto front
            if len(objectives) > 1:
                sorted_indices = np.argsort(obj1_values)
                sorted_obj1 = [obj1_values[i] for i in sorted_indices]
                sorted_obj2 = [obj2_values[i] for i in sorted_indices]
                plt.plot(sorted_obj1, sorted_obj2, color=color, alpha=0.3, linestyle='--')
        
        plt.xlabel('Total Service Time', fontsize=12)
        plt.ylabel('Total Tardiness', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_interactive_solution_plot(self, solution: Solution, depot: Depot,
                                       title: str = "Interactive Route Visualization"):
        """Create interactive plot using Plotly"""
        
        fig = go.Figure()
        
        # Add depot
        fig.add_trace(go.Scatter(
            x=[depot.x], y=[depot.y],
            mode='markers',
            marker=dict(size=20, symbol='square', color='red'),
            name='Depot',
            text=['Depot'],
            hovertemplate="<b>Depot</b><br>X: %{x}<br>Y: %{y}<extra></extra>"
        ))
        
        # Add routes
        for route_idx, route in enumerate(solution.routes):
            if not route.customers:
                continue
            
            # Route path
            x_coords = [depot.x]
            y_coords = [depot.y]
            
            for customer in route.customers:
                x_coords.append(customer.x)
                y_coords.append(customer.y)
            
            x_coords.append(depot.x)
            y_coords.append(depot.y)
            
            # Add route line
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                mode='lines',
                line=dict(width=2),
                name=f'Route {route_idx+1}',
                showlegend=False
            ))
            
            # Add customers
            customer_x = [c.x for c in route.customers]
            customer_y = [c.y for c in route.customers]
            customer_info = [
                f"Customer {c.id}<br>Demand: {c.demand}<br>Time Window: [{c.ready_time}, {c.due_date}]<br>Service Time: {c.service_time}"
                for c in route.customers
            ]
            
            fig.add_trace(go.Scatter(
                x=customer_x, y=customer_y,
                mode='markers+text',
                marker=dict(size=12),
                text=[str(c.id) for c in route.customers],
                textposition="middle center",
                name=f'Route {route_idx+1} ({len(route.customers)} customers)',
                hovertemplate="%{text}<extra></extra>",
                customdata=customer_info
            ))
        
        fig.update_layout(
            title=f'{title}<br>Service Time: {solution.objectives[0]:.2f}, Tardiness: {solution.objectives[1]:.2f}',
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            hovermode='closest'
        )
        
        return fig
    
    def plot_solution_metrics(self, solutions: List[Solution],
                            save_path: str = None, show: bool = True,
                            figsize: Tuple[int, int] = (15, 10)):
        """Plot detailed metrics for solutions"""
        
        if not solutions:
            print("No solutions to analyze")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Extract metrics
        service_times = [sol.objectives[0] for sol in solutions]
        tardiness_values = [sol.objectives[1] for sol in solutions]
        total_distances = [sum(route.total_distance for route in sol.routes) for sol in solutions]
        num_routes = [len([r for r in sol.routes if r.customers]) for sol in solutions]
        avg_route_length = [np.mean([len(r.customers) for r in sol.routes if r.customers]) 
                           if any(r.customers for r in sol.routes) else 0 for sol in solutions]
        
        # Plot 1: Service Time Distribution
        axes[0, 0].hist(service_times, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_xlabel('Service Time')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Service Time Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Tardiness Distribution
        axes[0, 1].hist(tardiness_values, bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].set_xlabel('Tardiness')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Tardiness Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Total Distance vs Service Time
        axes[0, 2].scatter(total_distances, service_times, alpha=0.6)
        axes[0, 2].set_xlabel('Total Distance')
        axes[0, 2].set_ylabel('Service Time')
        axes[0, 2].set_title('Distance vs Service Time')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Number of Routes Distribution
        route_counts = {}
        for num in num_routes:
            route_counts[num] = route_counts.get(num, 0) + 1
        
        axes[1, 0].bar(route_counts.keys(), route_counts.values(), alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_xlabel('Number of Routes')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Number of Routes Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Average Route Length
        axes[1, 1].hist([x for x in avg_route_length if x > 0], bins=15, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_xlabel('Average Customers per Route')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Average Route Length Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Objectives Correlation
        axes[1, 2].scatter(service_times, tardiness_values, alpha=0.6, color='orange')
        axes[1, 2].set_xlabel('Service Time')
        axes[1, 2].set_ylabel('Tardiness')
        axes[1, 2].set_title('Service Time vs Tardiness')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(service_times, tardiness_values)[0, 1]
        axes[1, 2].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=axes[1, 2].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('Solution Metrics Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()