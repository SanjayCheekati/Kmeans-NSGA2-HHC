"""
Solomon Benchmark Testing Suite for Enhanced Hybrid Algorithm
Tests the enhanced hybrid K-means + NSGA-II algorithm on real Solomon benchmark datasets
"""
import os
import sys
import time
import json
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.problem import Solution, Customer, Depot, Vehicle
from src.utils.solomon_parser import SolomonParser
from src.algorithms.nsga2 import NSGAII
from src.algorithms.enhanced_hybrid import EnhancedHybridKmeansNSGAII
from src.utils.visualization import HHCVisualizer
from solomon_dataset_manager import SolomonDatasetManager

class SolomonBenchmarkTester:
    """Solomon benchmark testing suite"""
    
    def __init__(self, results_dir: str = "solomon_results"):
        self.results_dir = results_dir
        self.dataset_manager = SolomonDatasetManager()
        self.visualizer = HHCVisualizer()
        
        # Setup directories
        os.makedirs(results_dir, exist_ok=True)
        
        # Clean up old results
        self.dataset_manager.cleanup_old_results()
        
    def run_comprehensive_benchmark(self, max_customers: int = 15) -> Dict[str, Any]:
        """Run comprehensive benchmark on Solomon instances"""
        
        print("="*100)
        print("🚀 SOLOMON BENCHMARK TESTING SUITE")
        print("="*100)
        print("✨ Enhanced Hybrid K-means + NSGA-II vs Standard NSGA-II")
        print("✨ Testing on Real Solomon VRPTW Benchmark Datasets")
        print("="*100)
        
        # Setup benchmark datasets
        print("\n📁 Setting up Solomon Benchmark Datasets")
        available_instances = self.dataset_manager.create_solomon_instances()
        test_instances = self.dataset_manager.get_test_suite()
        
        print(f"Available instances: {available_instances}")
        print(f"Test suite: {test_instances}")
        
        # Comprehensive results storage
        benchmark_results = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'max_customers': max_customers,
                'test_instances': test_instances
            },
            'instance_results': {},
            'summary_statistics': {}
        }
        
        # Test each instance
        for instance_file in test_instances:
            print(f"\n{'='*80}")
            print(f"🔬 TESTING INSTANCE: {instance_file}")
            print(f"{'='*80}")
            
            instance_path = self.dataset_manager.get_instance_path(instance_file)
            instance_name = instance_file.replace('.txt', '')
            
            try:
                # Load instance
                depot, customers, vehicles = SolomonParser.parse_instance(instance_path)
                instance_chars = SolomonParser.get_instance_characteristics(instance_path)
                
                # Limit customers for manageable computation
                if len(customers) > max_customers:
                    customers = customers[:max_customers]
                    print(f"   Limited to {max_customers} customers for testing")
                
                # Run algorithms
                instance_results = self._test_instance(
                    instance_name, depot, customers, vehicles
                )
                
                # Add instance characteristics
                instance_results['characteristics'] = instance_chars
                benchmark_results['instance_results'][instance_name] = instance_results
                
                # Create visualizations
                self._create_instance_visualizations(
                    instance_name, depot, customers, instance_results
                )
                
            except Exception as e:
                print(f"❌ Error testing instance {instance_file}: {e}")
                benchmark_results['instance_results'][instance_name] = {
                    'error': str(e)
                }
        
        # Generate summary statistics
        summary = self._generate_summary_statistics(benchmark_results['instance_results'])
        benchmark_results['summary_statistics'] = summary
        
        # Save comprehensive results
        results_file = os.path.join(self.results_dir, 'solomon_benchmark_results.json')
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_summary_report(benchmark_results)
        
        print(f"\n{'='*100}")
        print("✨ SOLOMON BENCHMARK TESTING COMPLETED!")
        print(f"📁 Results saved to: {self.results_dir}/")
        print(f"📊 Comprehensive report: solomon_benchmark_results.json")
        print(f"📋 Summary report: benchmark_summary.md")
        print("="*100)
        
        return benchmark_results
    
    def _test_instance(self, instance_name: str, depot: Depot, 
                      customers: List[Customer], vehicles: List[Vehicle]) -> Dict[str, Any]:
        """Test algorithms on a specific instance"""
        
        results = {
            'instance_info': {
                'name': instance_name,
                'customers': len(customers),
                'vehicles': len(vehicles),
                'vehicle_capacity': vehicles[0].capacity if vehicles else 0
            },
            'algorithms': {}
        }
        
        # 1. Standard NSGA-II
        print(f"\n🧬 Running Standard NSGA-II on {instance_name}")
        start_time = time.time()
        
        standard_nsga2 = NSGAII(
            population_size=50,
            max_generations=100,
            crossover_probability=0.85,
            mutation_probability=0.15,
            random_seed=42
        )
        
        try:
            standard_pareto = standard_nsga2.solve(depot, customers, vehicles)
            standard_time = time.time() - start_time
            standard_convergence = standard_nsga2.get_convergence_data()
            
            results['algorithms']['standard_nsga2'] = {
                'execution_time': standard_time,
                'pareto_front_size': len(standard_pareto),
                'solutions': [{'objectives': sol.objectives, 'feasible': sol.is_feasible} 
                            for sol in standard_pareto],
                'convergence_data': standard_convergence
            }
            
        except Exception as e:
            results['algorithms']['standard_nsga2'] = {'error': str(e)}
        
        # 2. Enhanced Hybrid Algorithm  
        print(f"\n🔬 Running Enhanced Hybrid K-means + NSGA-II on {instance_name}")
        start_time = time.time()
        
        enhanced_hybrid = EnhancedHybridKmeansNSGAII(
            n_clusters=None,  # Auto-determine
            population_size_per_cluster=30,
            max_generations=100,
            crossover_probability=0.85,
            mutation_probability=0.15,
            combination_strategies=['direct_merge', 'iterative_construction', 'greedy_merge'],
            random_seed=42
        )
        
        try:
            enhanced_pareto = enhanced_hybrid.solve(depot, customers, vehicles)
            enhanced_time = time.time() - start_time
            enhanced_details = enhanced_hybrid.get_detailed_results()
            
            results['algorithms']['enhanced_hybrid'] = {
                'execution_time': enhanced_time,
                'pareto_front_size': len(enhanced_pareto),
                'solutions': [{'objectives': sol.objectives, 'feasible': sol.is_feasible} 
                            for sol in enhanced_pareto],
                'detailed_results': enhanced_details
            }
            
        except Exception as e:
            results['algorithms']['enhanced_hybrid'] = {'error': str(e)}
        
        return results
    
    def _create_instance_visualizations(self, instance_name: str, depot: Depot, 
                                      customers: List[Customer], results: Dict[str, Any]):
        """Create visualizations for an instance"""
        
        instance_dir = os.path.join(self.results_dir, instance_name)
        os.makedirs(instance_dir, exist_ok=True)
        
        # 1. Instance map
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        
        x_coords = [c.x for c in customers]
        y_coords = [c.y for c in customers]
        demands = [c.demand for c in customers]
        
        plt.scatter(x_coords, y_coords, c=demands, cmap='viridis', s=100, alpha=0.7)
        plt.colorbar(label='Customer Demand')
        plt.scatter(depot.x, depot.y, c='red', s=300, marker='s', label='Depot', edgecolors='black')
        
        # Add customer info
        for customer in customers:
            plt.annotate(f'{customer.id}\\n[{customer.ready_time}-{customer.due_date}]', 
                        (customer.x, customer.y),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'Solomon Instance {instance_name}: {len(customers)} Customers')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{instance_dir}/{instance_name}_instance_map.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Algorithm comparison (if both successful)
        if ('standard_nsga2' in results['algorithms'] and 
            'enhanced_hybrid' in results['algorithms'] and
            'error' not in results['algorithms']['standard_nsga2'] and
            'error' not in results['algorithms']['enhanced_hybrid']):
            
            comparison_data = {}
            
            for algo_name, algo_results in results['algorithms'].items():
                if 'solutions' in algo_results:
                    valid_objectives = [sol['objectives'] for sol in algo_results['solutions'] 
                                     if sol['objectives'][0] > 0 or sol['objectives'][1] > 0]
                    if valid_objectives:
                        comparison_data[algo_name] = {
                            'objectives': valid_objectives,
                            'pareto_front_size': len(valid_objectives)
                        }
            
            if comparison_data:
                self.visualizer.plot_comparison_pareto_fronts(
                    comparison_data,
                    title=f"Algorithm Comparison: {instance_name}",
                    save_path=f"{instance_dir}/{instance_name}_algorithm_comparison.png",
                    show=False
                )
    
    def _generate_summary_statistics(self, instance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics across all instances"""
        
        summary = {
            'total_instances': len(instance_results),
            'successful_tests': 0,
            'algorithm_performance': {},
            'instance_characteristics': {}
        }
        
        algorithm_stats = {'standard_nsga2': [], 'enhanced_hybrid': []}
        
        for instance_name, results in instance_results.items():
            if 'error' not in results:
                summary['successful_tests'] += 1
                
                for algo_name in algorithm_stats.keys():
                    if algo_name in results.get('algorithms', {}) and 'error' not in results['algorithms'][algo_name]:
                        algo_result = results['algorithms'][algo_name]
                        algorithm_stats[algo_name].append({
                            'instance': instance_name,
                            'execution_time': algo_result.get('execution_time', 0),
                            'pareto_front_size': algo_result.get('pareto_front_size', 0),
                            'feasible_solutions': sum(1 for sol in algo_result.get('solutions', []) 
                                                    if sol.get('feasible', False))
                        })
        
        # Calculate performance statistics
        for algo_name, stats in algorithm_stats.items():
            if stats:
                execution_times = [s['execution_time'] for s in stats]
                pareto_sizes = [s['pareto_front_size'] for s in stats]
                feasible_counts = [s['feasible_solutions'] for s in stats]
                
                summary['algorithm_performance'][algo_name] = {
                    'avg_execution_time': sum(execution_times) / len(execution_times),
                    'avg_pareto_front_size': sum(pareto_sizes) / len(pareto_sizes),
                    'avg_feasible_solutions': sum(feasible_counts) / len(feasible_counts),
                    'success_rate': len(stats) / summary['total_instances']
                }
        
        return summary
    
    def _generate_summary_report(self, benchmark_results: Dict[str, Any]):
        """Generate markdown summary report"""
        
        report_path = os.path.join(self.results_dir, 'benchmark_summary.md')
        
        with open(report_path, 'w') as f:
            f.write("# Solomon Benchmark Testing Results\\n\\n")
            f.write(f"**Test Date**: {benchmark_results['experiment_info']['timestamp']}\\n")
            f.write(f"**Max Customers**: {benchmark_results['experiment_info']['max_customers']}\\n")
            f.write(f"**Test Instances**: {', '.join(benchmark_results['experiment_info']['test_instances'])}\\n\\n")
            
            # Summary statistics
            summary = benchmark_results['summary_statistics']
            f.write("## Summary Statistics\\n\\n")
            f.write(f"- **Total Instances Tested**: {summary['total_instances']}\\n")
            f.write(f"- **Successful Tests**: {summary['successful_tests']}\\n\\n")
            
            # Algorithm performance
            f.write("## Algorithm Performance\\n\\n")
            for algo_name, stats in summary['algorithm_performance'].items():
                f.write(f"### {algo_name.replace('_', ' ').title()}\\n")
                f.write(f"- Average Execution Time: {stats['avg_execution_time']:.2f}s\\n")
                f.write(f"- Average Pareto Front Size: {stats['avg_pareto_front_size']:.1f}\\n")
                f.write(f"- Average Feasible Solutions: {stats['avg_feasible_solutions']:.1f}\\n")
                f.write(f"- Success Rate: {stats['success_rate']:.1%}\\n\\n")
            
            # Instance details
            f.write("## Instance Details\\n\\n")
            for instance_name, results in benchmark_results['instance_results'].items():
                if 'error' not in results:
                    f.write(f"### {instance_name}\\n")
                    info = results['instance_info']
                    f.write(f"- Customers: {info['customers']}\\n")
                    f.write(f"- Vehicles: {info['vehicles']} (capacity: {info['vehicle_capacity']})\\n")
                    
                    for algo_name, algo_results in results['algorithms'].items():
                        if 'error' not in algo_results:
                            f.write(f"- {algo_name}: {algo_results['execution_time']:.2f}s, ")
                            f.write(f"{algo_results['pareto_front_size']} solutions\\n")
                    f.write("\\n")

def main():
    """Main function to run Solomon benchmark testing"""
    
    # Create tester
    tester = SolomonBenchmarkTester()
    
    # Run comprehensive benchmark
    results = tester.run_comprehensive_benchmark(max_customers=20)
    
    # Print summary
    print("\\n📊 BENCHMARK SUMMARY:")
    summary = results['summary_statistics']
    print(f"Successful tests: {summary['successful_tests']}/{summary['total_instances']}")
    
    for algo_name, stats in summary['algorithm_performance'].items():
        print(f"{algo_name}: {stats['avg_execution_time']:.2f}s avg, {stats['success_rate']:.1%} success rate")

if __name__ == "__main__":
    main()