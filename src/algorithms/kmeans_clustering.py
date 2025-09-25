"""
K-means clustering implementation for customer grouping in HHC problem
"""
import numpy as np
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from ..models.problem import Customer, Depot
import random

class CustomerClustering:
    """K-means clustering for customers in HHC problem"""
    
    def __init__(self, n_clusters: int, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.clusters: Dict[int, List[Customer]] = {}
        self.cluster_centers = None
        
    def fit_customers(self, customers: List[Customer], depot: Depot) -> Dict[int, List[Customer]]:
        """
        Cluster customers using K-means algorithm
        Returns dictionary mapping cluster_id -> list of customers
        """
        if len(customers) == 0:
            return {}
        
        if len(customers) <= self.n_clusters:
            # If fewer customers than clusters, assign each to its own cluster
            clusters = {}
            for i, customer in enumerate(customers):
                clusters[i] = [customer]
            return clusters
        
        # Extract coordinates for clustering
        coordinates = np.array([[customer.x, customer.y] for customer in customers])
        
        # Add depot to coordinates for better clustering (weighted)
        depot_weight = max(1, len(customers) // (self.n_clusters * 2))
        depot_coords = np.array([[depot.x, depot.y] for _ in range(depot_weight)])
        all_coords = np.vstack([coordinates, depot_coords])
        
        # Perform K-means clustering
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        # Fit only on customer coordinates (not depot)
        labels = self.kmeans.fit_predict(coordinates)
        self.cluster_centers = self.kmeans.cluster_centers_
        
        # Group customers by cluster
        self.clusters = {}
        for i in range(self.n_clusters):
            self.clusters[i] = []
        
        for customer, label in zip(customers, labels):
            self.clusters[label].append(customer)
        
        # Balance clusters if some are empty
        self._balance_clusters()
        
        print(f"K-means clustering results:")
        for cluster_id, cluster_customers in self.clusters.items():
            print(f"  Cluster {cluster_id}: {len(cluster_customers)} customers")
        
        return self.clusters
    
    def _balance_clusters(self):
        """Balance clusters by redistributing customers from large to empty clusters"""
        empty_clusters = [cluster_id for cluster_id, customers in self.clusters.items() if len(customers) == 0]
        
        if not empty_clusters:
            return
        
        # Find clusters with more than one customer
        redistributable_clusters = [(cluster_id, customers) for cluster_id, customers in self.clusters.items() 
                                  if len(customers) > 1]
        
        if not redistributable_clusters:
            return
        
        # Sort by size descending
        redistributable_clusters.sort(key=lambda x: len(x[1]), reverse=True)
        
        for empty_cluster_id in empty_clusters:
            if not redistributable_clusters:
                break
            
            # Take a customer from the largest cluster
            source_cluster_id, source_customers = redistributable_clusters[0]
            if len(source_customers) > 1:
                # Move last customer
                customer_to_move = source_customers.pop()
                self.clusters[empty_cluster_id].append(customer_to_move)
                
                # Update redistributable list
                if len(source_customers) <= 1:
                    redistributable_clusters.pop(0)
                else:
                    redistributable_clusters.sort(key=lambda x: len(x[1]), reverse=True)
    
    def get_cluster_statistics(self) -> Dict:
        """Get statistics about the clustering"""
        if not self.clusters:
            return {}
        
        cluster_sizes = [len(customers) for customers in self.clusters.values()]
        total_customers = sum(cluster_sizes)
        
        stats = {
            'n_clusters': len(self.clusters),
            'total_customers': total_customers,
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'avg_cluster_size': total_customers / len(self.clusters) if len(self.clusters) > 0 else 0,
            'empty_clusters': sum(1 for size in cluster_sizes if size == 0),
            'cluster_sizes': cluster_sizes
        }
        
        # Calculate balance ratio (how evenly distributed)
        if stats['max_cluster_size'] > 0:
            stats['balance_ratio'] = stats['min_cluster_size'] / stats['max_cluster_size']
        else:
            stats['balance_ratio'] = 0
        
        return stats
    
    def plot_clusters(self, depot: Depot, title: str = "Customer Clusters", save_path: str = None, show: bool = True):
        """Plot the clustering results"""
        if not self.clusters:
            return
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_clusters))
        
        # Plot clusters
        for cluster_id, customers in self.clusters.items():
            if customers:
                x_coords = [c.x for c in customers]
                y_coords = [c.y for c in customers]
                plt.scatter(x_coords, y_coords, c=[colors[cluster_id]], 
                           label=f'Cluster {cluster_id} ({len(customers)})', s=100, alpha=0.7)
                
                # Add customer IDs
                for customer in customers:
                    plt.annotate(f'{customer.id}', (customer.x, customer.y),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot depot
        plt.scatter(depot.x, depot.y, c='red', s=300, marker='s', label='Depot', edgecolors='black')
        
        # Plot cluster centers if available
        if self.cluster_centers is not None:
            plt.scatter(self.cluster_centers[:, 0], self.cluster_centers[:, 1], 
                       c='black', s=200, marker='x', linewidths=3, label='Cluster Centers')
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_optimal_clusters(self, customers: List[Customer], depot: Depot, 
                           max_clusters: int = None, min_customers_per_cluster: int = 2) -> int:
        """
        Determine optimal number of clusters using elbow method
        """
        if not customers:
            return 1
        
        n_customers = len(customers)
        if max_clusters is None:
            max_clusters = min(8, n_customers // min_customers_per_cluster)
        
        if max_clusters < 2:
            return 1
        
        coordinates = np.array([[customer.x, customer.y] for customer in customers])
        
        # Test different numbers of clusters
        inertias = []
        cluster_range = range(1, min(max_clusters + 1, n_customers))
        
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(coordinates)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point (simple implementation)
        if len(inertias) < 3:
            return max(1, len(inertias))
        
        # Calculate rate of change
        diffs = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
        
        # Find the point where improvement starts to diminish
        optimal_k = 1
        for i in range(1, len(diffs)):
            if len(diffs) > 1 and diffs[i-1] > 0 and diffs[i] > 0 and diffs[i-1] / diffs[i] > 2:
                optimal_k = i + 1
                break
        
        optimal_k = max(1, min(optimal_k, max_clusters))
        
        print(f"Optimal number of clusters determined: {optimal_k}")
        print(f"Cluster range tested: {list(cluster_range)}")
        print(f"Inertias: {[f'{x:.2f}' for x in inertias]}")
        
        return optimal_k