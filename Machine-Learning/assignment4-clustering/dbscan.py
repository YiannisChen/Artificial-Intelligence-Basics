import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

def dist(a, b):
    """
    Calculate Euclidean distance between two points.
    
    Parameters:
    a : tuple
        First point coordinates (x, y)
    b : tuple
        Second point coordinates (x, y)
        
    Returns:
    float
        Euclidean distance between points a and b
    """
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))

def DBSCAN(D, eps, MinPts):
    """
    Perform DBSCAN clustering.
    
    Parameters:
    D : list of tuples
        Dataset of points
    eps : float
        Maximum distance between points to be considered neighbors
    MinPts : int
        Minimum number of points required to form a cluster
        
    Returns:
    C : list of lists
        List of clusters, where each cluster is a list of points
    """
    # Initialize core objects set T, cluster number k, cluster set C, and unvisited set P
    T = set()
    k = 0
    C = []
    P = set(D)
    
    # Find all core points
    for d in D:
        if len([i for i in D if dist(d, i) <= eps]) >= MinPts:
            T.add(d)
    
    # Start clustering
    while len(T):
        P_old = P
        # Randomly select a core point
        o = list(T)[np.random.randint(0, len(T))]
        # Remove selected point from unvisited set
        P = P - {o}
        Q = []
        Q.append(o)
        
        # Expand cluster
        while len(Q):
            q = Q[0]
            # Find neighbors of q
            Nq = [i for i in D if dist(q, i) <= eps]
            
            # If q is a core point
            if len(Nq) >= MinPts:
                # Add unvisited neighbors to Q
                S = P & set(Nq)
                Q.extend(list(S))
                P = P - S
            Q.remove(q)
        
        # Create new cluster
        k += 1
        Ck = list(P_old - P)
        T = T - set(Ck)
        C.append(Ck)
    
    return C

def calculate_cluster_density(C, eps):
    """
    Calculate the density of each cluster.
    
    Parameters:
    C : list of lists
        List of clusters, where each cluster is a list of points
    eps : float
        The eps value used for clustering
        
    Returns:
    list
        List of densities for each cluster
    """
    densities = []
    for cluster in C:
        if len(cluster) == 0:
            densities.append(0)
            continue
            
        # Calculate average number of neighbors within eps for each point
        total_neighbors = 0
        for point in cluster:
            neighbors = sum(1 for p in cluster if dist(point, p) <= eps)
            total_neighbors += neighbors
            
        # Density = average number of neighbors / cluster size
        density = total_neighbors / len(cluster)
        densities.append(density)
    
    return densities

def draw_clusters(C, eps, minpts, densities=None):
    """
    Visualize the clustering results with enhanced information and save to file.
    """
    colors = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    plt.figure(figsize=(12, 8))
    
    # Plot points for each cluster
    for i in range(len(C)):
        if len(C[i]) == 0:
            continue
            
        # Extract x and y coordinates for points in this cluster
        coo_X = [point[0] for point in C[i]]
        coo_Y = [point[1] for point in C[i]]
        
        # Calculate cluster center
        center_x = np.mean(coo_X)
        center_y = np.mean(coo_Y)
        
        # Plot points with cluster-specific color
        plt.scatter(coo_X, coo_Y, marker='o', 
                   color=colors[i % len(colors)], 
                   label=f'Cluster {i+1} (size={len(C[i])})')
        
        # Plot cluster center
        plt.scatter(center_x, center_y, marker='x', 
                   color=colors[i % len(colors)], s=100, linewidths=2)
        
        # Add density information if available
        if densities is not None:
            plt.annotate(f'Density: {densities[i]:.2f}', 
                        (center_x, center_y),
                        xytext=(10, 10), textcoords='offset points')
    
    # Add eps circles around points to show neighborhood
    if len(C) > 0 and len(C[0]) > 0:
        sample_point = C[0][0]
        circle = plt.Circle(sample_point, eps, fill=False, linestyle='--', 
                          color='gray', alpha=0.5, label=f'eps={eps}')
        plt.gca().add_patch(circle)
    
    plt.title(f'DBSCAN Clustering Results (eps={eps}, MinPts={minpts})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot
    plt.savefig(f'results/dbscan_eps{eps}_minpts{minpts}.png')
    plt.close()

def experiment_with_parameters(dataset, eps_values, minpts_values):
    """
    Run DBSCAN with different parameter combinations and analyze results.
    """
    print("\nDBSCAN Parameter Analysis")
    print("========================")
    
    for eps in eps_values:
        print(f"\nEps = {eps}")
        print("-" * 20)
        
        for minpts in minpts_values:
            print(f"\nMinPts = {minpts}")
            print("-" * 10)
            
            # Run DBSCAN
            C = DBSCAN(dataset, eps, minpts)
            
            # Calculate cluster densities
            densities = calculate_cluster_density(C, eps)
            
            # Print results
            print(f"Number of clusters: {len(C)}")
            print(f"Cluster sizes: {[len(cluster) for cluster in C]}")
            print(f"Cluster densities: {[f'{d:.2f}' for d in densities]}")
            
            # Visualize results
            draw_clusters(C, eps, minpts, densities)
            
            # Print analysis
            if len(C) > 0:
                print("\nAnalysis:")
                print(f"- Average cluster size: {np.mean([len(c) for c in C]):.2f}")
                print(f"- Average cluster density: {np.mean(densities):.2f}")
                print(f"- Density variation: {np.std(densities):.2f}")
            else:
                print("\nNo clusters found with these parameters")

if __name__ == "__main__":
    # Load the watermelon dataset
    data = pd.read_csv('data/watermelon4.0.csv')
    dataset = [(i[0], i[1]) for i in data.values]
    
    # Run experiments with different parameters
    eps_values = [0.11, 0.15, 0.20]
    minpts_values = [3, 5, 7]
    
    print("DBSCAN Parameter Experiments")
    print("===========================")
    experiment_with_parameters(dataset, eps_values, minpts_values) 