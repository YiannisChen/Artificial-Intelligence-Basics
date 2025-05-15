import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn.cluster import KMeans
import os

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

def find_closest_centroids(X, centroids):
    """
    Find the closest centroid for each data point in X.
    
    Parameters:
    X : array-like, shape (n_samples, n_features)
        The input data points
    centroids : array-like, shape (k, n_features)
        The current centroids
        
    Returns:
    idx : array-like, shape (n_samples,)
        The index of the closest centroid for each data point
    """
    # Get number of samples and number of centroids
    m = X.shape[0]
    k = centroids.shape[0]
    
    # Initialize array to store the index of the closest centroid for each sample
    idx = np.zeros(m)
    
    # For each sample
    for i in range(m):
        # Initialize minimum distance to a large number
        min_dist = float('inf')
        # For each centroid
        for j in range(k):
            # Calculate the squared Euclidean distance
            dist = np.sum((X[i, :] - centroids[j, :])**2)
            # If this distance is smaller than the current minimum
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
                
    return idx

def compute_centroids(X, idx, k):
    """
    Compute new centroids based on the current cluster assignments.
    
    Parameters:
    X : array-like, shape (n_samples, n_features)
        The input data points
    idx : array-like, shape (n_samples,)
        The cluster assignments for each data point
    k : int
        The number of clusters
        
    Returns:
    centroids : array-like, shape (k, n_features)
        The new centroids
    """
    # Get number of features
    n = X.shape[1]
    
    # Initialize centroids array
    centroids = np.zeros((k, n))
    
    # For each cluster
    for i in range(k):
        # Get indices of points assigned to this cluster
        cluster_points = X[idx == i]
        
        # If cluster is not empty, compute mean
        if len(cluster_points) > 0:
            centroids[i] = np.mean(cluster_points, axis=0)
        else:
            # If cluster is empty, keep the old centroid
            print(f"Warning: Cluster {i} is empty")
            
    return centroids

def run_k_means(X, initial_centroids, max_iters=10, plot_progress=False):
    """
    Run the K-means algorithm.
    
    Parameters:
    X : array-like, shape (n_samples, n_features)
        The input data points
    initial_centroids : array-like, shape (k, n_features)
        The initial centroids
    max_iters : int, optional (default=10)
        Maximum number of iterations
    plot_progress : bool, optional (default=False)
        Whether to plot the progress of the algorithm
        
    Returns:
    idx : array-like, shape (n_samples,)
        The final cluster assignments
    centroids : array-like, shape (k, n_features)
        The final centroids
    """
    # Get number of samples and features
    m, n = X.shape
    k = initial_centroids.shape[0]
    
    # Initialize variables
    idx = np.zeros(m)
    centroids = initial_centroids.copy()
    previous_centroids = centroids.copy()
    
    # Run K-means
    for i in range(max_iters):
        # Output progress
        print(f'K-Means iteration {i+1}/{max_iters}...')
        
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)
        
        # Optionally plot progress
        if plot_progress:
            plt.figure(figsize=(10, 6))
            plt.scatter(X[:, 0], X[:, 1], c=idx, cmap='viridis', alpha=0.5)
            plt.scatter(centroids[:, 0], centroids[:, 1], 
                       c='red', marker='x', s=200, linewidths=3, label='Centroids')
            plt.title(f'K-Means iteration {i+1}')
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.legend()
            plt.show()
        
        # Compute new centroids
        previous_centroids = centroids.copy()
        centroids = compute_centroids(X, idx, k)
        
        # Check for convergence
        if np.all(previous_centroids == centroids):
            print(f'Converged after {i+1} iterations')
            break
            
    return idx, centroids

def init_centroids(X, k):
    """
    Randomly initialize k centroids from the data points.
    
    Parameters:
    X : array-like, shape (n_samples, n_features)
        The input data points
    k : int
        The number of clusters
        
    Returns:
    centroids : array-like, shape (k, n_features)
        The randomly initialized centroids
    """
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)
    
    for i in range(k):
        centroids[i, :] = X[idx[i], :]
    return centroids

def plot_clusters(X, idx, centroids, title, filename):
    """
    Plot clusters and save to file
    """
    plt.figure(figsize=(10, 8))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    # Plot points for each cluster
    for i in range(len(np.unique(idx))):
        cluster = X[np.where(idx == i)[0],:]
        plt.scatter(cluster[:,0], cluster[:,1], c=colors[i], label=f'Cluster {i+1}')
    
    # Plot centroids
    plt.scatter(centroids[:,0], centroids[:,1], c='black', marker='x', s=100, linewidths=2, label='Centroids')
    
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(f'results/{filename}.png')
    plt.close()

def plot_elbow_method(SSE_values):
    """
    Plot elbow method results and save to file
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(SSE_values) + 1), SSE_values, 'o-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('SSE')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    
    # Save plot
    plt.savefig('results/elbow_method.png')
    plt.close()

def plot_clusters_comparison(X, initial_centroids, final_centroids, final_idx, title, filename):
    """
    Plot comparison between initial and final states
    """
    plt.figure(figsize=(15, 6))
    
    # Plot initial state
    plt.subplot(1, 2, 1)
    plt.scatter(X[:,0], X[:,1], c='gray', alpha=0.5, label='Data points')
    plt.scatter(initial_centroids[:,0], initial_centroids[:,1], 
                c='red', marker='x', s=100, linewidths=2, label='Initial centroids')
    plt.title('Initial State')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.grid(True)
    
    # Plot final state
    plt.subplot(1, 2, 2)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i in range(len(np.unique(final_idx))):
        cluster = X[np.where(final_idx == i)[0],:]
        plt.scatter(cluster[:,0], cluster[:,1], c=colors[i], label=f'Cluster {i+1}')
    plt.scatter(final_centroids[:,0], final_centroids[:,1], 
                c='black', marker='x', s=100, linewidths=2, label='Final centroids')
    plt.title('Final State')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'results/{filename}_comparison.png')
    plt.close()

# Load data
data2 = pd.read_csv('data/ex7data2.csv')
X = data2.values

print("1. Manual Centroid Experiments\n")

# First manual initialization
print("a) First manual initialization: [3,3], [6,2], [8,5]")
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
idx, final_centroids = run_k_means(X, initial_centroids, 10)
plot_clusters_comparison(X, initial_centroids, final_centroids, idx, 
                        'K-Means Clustering (First Manual Initialization)', 'kmeans_manual1')

# Second manual initialization
print("\nb) Second manual initialization: [2,2], [5,3], [7,4]")
initial_centroids = np.array([[2, 2], [5, 3], [7, 4]])
idx, final_centroids = run_k_means(X, initial_centroids, 10)
plot_clusters_comparison(X, initial_centroids, final_centroids, idx, 
                        'K-Means Clustering (Second Manual Initialization)', 'kmeans_manual2')

print("\n2. Random Centroid Initialization")
# Random initialization
initial_centroids = init_centroids(X, 3)
print("Random initial centroids:", initial_centroids)
idx, final_centroids = run_k_means(X, initial_centroids, 10)
plot_clusters_comparison(X, initial_centroids, final_centroids, idx, 
                        'K-Means Clustering (Random Initialization)', 'kmeans_random')

print("\n3. Elbow Method Analysis")
# Elbow method
SSE = []
for k in range(1, 9):
    estimator = KMeans(n_clusters=k)
    estimator.fit(X)
    SSE.append(estimator.inertia_)
print("SSE values:", SSE)
plot_elbow_method(SSE) 