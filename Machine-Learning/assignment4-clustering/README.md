# Assignment 4: Clustering

This assignment implements and experiments with two clustering algorithms: K-means and DBSCAN.

## Files
- `kmeans.py`: Implementation of K-means clustering algorithm
- `dbscan.py`: Implementation of DBSCAN clustering algorithm
- `data/`: Directory containing datasets
  - `ex7data2.csv`: Dataset for K-means experiments
  - `watermelon4.0.csv`: Dataset for DBSCAN experiments
- `results/`: Directory for storing visualization outputs

## Dependencies
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Implementation Details

### K-means Clustering
- Custom implementation of K-means algorithm
- Features:
  - Manual and random centroid initialization
  - Convergence detection
  - Progress visualization
  - Elbow method for optimal k selection
- Experiments:
  - Comparison of manual vs. random initialization
  - Analysis of convergence iterations
  - Optimal k selection using elbow method

### DBSCAN Clustering
- Custom implementation of DBSCAN algorithm
- Features:
  - Density-based clustering
  - Core point detection
  - Cluster expansion
  - Density calculation
- Experiments:
  - Parameter analysis (eps and MinPts)
  - Cluster density visualization
  - Neighborhood visualization

## Usage

### K-means Clustering
```bash
python kmeans.py
```
This will:
1. Run experiments with manual centroid initialization
2. Run experiments with random centroid initialization
3. Perform elbow method analysis
4. Generate visualizations in the results directory

### DBSCAN Clustering
```bash
python dbscan.py
```
This will:
1. Run experiments with different eps and MinPts values
2. Calculate and display cluster densities
3. Generate visualizations in the results directory

## Output
The program generates:
- Cluster visualizations
- Convergence analysis
- Parameter sensitivity analysis
- Density calculations
- Comparison plots

## Parameter Analysis

### K-means
- Initial centroids:
  - Manual initialization: [3, 3], [6, 2], [8, 5]
  - Random initialization: Randomly selected data points
- Convergence:
  - Monitored through centroid movement
  - Early stopping when centroids stabilize

### DBSCAN
- eps (scanning radius):
  - Larger eps: More points considered neighbors
  - Smaller eps: Stricter neighborhood definition
- MinPts (minimum points):
  - Larger MinPts: Denser clusters required
  - Smaller MinPts: More lenient cluster formation 