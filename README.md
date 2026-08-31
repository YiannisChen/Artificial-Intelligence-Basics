# Artificial Intelligence & Machine Learning Coursework

Selected implementations and experiments from undergraduate AI and machine-learning coursework, covering heuristic search, game-tree search, optimization, neural networks, support vector machines, and clustering.

## Overview

This repository collects course assignments that emphasize algorithm implementation and experiment design. The three featured pieces are an 8-puzzle search visualizer, a support-vector machine trained with a handwritten Sequential Minimal Optimization (SMO) core, and a two-layer neural network with sample-wise SGD and batch-vectorized full-batch training paths.

The code is undergraduate coursework rather than research or production software. Dependencies and execution conventions vary by assignment.

## Featured Implementations

### 1. 8-Puzzle Search

[`eight_puzzle.py`](Intro-to-AI/8-puzzle%20solver/eight_puzzle.py) implements and visualizes:

- breadth-first search with a FIFO queue;
- depth-first search with a LIFO stack and a depth limit;
- A* search using Manhattan distance, excluding the blank tile;
- path reconstruction, visited-state tracking, step counts, and runtime reporting.

Manhattan distance is admissible for the standard unit-cost 8-puzzle because each numbered tile must move at least its grid distance to its target. On one representative board `[1, 2, 3, 5, 0, 6, 4, 7, 8]`, BFS and A* both found a 4-move solution; A* visited 10 states versus 60 for BFS. Depth-limited DFS returned a 34-move path and visited 13,715 states. Those figures describe that instance, not a general benchmark.

### 2. SVM with SMO

[`svm.py`](Machine-Learning/assignment3-SVM/svm.py) contains a handwritten SMO training loop with alpha-pair selection and updates, box constraints controlled by `C`, error-cache updates, bias updates, and linear and RBF kernels. The experiment entry points sweep `C` and, for RBF models, `gamma`, then save decision-boundary visualizations.

The SMO training loop is implemented directly; scikit-learn is used for evaluation utilities and the letter-recognition train/test split, not for fitting the SVM. On the 51-point linear dataset, the linear model reaches 0.9804 in-sample training accuracy (50/51). Linear and RBF kernels are both implemented; this README does not report an RBF accuracy figure.

### 3. Two-Layer Neural Network

[`two_layer_nn.py`](Machine-Learning/assignment2-digit-recognition/two_layer_nn.py) implements sigmoid forward propagation, manual backpropagation, MNIST IDX loading, and two training paths that share the same forward/backpropagation formulation but differ in update schedule: sample-wise SGD (`train`) and batch-vectorized full-batch gradient descent (`train_vector`).

The default script runs five sample-wise epochs and then five vectorized epochs on the same network, rather than training two independently initialized models.

## Additional Coursework

- **Gobang / Five-in-a-row:** Negamax search, alpha-beta pruning, pattern-based evaluation, neighboring-move ordering, and search/pruning counters.
- **Clustering:** handwritten K-means assignment/update iterations and a handwritten DBSCAN expansion routine; scikit-learn's KMeans is used separately for the elbow-method comparison.
- **LeNet:** MNIST experiments assembled with PyTorch layers, with pooling, activation, loss, and batch-size variations.
- **Regression:** linear and logistic regression experiment sources, convergence plots, learning-rate comparisons, and evaluation utilities.
- **Decision-tree experiment:** manual entropy/information-gain calculations plus a separate scikit-learn `DecisionTreeClassifier` experiment.

## Results

The repository retains figures produced during the original coursework, including:

- [linear SVM decision boundary](Machine-Learning/assignment3-SVM/results/linear_svm_C_1.png) and [accuracy across C values](Machine-Learning/assignment3-SVM/results/linear_svm_accuracy_vs_C.png);
- [two-layer network learning-rate run](Machine-Learning/assignment2-digit-recognition/results/accuracy_lr_0.1.png) and [LeNet experiment output](Machine-Learning/assignment2-digit-recognition/results/Batch_32_Pool_avg_Act_relu_Loss_cross_entropy.png);
- [K-means centroid comparison](Machine-Learning/assignment4-clustering/results/kmeans_manual1_comparison.png), [elbow analysis](Machine-Learning/assignment4-clustering/results/elbow_method.png), and [DBSCAN parameter output](Machine-Learning/assignment4-clustering/results/dbscan_eps0.15_minpts5.png);
- [linear-regression fit](Machine-Learning/assignment1-regression/1/fitted_line.png) and [logistic-regression decision boundary](Machine-Learning/assignment1-regression/2/decision_boundary.png).

These tracked images are historical outputs from the original coursework runs.

## Repository Structure

```text
Intro-to-AI/
  8-puzzle solver/          search algorithms and Tkinter visualizer
  gobang_AI/                game-tree search and graphical interface
  decision_tree/            entropy metrics and sklearn experiment
  NN_digit_recognition/     reference-derived neural-network exercise
Machine-Learning/
  assignment1-regression/   linear and logistic regression
  assignment2-digit-recognition/  two-layer network and LeNet
  assignment3-SVM/          handwritten SMO and SVM experiments
  assignment4-clustering/   K-means and DBSCAN
```

## Reproduction

Use separate virtual environments when assignments require different dependency sets.

### 8-puzzle GUI

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install pillow
python3 "Intro-to-AI/8-puzzle solver/eight_puzzle.py"
```

Tkinter must be available in the local Python installation. The graphical interface provides buttons for BFS, DFS, and A*.

### SMO / SVM experiments

```bash
cd Machine-Learning/assignment3-SVM
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
unzip -j data/Archive.zip -d data
python3 svm.py linear
```

Replace `linear` with `rbf`, `letter`, or `all` for the longer experiment groups. Generated figures are written to `output/`; the tracked `results/` directory contains historical figures.

### Two-layer neural network

```bash
cd Machine-Learning/assignment2-digit-recognition
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy scipy matplotlib
unzip -j data/data_mnist_NN.zip 'data_mnist/*' -d data
python3 two_layer_nn.py
```

The default script runs five sample-wise epochs followed by five vectorized epochs on the same network and the full training set.

## Attribution & Scope

Implementation boundaries, library use, helper-code provenance, and historical-result status are documented in [ATTRIBUTION.md](ATTRIBUTION.md).
