# Attribution and Scope

This file records authorship and dependency boundaries at assignment level. Unless a row says otherwise, the algorithm and experiment code is coursework written by the repository author. Datasets, libraries, established algorithm designs, and referenced implementations are not claimed as original inventions.

| Area | Implementation boundary | External or reference material |
| --- | --- | --- |
| `Intro-to-AI/8-puzzle solver/eight_puzzle.py` | Coursework implementation of BFS, depth-limited DFS, A*, Manhattan distance, path reconstruction, and the Tkinter interface. | Python standard library and Pillow provide queues/heaps, the GUI, and image handling. The search algorithms themselves are established methods. |
| `Intro-to-AI/gobang_AI/gobang_yiannischen.py` | Coursework implementation of Negamax, alpha-beta pruning, move ordering, pattern evaluation, and game state. | `graphics.py` is John Zelle's Simple Object Oriented Graphics Library, version 5.0, distributed under the GPL as stated in its header. Pillow is used for image handling. |
| `Intro-to-AI/decision_tree/decision_tree_metrics.py` | Coursework calculations for entropy, information gain, and gain ratio. | Python standard library. |
| `Intro-to-AI/decision_tree/decision_tree_entropy.py` | Coursework experiment and preprocessing around a library classifier. | Classification uses scikit-learn's `DecisionTreeClassifier`; the classifier is not a from-scratch implementation. Graphviz export and scikit-learn metrics are library utilities. |
| `Intro-to-AI/NN_digit_recognition/mnist_NN.py` | Reference-derived neural-network exercise retained as historical coursework. It is not used as evidence for an original network architecture. | The repository history does not encode a precise bibliographic citation. The repository author identifies it as textbook/reference-derived; an exact citation should be added if recovered. |
| `Machine-Learning/assignment1-regression/` | Coursework experiment sources for linear and logistic regression, including gradient-based optimization and visualizations. | NumPy, pandas, and Matplotlib support computation and plotting; the linear experiment imports scikit-learn's mean-squared-error metric. |
| `Machine-Learning/assignment2-digit-recognition/two_layer_nn.py` | Repository-authored coursework implementation of forward propagation, backpropagation, sample-by-sample updates, vectorized updates, evaluation, and IDX loading. | NumPy and SciPy supply array operations and the sigmoid function; Matplotlib supports experiment plots. MNIST is an external dataset. |
| `Machine-Learning/assignment2-digit-recognition/lenet.py` | Coursework construction and experimentation with the established LeNet architecture. | Network layers, automatic differentiation, optimizers, losses, tensors, and data loaders are provided by PyTorch and torchvision. This is not a from-scratch tensor or autodiff implementation. |
| `Machine-Learning/assignment3-SVM/svm.py` | Handwritten SMO core: kernel evaluation, KKT/error checks, alpha-pair optimization, bounds, alpha/bias updates, support-vector extraction, prediction, and parameter experiments. | NumPy/pandas/Matplotlib support computation, input, and plots. Scikit-learn is limited to accuracy/confusion-matrix metrics and `train_test_split`; it does not train the SVM. |
| `Machine-Learning/assignment4-clustering/kmeans.py` | Handwritten nearest-centroid assignment, centroid recomputation, initialization, and iteration. | Scikit-learn's `KMeans` is used separately to obtain inertia values for the elbow-method comparison. |
| `Machine-Learning/assignment4-clustering/dbscan.py` | Handwritten neighborhood search, core-point detection, and cluster expansion. | NumPy/pandas/Matplotlib support sampling, input, and visualization. |

## Datasets and Results

- Dataset archives in the assignment directories are retained course inputs. Their exact upstream provenance is not consistently recorded in the repository, so the repository does not claim ownership of those datasets.
- Figures under `results/` are retained historical coursework outputs. They were inspected but not regenerated or overwritten during the current reproducibility check.
- Current bounded checks used temporary extracted data and temporary output locations. They do not retroactively establish how every historical figure was produced.
