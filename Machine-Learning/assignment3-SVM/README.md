# Assignment 3: Support Vector Machine (SVM)

This assignment implements and experiments with Support Vector Machines for classification tasks, including spam email classification and handwritten letter recognition.

## Files
- `svm.py`: Implementation of SVM using SMO (Sequential Minimal Optimization) algorithm
- `data/`: Directory containing datasets
  - `svmdata1.csv`: Dataset for linear SVM experiments
  - `svmdata2.csv`: Dataset for non-linear SVM experiments
  - `letter-recognition.data`: Dataset for handwritten letter recognition
- `output/`: Directory for storing visualization outputs

## Dependencies
- numpy
- matplotlib
- pandas
- scikit-learn

## Implementation Details

### SVM Implementation
- Custom SMO (Sequential Minimal Optimization) algorithm implementation
- Supports both linear and RBF (Gaussian) kernels
- Includes early stopping and convergence checks
- Provides visualization of decision boundaries

### Experiments
1. Linear SVM Classification
   - Tests different C values (penalty parameter)
   - Visualizes decision boundaries
   - Compares classification accuracy

2. Non-linear SVM Classification
   - Tests different combinations of C and gamma parameters
   - Uses RBF kernel for non-linear classification
   - Visualizes decision boundaries and confidence scores

3. Handwritten Letter Recognition
   - Binary classification of letter 'C' vs non-'C'
   - Tests different C and gamma parameter combinations
   - Evaluates using accuracy and confusion matrix

## Usage
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run experiments:
```bash
python svm.py
```

## Output
The program generates:
- Decision boundary visualizations
- Classification accuracy results
- Confusion matrices
- Performance comparisons between different parameter combinations

## Parameter Analysis
- C (Penalty Parameter):
  - Larger C: Tighter margin, fewer misclassifications
  - Smaller C: Wider margin, more misclassifications
- Gamma (RBF Kernel Parameter):
  - Larger gamma: More complex decision boundary
  - Smaller gamma: Smoother decision boundary 