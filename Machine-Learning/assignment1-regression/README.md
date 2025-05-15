# Assignment 1: Linear Regression

This assignment implements linear regression using both gradient descent and the normal equation method. The implementation includes visualization of the learning process and comparison of different learning rates.

## Files
- `1/linear_regression_experiment.py`: Main implementation of linear regression
- `1/*.png`: Visualization outputs
- `ex2data1.txt`: Dataset for linear regression
- `ex2data2.txt`: Dataset for logistic regression
- `regress_data1.csv`: Additional regression dataset

## Dependencies
- numpy
- pandas
- matplotlib
- scikit-learn

## Usage
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the experiment:
```bash
python 1/linear_regression_experiment.py
```

## Implementation Details
- Implements batch gradient descent with customizable learning rates
- Includes cost function visualization
- Compares different learning rates
- Provides both gradient descent and normal equation solutions
- Generates visualizations of the fitted line and cost history

## Output
The program generates several visualization files:
- `scatter_plot.png`: Initial data visualization
- `fitted_line.png`: Linear regression fit
- `cost_history.png`: Cost function convergence
- `learning_rates_comparison.png`: Comparison of different learning rates 