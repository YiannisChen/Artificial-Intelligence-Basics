import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Configure matplotlib settings
plt.rcParams['font.family'] = 'Arial'
plt.switch_backend('agg')

def load_data(path):
    """Load and display dataset statistics"""
    data = pd.read_csv(path)
    print("Dataset head:")
    print(data.head())
    print("\nDataset description:")
    print(data.describe())
    return data

def visualize_data(data):
    """Create scatter plot of population vs profit"""
    plt.figure(figsize=(12, 8))
    plt.scatter(data['人口'], data['收益'], label='Training Data', alpha=0.6)
    plt.xlabel('Population', fontsize=14)
    plt.ylabel('Profit', fontsize=14)
    plt.title('Population vs. Profit', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.savefig('scatter_plot.png')
    plt.close()

def prepare_data(data):
    """Prepare data matrix with bias term"""
    data.insert(0, 'Ones', 1)
    cols = data.shape[1]
    X = data.iloc[:, :cols-1].values
    y = data.iloc[:, cols-1:].values
    return X, y

def compute_cost(X, y, w):
    """Compute mean squared error cost"""
    m = len(X)
    inner = np.power(X @ w - y, 2)
    return np.sum(inner) / (2 * m)

def batch_gradient_descent(X, y, w, alpha, iterations):
    """Perform batch gradient descent optimization"""
    m = len(X)
    costs = []
    w_copy = w.copy()
    
    for i in range(iterations):
        h = X @ w_copy
        gradient = (X.T @ (h - y)) / m
        w_copy = w_copy - alpha * gradient
        cost = compute_cost(X, y, w_copy)
        
        if not (np.isnan(cost) or np.isinf(cost)):
            costs.append(cost)
        
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.6f}")
            if np.isnan(cost) or np.isinf(cost):
                print("Warning: Cost has become invalid, stopping early")
                break
    
    return w_copy, costs

def plot_fit_line(data, w):
    """Plot regression line with training data"""
    plt.figure(figsize=(12, 8))
    plt.scatter(data['人口'], data['收益'], label='Training Data', alpha=0.6)
    
    x = np.linspace(data['人口'].min(), data['人口'].max(), 100)
    f = w[0] + w[1] * x
    plt.plot(x, f, 'r', label='Predictions', linewidth=2)
    
    plt.xlabel('Population', fontsize=14)
    plt.ylabel('Profit', fontsize=14)
    plt.title('Population vs. Profit with Linear Regression', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.savefig('fitted_line.png')
    plt.close()

def plot_cost_history(costs_dict, iterations):
    """Plot cost convergence for best learning rate"""
    plt.figure(figsize=(12, 8))
    best_alpha = min(costs_dict.keys(), key=lambda a: costs_dict[a][-1])
    costs = costs_dict[best_alpha]
    valid_costs = [c for c in costs if not (np.isnan(c) or np.isinf(c))]
    
    if valid_costs:
        plt.plot(range(len(valid_costs)), valid_costs, 'b-', linewidth=2)
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('Cost', fontsize=14)
        plt.title(f'Cost vs. Training Epochs (Learning Rate = {best_alpha})', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(min(valid_costs) * 0.95, max(valid_costs) * 1.05)
        plt.savefig('cost_history.png')
        plt.close()

def compare_learning_rates(X, y, w, iterations, alphas):
    """Compare convergence for different learning rates"""
    plt.figure(figsize=(12, 8))
    all_costs = []
    colors = ['b', 'r', 'g']
    
    for alpha, color in zip(alphas, colors):
        w_copy = w.copy()
        costs = []
        m = len(X)
        
        for i in range(iterations):
            h = X @ w_copy
            gradient = (X.T @ (h - y)) / m
            w_copy = w_copy - alpha * gradient
            cost = compute_cost(X, y, w_copy)
            
            if i % 10 == 0:
                costs.append(cost)
            
            if i % 500 == 0:
                print(f"Iteration {i}: Cost = {cost:.6f}")
                if np.isnan(cost) or np.isinf(cost):
                    break
        
        iterations_x = [i * 10 for i in range(len(costs))]
        plt.plot(iterations_x, costs, color=color, label=f'Learning rate = {alpha}', linewidth=2)
        all_costs.extend(costs)
    
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Cost', fontsize=14)
    plt.title('Convergence with Different Learning Rates', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='upper right')
    
    valid_costs = [c for c in all_costs if not (np.isnan(c) or np.isinf(c))]
    if valid_costs:
        min_cost = min(valid_costs)
        max_cost = max(valid_costs)
        plt.ylim(max(0, min_cost * 0.95), min(max_cost * 1.05, max_cost * 2))
    
    plt.savefig('learning_rates_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def normal_equation(X, y):
    """Solve linear regression using normal equation"""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def main():
    """Main execution function"""
    # Load and prepare data
    data = load_data('regress_data1.csv')
    visualize_data(data)
    X, y = prepare_data(data)
    
    # Initialize parameters
    w = np.zeros((X.shape[1], 1))
    iterations = 1500
    alphas = [0.01, 0.03, 0.1]
    
    # Compare learning rates
    compare_learning_rates(X, y, w, iterations, alphas)
    
    # Run gradient descent with best learning rate
    w_gd, costs = batch_gradient_descent(X, y, w, 0.01, iterations)
    plot_cost_history({0.01: costs}, iterations)
    plot_fit_line(data, w_gd)
    
    # Compare with normal equation
    w_ne = normal_equation(X, y)
    print("\nGradient Descent parameters:", w_gd.flatten())
    print("Normal Equation parameters:", w_ne.flatten())
    
    # Compare MSE
    y_pred_gd = X @ w_gd
    y_pred_ne = X @ w_ne
    mse_gd = mean_squared_error(y, y_pred_gd)
    mse_ne = mean_squared_error(y, y_pred_ne)
    print("\nMSE (Gradient Descent):", mse_gd)
    print("MSE (Normal Equation):", mse_ne)

if __name__ == "__main__":
    main() 