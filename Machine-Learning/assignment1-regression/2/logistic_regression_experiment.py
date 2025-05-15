import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Use non-interactive backend for matplotlib
plt.switch_backend('agg')

def load_data(path):
    """Load and prepare the dataset"""
    print("Loading data...")
    data = pd.read_csv(path, header=None, names=['score1', 'score2', 'Admitted'])
    print('Data head:', data.head())
    print('Data shape:', data.shape)
    return data

def visualize_data(data):
    """Visualize the data with different classes"""
    print("\nCreating scatter plot of the data...")
    positive = data[data['Admitted'].isin([1])]
    negative = data[data['Admitted'].isin([0])]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(positive['score1'],
               positive['score2'],
               s=50,
               c='b',
               marker='o',
               label='Admitted')
    ax.scatter(negative['score1'],
               negative['score2'],
               s=50,
               c='r',
               marker='x',
               label='Not Admitted')
    ax.legend()
    ax.set_xlabel('Score 1')
    ax.set_ylabel('Score 2')
    plt.savefig('data_visualization.png')
    plt.close()
    print("Data visualization saved as 'data_visualization.png'")

def sigmoid(z):
    """Compute sigmoid function with numerical stability"""
    # Clip z to avoid overflow in exp
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def plot_sigmoid():
    """Plot the sigmoid function"""
    print("\nPlotting sigmoid function...")
    nums = np.arange(-10, 10, step=1)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(nums, sigmoid(nums), 'r')
    ax.set_xlabel('z')
    ax.set_ylabel('sigmoid(z)')
    ax.set_title('Sigmoid Function')
    ax.grid(True)
    plt.savefig('sigmoid_function.png')
    plt.close()
    print("Sigmoid function plot saved as 'sigmoid_function.png'")

def compute_cost(X, y, w):
    """Compute cost function with numerical stability"""
    m = len(y)
    h = sigmoid(X @ w.T)
    
    # Clip predictions to avoid log(0)
    eps = 1e-15
    h = np.clip(h, eps, 1 - eps)
    
    # Compute cost using vectorized operations
    cost = (-1/m) * (y.T @ np.log(h) + (1-y).T @ np.log(1-h))
    return float(cost)

def gradient_descent(X, y, w, alpha, iterations, verbose=True):
    """Gradient descent with improved stability and monitoring"""
    m = len(y)
    costs = []
    
    for i in range(iterations):
        # Forward pass
        h = sigmoid(X @ w.T)
        
        # Compute gradient
        gradient = (1/m) * (X.T @ (h - y)).T
        
        # Update weights
        w = w - alpha * gradient
        
        # Compute and store cost
        if i % 100 == 0:  # Store cost less frequently to improve performance
            cost = compute_cost(X, y, w)
            costs.append(cost)
            
            if verbose and i % 10000 == 0:
                print(f"Iteration {i}: Cost = {cost:.6f}")
            
            # Check for divergence
            if np.isnan(cost) or np.isinf(cost):
                print(f"Warning: Training diverged at iteration {i}")
                break
    
    if verbose:
        print("Gradient descent completed.")
    
    return w, costs

def plot_convergence(costs):
    """Plot the convergence curve"""
    print("\nPlotting convergence curve...")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    ax.grid(True)
    plt.savefig('convergence_curve.png')
    plt.close()
    print("Convergence curve saved as 'convergence_curve.png'")

def plot_decision_boundary(data, w):
    """Plot the decision boundary"""
    print("\nPlotting decision boundary...")
    positive = data[data['Admitted'].isin([1])]
    negative = data[data['Admitted'].isin([0])]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot decision boundary
    x1 = np.linspace(data.score1.min(), data.score1.max(), 100)
    x2 = -(w[0,0] + w[0,1] * x1) / w[0,2]
    
    plt.plot(x1, x2, 'r', label='Decision Boundary')
    plt.scatter(positive.score1, positive.score2, marker='+', label='Admitted')
    plt.scatter(negative.score1, negative.score2, marker='o', label='Not Admitted')
    plt.xlabel('Score 1')
    plt.ylabel('Score 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.legend()
    plt.grid(True)
    plt.savefig('decision_boundary.png')
    plt.close()
    print("Decision boundary plot saved as 'decision_boundary.png'")

def predict(X, w):
    """Make predictions using the trained model"""
    probability = sigmoid(X @ w.T)
    return [1 if x >= 0.5 else 0 for x in probability]

def compare_learning_rates(X, y, iterations, alphas):
    """Compare different learning rates with improved visualization"""
    plt.figure(figsize=(12, 8))
    
    all_valid_costs = []
    colors = ['b', 'r', 'g']
    
    for alpha, color in zip(alphas, colors):
        print(f"\nTesting learning rate: {alpha}")
        
        # Initialize weights
        w = np.zeros((1, X.shape[1]))
        
        # Run gradient descent
        _, costs = gradient_descent(X, y, w, alpha, iterations, verbose=True)
        
        # Sample costs more sparsely for clearer visualization
        sampling_rate = max(1, len(costs) // 500)  # Aim for about 500 points
        sampled_indices = range(0, len(costs), sampling_rate)
        sampled_costs = [costs[i] for i in sampled_indices]
        iterations_x = [i * 100 * sampling_rate for i in range(len(sampled_costs))]
        
        # Filter out any invalid values
        valid_mask = [not (np.isnan(c) or np.isinf(c)) for c in sampled_costs]
        valid_costs = [c for i, c in enumerate(sampled_costs) if valid_mask[i]]
        valid_iterations = [x for i, x in enumerate(iterations_x) if valid_mask[i]]
        
        if valid_costs:
            plt.plot(valid_iterations, valid_costs, color=color, 
                    label=f'Learning rate = {alpha}', linewidth=2)
            all_valid_costs.extend(valid_costs)
    
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Cost', fontsize=14)
    plt.title('Convergence with Different Learning Rates', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='upper right')
    
    # Set y-axis to show full range of costs
    if all_valid_costs:
        min_cost = min(all_valid_costs)
        max_cost = max(all_valid_costs)
        # Show full range from 0 to slightly above max cost
        plt.ylim(0, max_cost * 1.1)
    
    plt.savefig('learning_rates_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Learning rates comparison saved as 'learning_rates_comparison.png'")

def main():
    try:
        # 1. Load data
        data = load_data('../ex2data1.txt')
        
        # 2. Visualize data
        visualize_data(data)
        
        # 3. Plot sigmoid function
        plot_sigmoid()
        
        # 4. Prepare data
        data.insert(0, 'Ones', 1)
        cols = data.shape[1]
        X = data.iloc[:, 0:cols-1].values  # Convert to numpy array
        y = data.iloc[:, cols-1:cols].values
        
        # Feature scaling
        X_scaled = X.copy()
        for i in range(1, X.shape[1]):  # Skip bias term
            X_scaled[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
        
        # 5. Initialize parameters
        initial_w = np.zeros((1, X_scaled.shape[1]))
        alpha = 0.001
        iterations = 15000  # Reduced number of iterations
        
        # 6. Train model using gradient descent
        w, costs = gradient_descent(X_scaled, y, initial_w, alpha, iterations)
        print('\nFinal cost:', costs[-1])
        print('Optimal parameters:', w)
        
        # 7. Plot convergence
        plot_convergence(costs)
        
        # 8. Plot decision boundary
        plot_decision_boundary(data, w)
        
        # 9. Calculate accuracy
        predictions = predict(X_scaled, w)
        accuracy = np.mean(predictions == y.flatten()) * 100
        print(f'\nAccuracy = {accuracy:.2f}%')
        
        # 10. Compare different learning rates
        print("\nComparing different learning rates...")
        # Use learning rates that will show more dramatic cost differences
        alphas = [0.0001, 0.001, 0.01]  # Increased the largest learning rate
        compare_learning_rates(X_scaled, y, iterations, alphas)
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 