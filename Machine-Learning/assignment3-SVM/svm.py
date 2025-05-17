import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import os
import sys
import time

class SMO:
    
    def __init__(self, C, tol, kernel='rbf', gamma=None):
        """Initialize SMO algorithm parameters.
        
        Args:
            C (float): Penalty parameter for misclassification
            tol (float): Tolerance for optimization convergence
            kernel (str): Type of kernel ('linear' or 'rbf')
            gamma (float): Parameter for RBF kernel
        """
        self.C = C
        self.tol = tol
        self.kernel = kernel
        
        if kernel == 'rbf':
            self.K = self._gaussian_kernel
            self.gamma = gamma
        else:
            self.K = self._linear_kernel

    def _gaussian_kernel(self, U, v):
        """Compute Gaussian (RBF) kernel between U and v.
        
        Args:
            U (ndarray): First input array
            v (ndarray): Second input array
            
        Returns:
            float or ndarray: Kernel values
        """
        if U.ndim == 1:
            p = np.dot(U - v, U - v)
        else:
            p = np.sum((U - v) * (U - v), axis=1)
        return np.exp(-self.gamma * p)
    
    def _linear_kernel(self, U, v):
        """Compute linear kernel between U and v.
        
        Args:
            U (ndarray): First input array
            v (ndarray): Second input array
            
        Returns:
            float or ndarray: Kernel values
        """
        if U.ndim == 1:
            return np.dot(U, v)
        return np.dot(U, v.T)

    def _g(self, x):
        """Compute decision function g(x) for a single input.
        
        Args:
            x (ndarray): Input vector
            
        Returns:
            float: Decision function value
        """
        alpha, b, X, y, E = self.args

        idx = np.nonzero(alpha > 0)[0]
        if idx.size > 0:
            return np.sum(y[idx] * alpha[idx] * self.K(X[idx], x)) + b[0]
        return b[0]

    def _optimize_alpha_i_j(self, i, j):
        """Optimize alpha_i and alpha_j using SMO algorithm.
        
        Args:
            i (int): Index of first alpha
            j (int): Index of second alpha
            
        Returns:
            int: 1 if optimization was successful, 0 otherwise
        """
        alpha, b, X, y, E = self.args
        C, tol, K = self.C, self.tol, self.K

        if i == j:
            return 0

        # Calculate bounds for alpha[j]
        if y[i] != y[j]:
            L = max(0, alpha[j] - alpha[i])
            H = min(C, C + alpha[j] - alpha[i])
        else:
            L = max(0, alpha[j] + alpha[i] - C)
            H = min(C, alpha[j] + alpha[i])

        if L == H:
            return 0

        # Calculate eta and update alpha[j]
        eta = K(X[i], X[i]) + K(X[j], X[j]) - 2 * K(X[i], X[j])
        if eta <= 0:
            return 0

        # Update E cache
        if 0 < alpha[i] < C:
            E_i = E[i]
        else:
            E_i = self._g(X[i]) - y[i]

        if 0 < alpha[j] < C:
            E_j = E[j]
        else:
            E_j = self._g(X[j]) - y[j]
        
        # Calculate new alpha[j]
        alpha_j_new = alpha[j] + y[j] * (E_i - E_j) / eta

        # Clip alpha[j]
        if alpha_j_new > H:
            alpha_j_new = H
        elif alpha_j_new < L:
            alpha_j_new = L
        alpha_j_new = np.round(alpha_j_new, 7)

        if np.abs(alpha_j_new - alpha[j]) < tol * (alpha_j_new + alpha[j] + tol):
            return 0

        # Calculate new alpha[i]
        alpha_i_new = alpha[i] + y[i] * y[j] * (alpha[j] - alpha_j_new)
        alpha_i_new = np.round(alpha_i_new, 7)

        # Calculate new b
        b1 = b[0] - E_i \
                -y[i] * (alpha_i_new - alpha[i]) * K(X[i], X[i]) \
                -y[j] * (alpha_j_new - alpha[j]) * K(X[i], X[j])

        b2 = b[0] - E_j \
                -y[i] * (alpha_i_new - alpha[i]) * K(X[i], X[j]) \
                -y[j] * (alpha_j_new - alpha[j]) * K(X[j], X[j])

        if 0 < alpha_i_new < C:
            b_new = b1
        elif 0 < alpha_j_new < C:
            b_new = b2
        else:
            b_new = (b1 + b2) / 2

        # Update E cache
        E[i] = E[j] = 0
        mask = (alpha != 0) & (alpha != C)
        mask[i] = mask[j] = False
        non_bound_idx = np.nonzero(mask)[0]
        for k in non_bound_idx:
            E[k] += b_new - b[0] + y[i] * K(X[i], X[k]) * (alpha_i_new - alpha[i]) \
                                 + y[j] * K(X[j], X[k]) * (alpha_j_new - alpha[j])

        # Update parameters
        alpha[i] = alpha_i_new
        alpha[j] = alpha_j_new
        b[0] = b_new

        return 1

    def _optimize_alpha_i(self, i):
        
        alpha, b, X, y, E = self.args

        if 0 < alpha[i] < self.C:
            E_i = E[i]
        else:
            E_i = self._g(X[i]) - y[i]

        if (E_i * y[i] < -self.tol and alpha[i] < self.C) or \
                (E_i * y[i] > self.tol and alpha[i] > 0):
            mask = (alpha != 0) & (alpha != self.C)
            non_bound_idx = np.nonzero(mask)[0]
            bound_idx = np.nonzero(~mask)[0]

            if len(non_bound_idx) > 1:
                if E[i] > 0:
                    j = non_bound_idx[np.argmin(E[non_bound_idx])]
                else:
                    j = non_bound_idx[np.argmax(E[non_bound_idx])]

                if self._optimize_alpha_i_j(i, j):
                    return 1

            np.random.shuffle(non_bound_idx)
            for j in non_bound_idx:
                if self._optimize_alpha_i_j(i, j):
                    return 1

            np.random.shuffle(bound_idx)
            for j in bound_idx:
                if self._optimize_alpha_i_j(i, j):
                    return 1

        return 0

    def train(self, X_train, y_train):
        #Train SVM using SMO algorithm.
        
        m, _ = X_train.shape
        y_train = np.where(y_train == 0, -1, 1)

        alpha = np.zeros(m)
        b = np.zeros(1)
        E = np.zeros(m)
        self.args = [alpha, b, X_train, y_train, E]

        n_changed = 0
        examine_all = True
        iteration = 0
        max_iter = 50
        no_change_count = 0
        max_no_change = 5
        
        while (n_changed > 0 or examine_all) and iteration < max_iter:
            n_changed = 0
            iteration += 1

            for i in range(m):
                if examine_all or 0 < alpha[i] < self.C:
                    n_changed += self._optimize_alpha_i(i)

            print(f'Iteration {iteration}: n_changed = {n_changed}')
            print(f'Number of support vectors: {np.count_nonzero((alpha > 0) & (alpha < self.C))}')

            if n_changed == 0:
                no_change_count += 1
                if no_change_count >= max_no_change:
                    print("Early stopping: No changes for several iterations")
                    break
            else:
                no_change_count = 0

            examine_all = (not examine_all) and (n_changed == 0)

        if iteration >= max_iter:
            print("Warning: Maximum iterations reached. Training may not have converged.")

        idx = np.nonzero(alpha > 0)[0]
        self.sv_alpha = alpha[idx]
        self.sv_X = X_train[idx]
        self.sv_y = y_train[idx]
        self.sv_b = b[0]

    def _predict_one(self, x):
        #redict class for a single input.
        
        k = self.K(self.sv_X, x)
        return np.sum(self.sv_y * self.sv_alpha * k) + self.sv_b

    def predict(self, X_test):
        #Predict classes for multiple inputs.
        
        return np.sign(np.array([self._predict_one(x) for x in X_test]))

    def plot_decision_boundary(self, X, y, title=None, save_path=None):
        #Plot decision boundary and support vectors.
        
        plt.figure(figsize=(10, 8))
        
        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        
        # Plot support vectors
        plt.scatter(self.sv_X[:, 0], self.sv_X[:, 1], 
                   s=100, linewidth=1, facecolors='none', edgecolors='k')
        
        plt.title(title or 'SVM Decision Boundary')
        plt.xlabel('X1')
        plt.ylabel('X2')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

def ensure_output_dir():
    #Create output directory if it doesn't exist."""
    if not os.path.exists('output'):
        os.makedirs('output')

def experiment_linear_svm(data_path, C_values=[0.1, 1, 10, 100]):
    #Run experiments with linear SVM.
    data = pd.read_csv(data_path)
    X = data[['X1', 'X2']].values
    y = data['y'].values
    
    ensure_output_dir()
    
    for C in C_values:
        svm = SMO(C=C, tol=0.01, kernel='linear')
        svm.train(X, y)
        
        accuracy = accuracy_score(y, svm.predict(X))
        print(f'C = {C}, Accuracy: {accuracy:.4f}')
        
        svm.plot_decision_boundary(
            X, y,
            title=f'Linear SVM (C={C})',
            save_path=f'output/linear_svm_C{C}.png'
        )

def experiment_rbf_svm(data_path, C_values=[0.1, 1, 10, 100], gamma_values=[0.01, 0.1, 1, 10]):
    #Run experiments with RBF SVM.
    data = pd.read_csv(data_path)
    X = data[['X1', 'X2']].values
    y = data['y'].values
    
    ensure_output_dir()
    
    for C in C_values:
        for gamma in gamma_values:
            svm = SMO(C=C, tol=0.01, kernel='rbf', gamma=gamma)
            svm.train(X, y)
            
            accuracy = accuracy_score(y, svm.predict(X))
            print(f'C = {C}, gamma = {gamma}, Accuracy: {accuracy:.4f}')
            
            svm.plot_decision_boundary(
                X, y,
                title=f'RBF SVM (C={C}, gamma={gamma})',
                save_path=f'output/rbf_svm_C{C}_gamma{gamma}.png'
            )

def experiment_letter_recognition(data_path, C_values=[1, 10, 100], gamma_values=[0.01, 0.1, 1]):
    #Run experiments with letter recognition.
    
    X = np.genfromtxt(data_path, delimiter=',', usecols=range(1, 17))
    y = np.genfromtxt(data_path, delimiter=',', usecols=0, dtype=str)
    y = np.where(y == 'C', 1, -1)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    ensure_output_dir()
    
    results = []
    for C in C_values:
        for gamma in gamma_values:
            svm = SMO(C=C, tol=0.01, kernel='rbf', gamma=gamma)
            svm.train(X_train, y_train)
            
            y_pred = svm.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            results.append({
                'C': C,
                'gamma': gamma,
                'accuracy': accuracy,
                'confusion_matrix': conf_matrix
            })
            
            print(f'C = {C}, gamma = {gamma}')
            print(f'Accuracy: {accuracy:.4f}')
            print('Confusion Matrix:')
            print(conf_matrix)
            print()
    
    # Find best parameters
    best_result = max(results, key=lambda x: x['accuracy'])
    print('Best Parameters:')
    print(f'C = {best_result["C"]}, gamma = {best_result["gamma"]}')
    print(f'Best Accuracy: {best_result["accuracy"]:.4f}')
    print('Best Confusion Matrix:')
    print(best_result['confusion_matrix'])

def print_usage():
    """Print usage instructions."""
    print('Usage: python svm.py [experiment]')
    print('Experiments:')
    print('  linear    - Run linear SVM experiments')
    print('  rbf       - Run RBF SVM experiments')
    print('  letter    - Run letter recognition experiments')
    print('  all       - Run all experiments')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)
        
    experiment = sys.argv[1]
    
    if experiment == 'linear':
        experiment_linear_svm('data/svmdata1.csv')
    elif experiment == 'rbf':
        experiment_rbf_svm('data/svmdata2.csv')
    elif experiment == 'letter':
        experiment_letter_recognition('data/letter-recognition.data')
    elif experiment == 'all':
        experiment_linear_svm('data/svmdata1.csv')
        experiment_rbf_svm('data/svmdata2.csv')
        experiment_letter_recognition('data/letter-recognition.data')
    else:
        print_usage()
        sys.exit(1)
