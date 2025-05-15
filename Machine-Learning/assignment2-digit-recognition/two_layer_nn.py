import numpy as np
import matplotlib.pyplot as plt
import math
import struct
import time
from scipy.special import expit

def data_fetch_preprocessing():
    """Load and preprocess MNIST dataset"""
    # Read MNIST data files
    train_image = open('data/train-images.idx3-ubyte', 'rb')
    test_image = open('data/t10k-images.idx3-ubyte', 'rb')
    train_label = open('data/train-labels.idx1-ubyte', 'rb')
    test_label = open('data/t10k-labels.idx1-ubyte', 'rb')

    # Read training labels and convert to one-hot encoding
    magic, n = struct.unpack('>II', train_label.read(8))
    y_train_label = np.array(np.fromfile(train_label, dtype=np.uint8), ndmin=1)
    y_train = np.ones((10, 60000)) * 0.01
    for i in range(60000):
        y_train[y_train_label[i]][i] = 0.99

    # Read test labels
    magic_t, n_t = struct.unpack('>II', test_label.read(8))
    y_test = np.fromfile(test_label, dtype=np.uint8).reshape(10000, 1)

    # Read image data
    magic, num, rows, cols = struct.unpack('>IIII', train_image.read(16))
    x_train = np.fromfile(train_image, dtype=np.uint8).reshape(len(y_train_label), 784).T

    magic_2, num_2, rows_2, cols_2 = struct.unpack('>IIII', test_image.read(16))
    x_test = np.fromfile(test_image, dtype=np.uint8).reshape(len(y_test), 784).T

    # Normalize data to range [0.01, 0.99]
    x_train = x_train / 255 * 0.99 + 0.01
    x_test = x_test / 255 * 0.99 + 0.01

    # Close file handles
    train_image.close()
    train_label.close()
    test_image.close()
    test_label.close()

    return x_train, y_train, x_test, y_test

class Neural_Network(object):
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        """Initialize neural network with given architecture"""
        self.inputnodes = inputnodes
        self.hiddennodes = hiddennodes
        self.outputnodes = outputnodes
        self.learningrate = learningrate
        
        # Initialize weights and biases
        self.w1 = np.random.randn(self.hiddennodes, self.inputnodes) * 0.01
        self.w2 = np.random.randn(self.outputnodes, self.hiddennodes) * 0.01
        self.b1 = np.zeros((self.hiddennodes, 1))
        self.b2 = np.zeros((self.outputnodes, 1))

    def softmax(self, x):
        """Apply sigmoid activation function"""
        return expit(x)

    def forward_propagation(self, input_data, weight_matrix, b):
        """Compute forward pass: z = W·X + b, a = σ(z)"""
        z = np.add(np.dot(weight_matrix, input_data), b)
        return z, self.softmax(z)

    def train(self, input_data, label_data, epochs):
        """Train network using non-vectorized approach"""
        accuracies = []
        
        for epoch in range(epochs):
            for i in range(60000):
                # Forward pass
                z1, a1 = self.forward_propagation(input_data[:, i].reshape(-1, 1), self.w1, self.b1)
                z2, a2 = self.forward_propagation(a1, self.w2, self.b2)
                
                # Backward pass
                dz2 = a2 - label_data[:, i].reshape(-1, 1)
                dz1 = np.dot(self.w2.T, dz2) * a1 * (1 - a1)
                
                # Update parameters
                self.w2 -= self.learningrate * np.dot(dz2, a1.T)
                self.b2 -= self.learningrate * dz2
                self.w1 -= self.learningrate * np.dot(dz1, input_data[:, i].reshape(-1, 1).T)
                self.b1 -= self.learningrate * dz1
            
            accuracy = self.evaluate(input_data, label_data)
            accuracies.append(accuracy)
            print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.2f}%')
        
        return accuracies

    def train_vector(self, input_data, label_data, epochs):
        """Train network using vectorized approach"""
        accuracies = []
        
        for epoch in range(epochs):
            # Forward pass
            z1, a1 = self.forward_propagation(input_data, self.w1, self.b1)
            z2, a2 = self.forward_propagation(a1, self.w2, self.b2)
            
            # Backward pass
            dz2 = a2 - label_data
            dz1 = np.dot(self.w2.T, dz2) * a1 * (1 - a1)
            
            # Update parameters
            self.w2 -= self.learningrate * np.dot(dz2, a1.T) / 60000
            self.b2 -= self.learningrate * np.sum(dz2, axis=1, keepdims=True) / 60000
            self.w1 -= self.learningrate * np.dot(dz1, input_data.T) / 60000
            self.b1 -= self.learningrate * np.sum(dz1, axis=1, keepdims=True) / 60000
            
            accuracy = self.evaluate(input_data, label_data)
            accuracies.append(accuracy)
            print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.2f}%')
        
        return accuracies

    def evaluate(self, input_data, label_data):
        """Evaluate model accuracy"""
        z1, a1 = self.forward_propagation(input_data, self.w1, self.b1)
        z2, a2 = self.forward_propagation(a1, self.w2, self.b2)
        
        predictions = np.argmax(a2, axis=0)
        true_labels = np.argmax(label_data, axis=0)
        accuracy = np.mean(predictions == true_labels) * 100
        
        return accuracy

    def predict(self, input_data, label):
        """Predict using non-vectorized approach"""
        precision = 0
        for i in range(10000):
            z1, a1 = self.forward_propagation(input_data[:, i].reshape(-1, 1), self.w1, self.b1)
            z2, a2 = self.forward_propagation(a1, self.w2, self.b2)
            if np.argmax(a2) == label[i]:
                precision += 1
        accuracy = 100 * precision / 10000
        print(f'Test accuracy: {accuracy:.2f}%')
        return accuracy

    def predict_vector(self, input_data, label):
        """Predict using vectorized approach"""
        z1, a1 = self.forward_propagation(input_data, self.w1, self.b1)
        z2, a2 = self.forward_propagation(a1, self.w2, self.b2)
        predictions = np.argmax(a2, axis=0)
        true_labels = label.flatten()
        accuracy = np.mean(predictions == true_labels) * 100
        print(f'Test accuracy: {accuracy:.2f}%')
        return accuracy

def main():
    """Main execution function"""
    # Load and preprocess data
    x_train, y_train, x_test, y_test = data_fetch_preprocessing()
    
    # Initialize network
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.1
    epochs = 5
    
    # Create and train network
    network = Neural_Network(input_nodes, hidden_nodes, output_nodes, learning_rate)
    
    # Train using both approaches
    print("\nTraining with non-vectorized approach:")
    accuracies_non_vector = network.train(x_train, y_train, epochs)
    
    print("\nTraining with vectorized approach:")
    accuracies_vector = network.train_vector(x_train, y_train, epochs)
    
    # Evaluate on test set
    print("\nEvaluating on test set:")
    network.predict_vector(x_test, y_test)

if __name__ == "__main__":
    main() 