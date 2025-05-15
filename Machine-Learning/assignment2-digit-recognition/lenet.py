import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import torchvision.utils
import torch.utils.data as data
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
import time
import struct

class Reshape(nn.Module):
    """Reshape input to 28x28 image format"""
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

def load_mnist_data(batch_size):
    """Load MNIST dataset and create data loaders"""
    # Read training data
    train_image = open('data/train-images.idx3-ubyte', 'rb')
    train_label = open('data/train-labels.idx1-ubyte', 'rb')
    
    # Read training labels and images
    magic, n = struct.unpack('>II', train_label.read(8))
    y_train = np.fromfile(train_label, dtype=np.uint8)
    
    magic, num, rows, cols = struct.unpack('>IIII', train_image.read(16))
    x_train = np.fromfile(train_image, dtype=np.uint8).reshape(len(y_train), 784)
    
    # Read test data
    test_image = open('data/t10k-images.idx3-ubyte', 'rb')
    test_label = open('data/t10k-labels.idx1-ubyte', 'rb')
    
    magic, n = struct.unpack('>II', test_label.read(8))
    y_test = np.fromfile(test_label, dtype=np.uint8)
    
    magic, num, rows, cols = struct.unpack('>IIII', test_image.read(16))
    x_test = np.fromfile(test_image, dtype=np.uint8).reshape(len(y_test), 784)
    
    # Convert to PyTorch tensors and normalize
    x_train = torch.FloatTensor(x_train) / 255.0
    y_train = torch.LongTensor(y_train)
    x_test = torch.FloatTensor(x_test) / 255.0
    y_test = torch.LongTensor(y_test)
    
    # Create datasets and data loaders
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class LeNet(nn.Module):
    """LeNet-5 architecture implementation"""
    def __init__(self, pooling_type='avg', activation_type='sigmoid'):
        super(LeNet, self).__init__()
        
        # Select pooling layer type
        if pooling_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:  # max pooling
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
        # Select activation function
        if activation_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:  # ReLU
            self.activation = nn.ReLU()
            
        # Define network architecture
        self.net = nn.Sequential(
            Reshape(),
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # First conv layer
            self.activation,
            self.pool,
            nn.Conv2d(6, 16, kernel_size=5),  # Second conv layer
            self.activation,
            self.pool,
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),  # Fully connected layers
            self.activation,
            nn.Linear(120, 84),
            self.activation,
            nn.Linear(84, 10)  # Output layer
        )

    def forward(self, x):
        return self.net(x)

def train_model(model, train_loader, test_loader, num_epochs, loss_function, optimizer):
    """Train model and evaluate performance"""
    train_losses = []
    test_accuracies = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        # Training phase
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(x)
            
            # Handle MSE loss with one-hot encoding
            if isinstance(loss_function, nn.MSELoss):
                y_one_hot = torch.zeros(y.size(0), 10)
                y_one_hot.scatter_(1, y.unsqueeze(1), 1)
                loss = loss_function(out, y_one_hot)
            else:
                loss = loss_function(out, y)
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
        
        # Evaluation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                out = model(x)
                _, predicted = torch.max(out.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
    
    return train_losses, test_accuracies, time.time() - start_time

def plot_results(train_losses, test_accuracies, title):
    """Plot training loss and test accuracy curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    ax2.plot(test_accuracies)
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.close()

def main():
    """Main execution function"""
    # Experiment parameters
    batch_sizes = [16, 32, 64]
    epochs = 3
    
    results = []
    for batch_size in batch_sizes:
        train_loader, test_loader = load_mnist_data(batch_size)
        
        # Test different pooling methods
        for pooling_type in ['avg', 'max']:
            # Test different activation functions
            for activation_type in ['sigmoid', 'relu']:
                model = LeNet(pooling_type=pooling_type, activation_type=activation_type)
                
                # Test different loss functions
                for loss_type in ['cross_entropy', 'mse']:
                    loss_function = nn.CrossEntropyLoss() if loss_type == 'cross_entropy' else nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters())
                    
                    # Train model and record results
                    train_losses, test_accuracies, training_time = train_model(
                        model, train_loader, test_loader, epochs, loss_function, optimizer
                    )
                    
                    results.append({
                        'batch_size': batch_size,
                        'pooling': pooling_type,
                        'activation': activation_type,
                        'loss': loss_type,
                        'accuracy': test_accuracies[-1],
                        'time': training_time
                    })

if __name__ == '__main__':
    main() 