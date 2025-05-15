# Assignment 2: Digit Recognition

This assignment implements two different approaches for MNIST digit recognition:
1. A two-layer neural network from scratch
2. LeNet-5 implementation using PyTorch

## Files
- `two_layer_nn.py`: Implementation of a two-layer neural network from scratch
- `lenet.py`: Implementation of LeNet-5 using PyTorch
- `data/`: Directory containing MNIST dataset files
- `pics/`: Directory for storing visualization outputs

## Dependencies
- numpy
- matplotlib
- scipy
- torch
- torchvision

## Usage
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the two-layer neural network:
```bash
python two_layer_nn.py
```

3. Run the LeNet implementation:
```bash
python lenet.py
```

## Implementation Details

### Two-Layer Neural Network
- Implements forward and backward propagation
- Supports both vectorized and non-vectorized training
- Uses sigmoid activation function
- Includes accuracy tracking and visualization

### LeNet-5 Implementation
- Implements the LeNet-5 architecture
- Supports different pooling methods (average/max)
- Supports different activation functions (sigmoid/ReLU)
- Includes batch training and evaluation
- Generates training loss and accuracy plots

## Output
The programs generate:
- Training accuracy plots
- Test accuracy results
- Performance comparisons between different configurations 