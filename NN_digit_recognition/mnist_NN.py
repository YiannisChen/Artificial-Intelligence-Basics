import numpy as np
import scipy.special as ssp
import matplotlib.pyplot as plt

# Neural Network class
class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.activation_function = lambda x: ssp.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     hidden_outputs.T)
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                     inputs.T)

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


# Function to train and test
def train_and_test(train_file_path, test_file_path, input_nodes, hidden_nodes, output_nodes,
                   learning_rate, epochs, show_all_test_result):
    # Create the neural network instance
    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # Load training data
    with open(train_file_path, 'r') as f:
        training_data_list = f.readlines()

    # Training the neural network
    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            nn.train(inputs, targets)

    # Load test data
    with open(test_file_path, 'r') as f:
        test_data_list = f.readlines()

    # Test the neural network
    scorecard = 0
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = nn.query(inputs)
        label = np.argmax(outputs)

        if show_all_test_result:
            print(f"Correct: {correct_label}, Predicted: {label}")
            image_array = np.asfarray(all_values[1:]).reshape((28, 28))
            plt.imshow(image_array, cmap='Greys', interpolation='None')
            plt.show()

        if label == correct_label:
            scorecard += 1

    performance = scorecard / len(test_data_list)
    print(f"Performance: {performance:.4f}")
    return performance


# Experiment setup
input_nodes = 784
output_nodes = 10

# Define experiments with various parameter adjustments
experiments = [
    {"learning_rate": 0.1, "hidden_nodes": 100, "epochs": 5, "show_all_test_result": False},
    {"learning_rate": 0.01, "hidden_nodes": 100, "epochs": 5, "show_all_test_result": False},
    {"learning_rate": 0.5, "hidden_nodes": 100, "epochs": 5, "show_all_test_result": False},
    {"learning_rate": 0.1, "hidden_nodes": 50, "epochs": 5, "show_all_test_result": False},
    {"learning_rate": 0.1, "hidden_nodes": 200, "epochs": 5, "show_all_test_result": False},
    {"learning_rate": 0.1, "hidden_nodes": 100, "epochs": 10, "show_all_test_result": False},
]

# Run experiments
for i, experiment in enumerate(experiments, 1):
    print(f"Experiment {i}: {experiment}")
    train_and_test("mnist_train_100.csv", "mnist_test_100.csv",
                   input_nodes, experiment["hidden_nodes"], output_nodes,
                   experiment["learning_rate"], experiment["epochs"],
                   experiment["show_all_test_result"])

# Test with a larger dataset
print("Testing with larger dataset: mnist_train_400.csv")
train_and_test("mnist_train_400.csv", "mnist_test_100.csv",
               input_nodes, 100, output_nodes,
               learning_rate=0.1, epochs=5, show_all_test_result=False)
