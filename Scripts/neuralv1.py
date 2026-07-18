"""
Neural Network v1.0
Author: ...
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, n_inputs,):

        # Initialize weights and bias randomly.
        # Each neuron has one weight per input.
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()

        # Values stored during forward propagation.
        # They are reused later during backpropagation.
        self.inputs = None
        self.z = None
        self.activation = None
        self.delta = None

    @staticmethod
    def sigmoid(x): # Sigmoid Activation Function
        return 1 / (1 + np.exp(-x))

    def derivative(self): 

        # Derivative of sigmoid.
        # Since activation = sigmoid(z),
        # sigmoid'(z) = activation * (1 - activation)
        return self.activation * (1 - self.activation)

    def forward(self, inputs):

        # Save the inputs for gradient computation later.
        self.inputs = np.asarray(inputs).ravel()

        # Weighted sum.
        self.z = np.dot(self.weights, self.inputs) + self.bias

        # Apply activation function.
        self.activation = self.sigmoid(self.z)

        return self.activation

    def gradient_step(self,lr):

        # Gradient descent update.
        # delta = dL/dz
        # dL/dw = delta * input
        self.weights -= lr * self.delta * self.inputs
        self.bias -= lr * self.delta

class Network:
    """Simple fully-connected feedforward neural network."""
    def __init__(self,architecture=[2,5,5,1]):

        # self.net is a list of layers.
        # Each layer is a list of Neuron objects.
        self.net = []
        for i in range(1, len(architecture)):
            inputs = architecture[i - 1]
            neurons = architecture[i]

            layer = []

            for _ in range(neurons):
                layer.append(Neuron(inputs))

            self.net.append(layer)

    def forward_pass(self,inputs):

        # Store activations of every layer.
        # activations[0] is the network input.
        self.activations = [inputs]

        for layer in self.net:
            outputs = []

            # Every neuron in the layer receives
            # the previous layer's activations.
            for neuron in layer:
                outputs.append(neuron.forward(self.activations[-1]))

            self.activations.append(outputs)

        return self.activations
    
    def backward_pass(self, target,lr):

        # Output layer
        output_layer = self.net[-1]

        for neuron, t in zip(output_layer, np.atleast_1d(target)):
        
            # Error at output neuron.
            error = neuron.activation - t
            # δ = error × activation derivative
            neuron.delta = error * neuron.derivative()

        # Hidden layers
        for layer_idx in reversed(range(len(self.net) - 1)):
            current_layer = self.net[layer_idx]
            next_layer = self.net[layer_idx + 1]

            for j, neuron in enumerate(current_layer):

                # Sum of all downstream gradients
                # weighted by the connecting weights.
                weighted_error = 0

                for next_neuron in next_layer:
                    weighted_error += next_neuron.weights[j] * next_neuron.delta
                
                # δ_hidden = (Σ weight × δ_next) × activation derivative
                neuron.delta = weighted_error * neuron.derivative()

        # After every delta has been computed,
        # update all weights.
        for layer in self.net:
            for neuron in layer:
                neuron.gradient_step(lr)
    def train(self,X,y,epochs,lr):
        """Train the network using stochastic gradient descent."""

        # One epoch = one complete pass
        # through the training dataset.

        for epoch in range(epochs):
            loss = 0
            correct = 0

            for x_sample, y_true in zip(X, y):

                activations = self.forward_pass(x_sample)
                prediction = np.array(activations[-1])

                if np.argmax(prediction) == np.argmax(y_true):
                    correct += 1

                loss += 0.5 * np.sum((y_true - prediction) ** 2)
                self.backward_pass(y_true, lr)

            accuracy = correct / len(X)

            print(f"Epoch {epoch}: Loss = {loss:.2f}, Train_Accuracy = {accuracy*100:.2f}%")

            
def main():
    print("Importing Data set")
    mnist = fetch_openml('mnist_784', as_frame=False)


    x = mnist.data
    x /= 255.0

    y = mnist.target.astype(int)
    y_onehot = np.eye(10)[y]

    print("T_Test_Split")
    x_train , x_test , y_train , y_test = train_test_split(x,y_onehot,test_size=0.3,random_state=91)

    lr = 0.1

    model = Network([784, 32, 16, 10])

    model.train(x_train,y_train,epochs=10,lr=lr)
    incorrect = np.zeros(10, dtype=int)
    correct = 0
    confusion = np.zeros((10, 10), dtype=int)

    for sample, target in zip(x_test, y_test):
        pred = np.array(model.forward_pass(sample)[-1])

        actual = np.argmax(target)
        predicted = np.argmax(pred)

        if actual == predicted:
            correct += 1
        else:
            incorrect[actual] += 1

        confusion[actual][predicted] += 1

    accuracy = correct / len(x_test)
    print(f"Accuracy: {accuracy*100:.2f}%")

    plt.figure(figsize=(8,5))
    plt.bar(range(10), incorrect)
    plt.xticks(range(10))
    plt.xlabel("Actual Digit")
    plt.ylabel("Number of Misclassifications")
    plt.title("Misclassifications per Digit")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    plt.imshow(confusion, cmap="Blues")
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(10))
    plt.yticks(range(10))

    for i in range(10):
        for j in range(10):
            plt.text(j, i, confusion[i, j],
                     ha="center", va="center", fontsize=8)

    plt.show()

if __name__ == "__main__":
    main()