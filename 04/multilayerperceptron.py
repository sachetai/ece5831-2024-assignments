import numpy as np

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

        # Initialize biases
        self.bias_hidden = np.random.randn(hidden_size)
        self.bias_output = np.random.randn(output_size)

        # Activation function
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.sigmoid_derivative = lambda x: x * (1 - x)

    def forward(self, X):
        # Forward propagation through the network
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)

        return self.final_output

    def backward(self, X, y, learning_rate):
        # Calculate the error in the output
        error = y - self.final_output
        d_output = error * self.sigmoid_derivative(self.final_output)

        # Calculate the error in the hidden layer
        hidden_error = d_output.dot(self.weights_hidden_output.T)
        d_hidden = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update the weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * learning_rate
        self.weights_input_hidden += X.T.dot(d_hidden) * learning_rate

        self.bias_output += np.sum(d_output, axis=0) * learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0) * learning_rate

    def train(self, X, y, learning_rate, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)

    def predict(self, X):
        return self.forward(X)

