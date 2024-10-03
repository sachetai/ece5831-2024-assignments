import numpy as np
from multilayerperceptron import MultiLayerPerceptron

# Initialize MLP with input size, hidden size, and output size
mlp = MultiLayerPerceptron(input_size=2, hidden_size=3, output_size=1)

# Training data: XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train the network
mlp.train(X, y, learning_rate=0.1, epochs=10000)

# Test the network
output = mlp.predict(X)
print("Predictions after training:")
print(output)
