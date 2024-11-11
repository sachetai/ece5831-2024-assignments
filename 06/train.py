import numpy as np
import pickle
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from two_layer_net_with_back_prop import TwoLayerNetWithBackProp

# Load MNIST data without using pandas
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data / 255.0, mnist.target.astype(int)
y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the network
network = TwoLayerNetWithBackProp(input_size=784, hidden_size=50, output_size=10)

# Training parameters
iterations = 10000
batch_size = 16
learning_rate = 0.01

for i in range(iterations):
    # Select a random mini-batch
    batch_mask = np.random.choice(X_train.shape[0], batch_size)
    X_batch = X_train[batch_mask]
    y_batch = y_train[batch_mask]

    # Compute gradients using backpropagation
    grads = network.gradient(X_batch, y_batch)

    # Update the weights and biases
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grads[key]

    # Print loss every 1000 iterations
    if i % 1000 == 0:
        loss = network.loss(X_batch, y_batch)
        print(f"Iteration {i}, Loss: {loss}")

# Save the trained model
with open("utekar_mnist_model.pkl", "wb") as f:
    pickle.dump(network, f)

print("Training complete. Model saved as 'utekar_mnist_model.pkl'.")
