from collections import OrderedDict
import numpy as np

class TwoLayerNetWithBackProp:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # Initialize weights
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        # Define layers and initialize them
        self.layers = OrderedDict()
        self.update_layers()
        self.last_layer = SoftmaxWithLoss()

    def update_layers(self):
        """Update layers with current weights (used when loading model parameters)."""
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    def backward(self, x, t):
        # Forward pass
        self.loss(x, t)
        
        # Backward pass
        dout = 1
        dout = self.last_layer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        # Store gradients
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        
        return grads

# Loading the Model with Updated Layers
import pickle

# Load model parameters
with open('utekar_mnist_model.pkl', 'rb') as f:
    network = TwoLayerNetWithBackProp(input_size=784, hidden_size=50, output_size=10)
    network.params = pickle.load(f)
    network.update_layers()  # Important: call update_layers() after loading


