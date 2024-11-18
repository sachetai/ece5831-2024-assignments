from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np


class LeNet:
    def __init__(self):
        self.model = None  # Placeholder for the model

    def _create_lenet(self):
        # Define the LeNet architecture with ReLU activation for better gradient flow
        self.model = models.Sequential([
            layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1), padding='same'),
            layers.AveragePooling2D(pool_size=(2, 2)),
            layers.Conv2D(16, (5, 5), activation='relu'),
            layers.AveragePooling2D(pool_size=(2, 2)),
            layers.Conv2D(120, (5, 5), activation='relu'),
            layers.Flatten(),
            layers.Dense(84, activation='relu'),
            layers.Dense(10, activation='softmax')  # Output layer for 10 classes
        ])

    def _compile(self):
        # Compile the model with Adam optimizer for better convergence
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def _preprocess(self, images):
        # Ensure images have 4 dimensions (batch_size, height, width, channels)
        if len(images.shape) == 3:  # If the image is a single image or 1 sample
            images = np.expand_dims(images, axis=0)  # Add batch dimension

        # Resize to 32x32, add channel dimension, and normalize
        images = tf.image.resize(images, [32, 32])  # Resize to 32x32
        images = images / 255.0  # Normalize pixel values to [0, 1]
        return images

    def train(self, x_train, y_train, epochs=10, batch_size=32, validation_data=None):
        # Add EarlyStopping callback to prevent overfitting
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Train the model with validation monitoring
        self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[early_stop]
        )

    def save(self, model_path_name):
        # Save the model with a specified name
        model_filename = f"{model_path_name}.keras"
        self.model.save(model_filename)

    def load(self, model_path_name):
        # Load a previously saved model
        model_filename = f"{model_path_name}.keras"
        self.model = models.load_model(model_filename)

    def predict(self, images):
        # Preprocess images and make predictions
        images = self._preprocess(images)  # Ensure the images are preprocessed correctly
        predictions = self.model.predict(images)
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        return predicted_classes, confidences

    def load_and_preprocess_data(self):
        # Load and preprocess MNIST dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Expand dimensions to have a channel for grayscale (28, 28) -> (28, 28, 1)
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

        # Apply preprocessing (resize to 32x32, normalize, etc.)
        x_train = self._preprocess(x_train)
        x_test = self._preprocess(x_test)

        # One-hot encode labels
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        return (x_train, y_train), (x_test, y_test)


# Create the LeNet model
lenet_model = LeNet()
lenet_model._create_lenet()
lenet_model._compile()

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = lenet_model.load_and_preprocess_data()

# Train the model with validation set
lenet_model.train(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Save the model
lenet_model.save('utekar_cnn_model')

# Test the model
test_images = x_test[:10]  # Test with the first 10 images
test_labels = np.argmax(y_test[:10], axis=1)  # Get true labels
predicted_classes, confidences = lenet_model.predict(test_images)

# Print predictions
for i, (true_label, predicted, confidence) in enumerate(zip(test_labels, predicted_classes, confidences)):
    print(f"Test Image {i}: True Label = {true_label}, Predicted = {predicted}, Confidence = {confidence:.2f}")
