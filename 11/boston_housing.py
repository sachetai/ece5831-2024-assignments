import os
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt


class BostonHousing:
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.history = None

    def prepare_data(self):
        # Correctly load the Boston Housing dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = boston_housing.load_data()

        # Normalize the data (feature scaling)
        mean = self.x_train.mean(axis=0)
        std = self.x_train.std(axis=0)

        self.x_train = (self.x_train - mean) / std
        self.x_test = (self.x_test - mean) / std

    def build_model(self):
        # Build the regression model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.x_train.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)  # Single output for regression
        ])
        self.model.compile(optimizer='adam',
                           loss='mse',  # Mean Squared Error for regression
                           metrics=['mae'])  # Mean Absolute Error for evaluation

    def train(self, epochs=250, batch_size=32):
        self.history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0  # Suppress verbose output for cleaner logs
        )
        print("Training complete.")

    def save_model(self):
        # Save model to the specified path
        filepath = '/Users/sachetutekar/PycharmProjects/ECE 5831/HW11/sachetutekar_BOSTON_HOUSING.keras'
        if self.model:
            self.model.save(filepath)
            print(f"Model saved as {filepath}")
        else:
            print("No model found to save.")

    def plot_loss(self):
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.title('Loss over Epochs')
        plt.show()

    def evaluate(self):
        # Evaluate the model on the test data
        loss, mae = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"Test Loss (MSE): {loss}")
        print(f"Test MAE: {mae}")

        # Calculate custom "accuracy" based on tolerance
        y_pred = self.model.predict(self.x_test)
        accuracy = self.custom_accuracy(self.y_test, y_pred)
        print(f"Custom Accuracy (within 5%): {accuracy.numpy()*100}%")

    def custom_accuracy(self, y_true, y_pred, tolerance=0.05):
        # Calculate percentage of predictions within tolerance
        accurate_predictions = tf.abs(y_true - y_pred) / y_true < tolerance
        accuracy = tf.reduce_mean(tf.cast(accurate_predictions, tf.float32))
        return accuracy


# Usage example
if __name__ == "__main__":
    boston_housing_model = BostonHousing()  # Rename variable to avoid conflict
    boston_housing_model.prepare_data()
    boston_housing_model.build_model()
    boston_housing_model.train(epochs=50)
    boston_housing_model.plot_loss()
    boston_housing_model.evaluate()
    boston_housing_model.save_model()

'''Explanation of Custom Accuracy:

custom_accuracy function:
It compares the predicted values to the true values, and counts how many predictions are within a specified tolerance (default is 5%).
tf.abs(y_true - y_pred) / y_true < tolerance calculates if the difference between predicted and actual values is within 5% of the actual value.
The accuracy is then calculated as the average of true/false values. After evaluation, you'll see the custom "accuracy" which tells you how many predictions are within a ±5% tolerance.'''
