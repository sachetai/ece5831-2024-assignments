import os
import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


class Reuters:
    def __init__(self, vocab_size=10000, maxlen=200, num_classes=None):
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self):
        # Load the Reuters dataset
        (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=self.vocab_size)
        self.num_classes = max(y_train) + 1  # Number of output classes
        self.x_train = pad_sequences(x_train, maxlen=self.maxlen, padding='post')
        self.x_test = pad_sequences(x_test, maxlen=self.maxlen, padding='post')
        self.y_train = to_categorical(y_train, self.num_classes)
        self.y_test = to_categorical(y_test, self.num_classes)

    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, 128, input_length=self.maxlen),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, epochs=10, batch_size=32):
        self.history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )

    def save_model(self):
        # Path to save the model
        filepath = '/Users/sachetutekar/PycharmProjects/ECE 5831/HW11/sachetutekar_REUTERS.keras'
        if self.model:
            self.model.save(filepath)
            print(f"Model saved as {filepath}")
        else:
            print("No model found to save.")

    def plot_loss(self):
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss over Epochs')
        plt.show()

    def plot_accuracy(self):
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy over Epochs')
        plt.show()

    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")


# Usage example
if __name__ == "__main__":
    reuters_classifier = Reuters()
    reuters_classifier.prepare_data()
    reuters_classifier.build_model()
    reuters_classifier.train(epochs=5)
    reuters_classifier.plot_loss()
    reuters_classifier.plot_accuracy()
    reuters_classifier.evaluate()
    reuters_classifier.save_model()