import tensorflow as tf
from tensorflow.keras.datasets import imdb as keras_imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

class Imdb:
    def __init__(self, vocab_size=10000, maxlen=200):
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.model = None
        self.history = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self):
        # Load IMDB dataset
        (x_train, y_train), (x_test, y_test) = keras_imdb.load_data(num_words=self.vocab_size)
        self.x_train = pad_sequences(x_train, maxlen=self.maxlen, padding='post')
        self.x_test = pad_sequences(x_test, maxlen=self.maxlen, padding='post')
        self.y_train = y_train
        self.y_test = y_test

    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, 64, input_length=self.maxlen),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def train(self, epochs=10, batch_size=32):
        self.history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )

    def plot_loss(self):
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss over Epochs')
        plt.show()

    def save_model(self):
        # Explicitly set the path for the model
        filepath = '/Users/sachetutekar/PycharmProjects/ECE 5831/HW11/sachetutekar_IMDB.keras'

        if self.model:
            self.model.save(filepath)
            print(f"Model saved as {filepath}")
        else:
            print("No model found to save.")

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


if __name__ == "__main__":
    # Step 1: Initialize the IMDB object
    imdb = Imdb()

    # Step 2: Prepare the data
    imdb.prepare_data()
    print("Data preparation completed.")

    # Step 3: Build the model
    imdb.build_model()
    print("Model built successfully.")

    # Step 4: Train the model
    print("Training started...")
    imdb.train(epochs=5, batch_size=32)
    print("Training completed.")

    # Step 5: Plot loss and accuracy
    print("Plotting training loss...")
    imdb.plot_loss()

    print("Plotting training accuracy...")
    imdb.plot_accuracy()

    # Step 6: Evaluate the model
    print("Evaluating the model on test data...")
    imdb.evaluate()

    imdb.save_model()

