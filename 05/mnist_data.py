import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt

class MnistData:
    def __init__(self):
        # Load the MNIST dataset using TensorFlow/Keras
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = tf.keras.datasets.mnist.load_data()

        # Flatten the images to 784 bytes (28x28 = 784)
        self.train_images = self.train_images.reshape(-1, 784).astype('float32') / 255.0
        self.test_images = self.test_images.reshape(-1, 784).astype('float32') / 255.0

        # Convert labels to one-hot encoding
        self.train_labels_one_hot = tf.keras.utils.to_categorical(self.train_labels, num_classes=10)
        self.test_labels_one_hot = tf.keras.utils.to_categorical(self.test_labels, num_classes=10)

    def load(self):
        """
        Returns:
            (train_images, train_labels): Training images and their corresponding one-hot encoded labels
            (test_images, test_labels): Test images and their corresponding one-hot encoded labels
        """
        return (self.train_images, self.train_labels_one_hot), (self.test_images, self.test_labels_one_hot)

    def display_image(self, images, labels, index):
        """
        Display an image and its corresponding label.

        Args:
            images (ndarray): Array of images.
            labels (ndarray): Array of labels.
            index (int): Index of the image to display.
        """
        # Reshape the image back to 28x28 for display purposes
        image = images[index].reshape(28, 28)
        label = np.argmax(labels[index])  # Convert one-hot label back to integer

        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {label}")
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # If no arguments are passed, print the class description
        print("MnistData class is to load MNIST datasets.")
        print("load()")
        print("    Return (train_images, train_labels), (test_images, test_labels)")
        print("    Each image is flattened to 784 bytes. To display an image, reshaping is necessary.")
        print("    Each label is one-hot-encoded. To get a number, use argmax to get the index where 1 is located.")
    else:
        # If arguments are provided, load the dataset and display the image
        dataset_type = sys.argv[1]  # 'train' or 'test'
        index = int(sys.argv[2])  # Index of the image to display

        mnist = MnistData()
        (train_images, train_labels), (test_images, test_labels) = mnist.load()

        if dataset_type == 'train':
            mnist.display_image(train_images, train_labels, index)
        elif dataset_type == 'test':
            mnist.display_image(test_images, test_labels, index)
        else:
            print("Invalid dataset type. Please specify 'train' or 'test'.")


'''Output:

If you run:
python mnist_data.py train 0
It will display the first training image along with the label printed on the terminal.

If you run the script with no arguments:
python mnist_data.py
It will print:
MnistData class is to load MNIST datasets.
load()
    Return (train_images, train_labels), (test_images, test_labels)
    Each image is flattened to 784 bytes. To display an image, reshaping is necessary.
    Each label is one-hot-encoded. To get a number, use argmax to get the index where 1 is located.
'''
