import argparse
from mnist_data import MnistData
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test MnistData class.')
    parser.add_argument('dataset', choices=['train', 'test'], help="Choose 'train' or 'test' dataset.")
    parser.add_argument('index', type=int, help="Index of the image to display.")
    args = parser.parse_args()

    # Load MNIST dataset using MnistData class
    mnist = MnistData()
    (train_images, train_labels), (test_images, test_labels) = mnist.load()

    if args.dataset == 'train':
        images, labels = train_images, train_labels
    else:
        images, labels = test_images, test_labels

    # Get the image and label for the given index
    image = images[args.index].reshape(28, 28)
    label = np.argmax(labels[args.index])

    # Display the image using matplotlib
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.show()

    # Print the label in the terminal
    print(f"Label of the selected image: {label}")

if __name__ == '__main__':
    main()

''' The first argument specifies whether to use the "train" or "test" dataset,
and the second argument is the index number of the image to display. '''

# python module5-2.py train 5
