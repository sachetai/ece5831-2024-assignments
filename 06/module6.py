import sys
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from two_layer_net_with_back_prop import TwoLayerNetWithBackProp


def load_image(filename):
    # Load the image, convert to grayscale, and resize to 28x28 pixels
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (28, 28))
    img_normalized = img_resized / 255.0  # Normalize pixel values to [0, 1]
    return img_normalized.reshape(1, -1)  # Flatten to match the model's input shape


def main():
    # Check for the correct number of input arguments
    if len(sys.argv) != 3:
        print("Usage: python module6.py <image_filename> <actual_digit>")
        sys.exit(1)

    filename = sys.argv[1]  # The path to the image file
    actual_digit = int(sys.argv[2])  # The actual digit label

    # Load the trained model
    with open("utekar_mnist_model.pkl", "rb") as f:
        network = pickle.load(f)

    # Load the image
    image = load_image(filename)

    # Display the input image
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f"Input Image - Expected Digit: {actual_digit}")
    plt.show()

    # Make a prediction
    prediction = np.argmax(network.predict(image))

    # Display the result
    if prediction == actual_digit:
        print(f"Success: Image {filename} is for digit {actual_digit} and recognized as {prediction}.")
    else:
        print(f"Fail: Image {filename} is for digit {actual_digit} but the inference result is {prediction}.")


if __name__ == "__main__":
    main()
