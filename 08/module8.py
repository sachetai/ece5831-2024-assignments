import sys
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_model(model_path='utekar_cnn_model.keras'):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def load_image(filename):
    """
    Load an image in grayscale mode.
    Args:
        filename (str): The path to the image file.
    Returns:
        Image: The loaded grayscale image.
    """
    image = Image.open(filename).convert("L")
    return image


def predict_digit(image):
    # Simulating prediction by asking user for input
    return int(input("Enter predicted digit (for testing): "))


def main():
    if len(sys.argv) != 3:
        print("Usage: python module8.py <image_filename> <digit>")
        sys.exit(1)

    image_filename = sys.argv[1]
    actual_digit = int(sys.argv[2])

    # Load and display the image
    image = load_image(image_filename)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()

    # Predict digit
    predicted_digit = predict_digit(image)

    # Display success or failure message
    if predicted_digit == actual_digit:
        print(f"Success: Image {image_filename} is for digit {actual_digit} recognized as {predicted_digit}.")
    else:
        print(
            f"Fail: Image {image_filename} is for digit {actual_digit} but the inference result is {predicted_digit}.")


if __name__ == "__main__":
    main()
