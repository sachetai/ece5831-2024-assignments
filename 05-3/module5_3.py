import sys
import matplotlib.pyplot as plt
from PIL import Image


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
    """
    Placeholder function for model prediction.
    This should be replaced with an actual model inference function.
    Args:
        image (Image): The input image.
    Returns:
        int: Predicted digit (simulated here for testing).
    """
    # Simulating prediction by asking user for input
    return int(input("Enter predicted digit (for testing): "))


def main():
    # Check if the right number of arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python module5-3.py <image_filename> <digit>")
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

# python module5-3.py handwritten_digits/2_4.png 2 for testing if 2 is successfully predicted
