import os

import numpy as np
from PIL import Image, ImageDraw

# Ensure output directory exists
output_dir = 'handwritten_digits'
os.makedirs(output_dir, exist_ok=True)


# Define function to create images
def create_digit_image(digit, variant, size=(28, 28)):
    # Create a blank grayscale image
    image = Image.new('L', size, color=255)  # 'L' mode for grayscale
    draw = ImageDraw.Draw(image)

    # Choose font (make sure to have a font that supports numbers; optional)
    # font = ImageFont.truetype("arial.ttf", 20)

    # Draw digit with a slight random offset for variation
    text = str(digit)
    draw.text((7 + np.random.randint(-2, 3), 4 + np.random.randint(-2, 3)),
              text, fill=np.random.randint(0, 100))  # varying shades of gray

    # Save the image
    filename = f"{output_dir}/{digit}_{variant}.png"
    image.save(filename)
    print(f"Saved {filename}")


# Generate 5 images per digit (0-9)
for digit in range(10):
    for variant in range(5):
        create_digit_image(digit, variant)
