import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

def load_labels(label_file):
    """Load class labels from a file."""
    with open(label_file, 'r') as f:
        return f.read().splitlines()

def classify_image(image_path, model, class_names):
    """Classify the given image and return the result."""
    img = cv2.imread("C://ECE5831/images/01.jpg")
    img_resized = cv2.resize(img, (224, 224))  # Model's input size
    img_normalized = np.array(img_resized, dtype=np.float32) / 255.0  # Normalize image
    img_expanded = np.expand_dims(img_normalized, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_expanded)
    class_idx = np.argmax(predictions)
    confidence_score = predictions[0][class_idx]

    # Display the image using matplotlib
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'Class: {class_names[class_idx]} \n Confidence Score: {confidence_score:.4f}')
    plt.axis('off')
    plt.show()

    return class_names[class_idx], confidence_score

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python rock-paper-scissors.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Load the trained model
    model = tf.keras.models.load_model('model.h5')

    # Load class labels
    class_names = load_labels('labels.txt')

    # Classify the image
    label, confidence = classify_image(image_path, model, class_names)
    print(f'Class: {label}')
    print(f'Confidence Score: {confidence:.4f}')
