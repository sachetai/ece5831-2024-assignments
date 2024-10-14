import cv2
import numpy as np
import tensorflow as tf

def load_labels(label_file):
    """Load class labels from a file."""
    with open(label_file, 'r') as f:
        return f.read().splitlines()

# Load the trained model from Teachable Machine
model = tf.keras.models.load_model('model.h5')

# Load class labels
class_names = load_labels('labels.txt')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the image for model prediction
    img = cv2.resize(frame, (224, 224))
    img_normalized = np.array(img, dtype=np.float32) / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)

    # Predict the class
    predictions = model.predict(img_expanded)
    class_idx = np.argmax(predictions)
    confidence_score = predictions[0][class_idx]
    prediction_label = class_names[class_idx]

    # Display the prediction on the frame
    cv2.putText(frame, f'Prediction: {prediction_label} ({confidence_score:.4f})', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Rock Paper Scissors', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#Couldn't record video due to some unsolvable error. However, an attempt is made. 
