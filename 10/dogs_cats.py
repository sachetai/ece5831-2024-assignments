import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib
import numpy as np
import matplotlib.pyplot as plt


class DogsCats:
    CLASS_NAMES = ['dog', 'cat']
    IMAGE_SHAPE = (180, 180, 3)
    BATCH_SIZE = 32
    EPOCHS = 20

    def __init__(self):
        self.base_path = pathlib.Path('/Users/sachetutekar/PycharmProjects/HW10/dogs-vs-cats-original')
        self.train_path = self.base_path / 'train'
        self.test_path = self.base_path / 'test1'
        self.train_dataset = None
        self.test_images = None
        self.model = None

    def _make_dataset(self, subset_name):
        """Create tf.data.Dataset object for training."""
        if subset_name == 'train':
            return tf.keras.preprocessing.image_dataset_from_directory(
                self.train_path,
                image_size=self.IMAGE_SHAPE[:2],
                batch_size=self.BATCH_SIZE,
                label_mode='int',
                shuffle=True  # Only shuffle the dataset
            )
        else:
            raise ValueError("Invalid subset name for dataset. Use 'train'.")

    def load_test_images(self):
        """Load all test images from the test directory."""
        image_paths = list(self.test_path.glob('*'))
        test_images = []
        test_file_names = []

        for image_path in image_paths:
            img = tf.keras.utils.load_img(image_path, target_size=self.IMAGE_SHAPE[:2])
            img_array = tf.keras.utils.img_to_array(img)
            test_images.append(img_array)
            test_file_names.append(image_path.name)

        self.test_images = (np.array(test_images), test_file_names)

    def make_dataset(self):
        """Create the training dataset."""
        self.train_dataset = self._make_dataset('train')

    def build_network(self, augmentation=True):
        """Build the neural network model with transfer learning."""
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,  # Exclude the final classification layers
            weights='imagenet',  # Load pretrained weights
            input_shape=self.IMAGE_SHAPE
        )

        # Freeze base model layers to retain pre-trained weights
        base_model.trainable = False

        model = models.Sequential()

        # Data Augmentation (Advanced)
        if augmentation:
            model.add(layers.RandomFlip('horizontal'))
            model.add(layers.RandomRotation(0.2))
            model.add(layers.RandomZoom(0.2))
            model.add(layers.RandomContrast(0.2))  # More augmentation for better generalization

        # Base model
        model.add(base_model)

        # Add custom layers for classification
        model.add(layers.GlobalAveragePooling2D())  # Global Pooling instead of Flatten
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification (dog or cat)

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Lower learning rate
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        self.model = model

    def train(self, model_name):
        """Train the model and save it."""
        if not self.model or not self.train_dataset:
            raise ValueError("Model or datasets not initialized. Call make_dataset() and build_network() first.")

        # Use a learning rate scheduler to adjust the learning rate during training
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10 ** (epoch / 20))

        # Add EarlyStopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            patience=3,  # Stop after 3 epochs of no improvement
            restore_best_weights=True,  # Restore the best weights after stopping
            verbose=1
        )

        # Fit the model
        history = self.model.fit(
            self.train_dataset,
            epochs=self.EPOCHS,
            validation_data=self.train_dataset,  # You can use a separate validation dataset
            callbacks=[lr_schedule, early_stopping]  # Add early stopping and learning rate scheduler
        )

        # Save the trained model
        self.model.save(model_name)

        # Plot accuracy and loss
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

    def predict(self):
        """Make predictions for the test images."""
        if not self.model or self.test_images is None:
            raise ValueError("Model not loaded or test images not available. Load model and test images first.")

        test_images, file_names = self.test_images
        predictions = self.model.predict(test_images)

        # Display predictions
        for idx, pred in enumerate(predictions):
            predicted_class = self.CLASS_NAMES[int(pred > 0.5)]
            confidence = pred[0]

            # Display image with prediction
            plt.imshow(test_images[idx].astype("uint8"))
            plt.axis('off')
            plt.title(f"{file_names[idx]}: {predicted_class} ({confidence:.2f})")
            plt.show()


# Main execution
if __name__ == '__main__':
    dogscats = DogsCats()

    # Step 1: Create datasets
    dogscats.make_dataset()

    # Step 2: Load test images
    dogscats.load_test_images()

    # Step 3: Build and train the network
    dogscats.build_network()
    dogscats.train("model.dogs-vs-cats.keras")

    # Step 4: Predict on test images
    dogscats.predict()
