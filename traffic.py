import cv2 as cv
import numpy as np
import os
import sys
import tensorflow as tf
from tqdm import tqdm

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CATEGORIES)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    # Get all category folders (digits only)
    categories = [path for path in os.listdir(data_dir) if os.path.isdir(
        os.path.join(data_dir, path)) and path.isdigit()]

    # Wrap outer loop with tqdm for progress bar
    for path in tqdm(categories, desc="Loading categories"):
        folder_path = os.path.join(data_dir, path)
        label = int(path)

        # List all files in folder
        files = [file for file in os.listdir(folder_path) if file.lower().endswith(".ppm")]

        # Optionally, add tqdm here too if there are many files per folder
        for file in tqdm(files, desc=f"Loading images from category {label}", leave=False):
            abs_path = os.path.abspath(os.path.join(folder_path, file))
            if os.path.isfile(abs_path):
                img = cv.imread(abs_path, cv.IMREAD_COLOR)
                if img is not None:
                    img_resized = cv.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    images.append(img_resized)
                    labels.append(label)

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential()

    # First convolutional layer: 32 filters, 3x3 kernel, ReLU activation
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                                     input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Second convolutional layer: 64 filters
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Third convolutional layer: 128 filters
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten before feeding to dense layers
    model.add(tf.keras.layers.Flatten())

    # Fully connected hidden layer with dropout to reduce overfitting
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    # Output layer: NUM_CATEGORIES units with softmax activation for multi-class classification
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax'))

    # Compile model with categorical crossentropy loss and adam optimizer
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    ) 
    return model


if __name__ == "__main__":
    main()
