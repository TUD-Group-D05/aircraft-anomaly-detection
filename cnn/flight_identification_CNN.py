""" The Plan
        The end goal is to train a CNN using the labelled rajectories from before.
        Make sure that the labelled images are in a folder named images_labelled.
        1. Images are converted into Numpy arrays with a grayscale color space and 
           are normalised to have a fractional value between 0 and 1
        2. Classification names are defined as a list in order.
        3. Labels are extracted for every image from its filename.
        4. About three quarters of the dataset is used to train the neural network.
           and the rest are used for testing.
        5. A TensorFlow model is instantiated in sequential mode.
        6. Subsequent groups of layers are added to the model.
        7. The model is then trained with the training images and labels for 15 epochs.
        8. The model is saved to the folder 'saved_models'.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def flight_identification(model_name):
    current_dir = os.getcwd() # Get current working directory path
    directory = "images_labelled" # Name of the labelled image folder
    # Create list containing the path of every image
    image_paths = [os.path.join(current_dir, directory, file) 
        for file in os.listdir(os.path.join(current_dir, directory)) if file.endswith('.png')]
    # Read every image from its path as a numpy array and divide by 255 to get a fraction between 0 and 1. The images are all saved in a list.   
    images = [(cv2.imread(path, cv2.IMREAD_GRAYSCALE)/255) for path in image_paths]
    # Reshape the numpy array such that it is in the form necessary for tensorflow
    number_of_images = len(images)
    images = np.reshape(images, (number_of_images, 369, 369, 1))

    training_images = np.array(images[111:number_of_images]) # Take the last 400 out of the 511 (len(images)=511) images as the training images 
    # (it's 511 because that is when we got bored of labelling the images)
    testing_images = np.array(images[:111]) # Take the remaining 111 out of 511 (len(images)=511) images ad the testing images 

    # Define all of the classes that a flight can fall into, note that the order here matters since each of them has an index
    class_names = ['Normal', 'Go Around', 'Single Holding Pattern', 
        'Double Holding Pattern', 'General Delay', 'Extreme Delay', 'Unknown']

    # Find the index of the class name by getting the ID from the image name. The ID is the 5th last character in the filename.
    labels = np.array([int((path).split("\\")[-1][-5]) for path in image_paths])
    training_labels = labels[111:number_of_images] # Take the last 400 labels for training
    testing_labels = labels[:111] # Take the first 111 of the set for testing.

    input_shape = (400, 369, 369, 1) # Input shape to tensor

    # Create a sequential model, where layers can be added to it on separate lines
    model = models.Sequential()

    # Add Convolutional 2D ('relu') and Max Pooling 2D layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(input_shape[1:]), strides=(2, 2)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Add Convolutional 2D ('relu'), Max Pooling 2D, and Dropout layers
    model.add(layers.Conv2D(128, (3, 3), activation = 'relu', strides=(1, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.05))

    # Add Flatten, Dense (32), and Dense (7) layers
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(7))

    # Print a summary of the CNN
    model.summary()

    # Compile the model using the Adam algorithm and space categorical crossentropy loss function.
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    print([np.shape(a) for a in [training_images, training_labels]])

    history = model.fit(training_images, training_labels, epochs=15, 
                        validation_data=(testing_images, testing_labels))

    # Make a plot of the accuracy and the validation accuracy against the epoch number
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    _, test_acc = model.evaluate(testing_images,  testing_labels, verbose=2)
    print(test_acc) # Print final accuracy

    # If folder doesn't exist, make it.
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    # Save the model in the saved_models folder
    model.save(f'saved_models/{model_name}')