from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

# private fanction
def compile_and_train_the_model(model, val_data_gen_, total_val_):
    # Compile the model
    # For this tutorial, choose the ADAM optimizer and binary cross entropy loss function. To view training and
    # validation accuracy for each training epoch, pass the metrics argument.
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # Train the model
    # Use the fit_generator method of the ImageDataGenerator class to train the network.

    history_ = model.fit_generator(
        train_data_gen,
        steps_per_epoch= total_train // batch_size,
        epochs=epochs,
        validation_data=val_data_gen_,
        validation_steps=total_val_ // batch_size
    )
    return history_


def plotVSVal(history_):
    # Visualize training results
    # Now visualize the results after training the network.
    acc = history_.history['acc']
    val_acc = history_.history['val_acc']

    loss = history_.history['loss']
    val_loss = history_.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def plotVStest(history_):
    # Visualize training results
    # Now visualize the results after training the network.
    acc = history_.history['acc']
    val_acc = history_.history['val_acc']

    loss = history_.history['loss']
    val_loss = history_.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Test Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Test Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Test Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Test Loss')
    plt.show()


def trainAugmentation():
    # Put it all together
    # Apply all the previous augmentations. Here, you applied rescale, 45 degree rotation, width shift, height shift, horizontal flip and zoom augmentation to the training images.
   image_gen_train = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=45,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,
        zoom_range=0.5
    )

   train_data_gen_ = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                         directory=train_dir,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         class_mode='binary')
   return train_data_gen_

def validationAugmentation():
        image_gen_val = ImageDataGenerator(rescale=1. / 255)
        val_data_gen_ = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=validation_dir,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='binary')
        return val_data_gen_

def creatingModelWithDropout():
    # Creating a new network with Dropouts
    model_new_ = Sequential([
        Conv2D(16, 3, padding='same', activation='relu',
               input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model_new_



# initializations
_URL = 'https://firebasestorage.googleapis.com/v0/b/horses-vs-zebras.appspot.com/o/horses%20vs%20zebras.zip?alt=media&token=08f75cf3-af2b-49e3-bb3c-1183939d5846'
path_to_zip = tf.keras.utils.get_file('horses vs zebras.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'horses vs zebras')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

train_zebras_dir = os.path.join(train_dir, 'zebras')  # directory with our training zebra pictures
train_horses_dir = os.path.join(train_dir, 'horses')  # directory with our training horse pictures
validation_zebras_dir = os.path.join(validation_dir, 'zebras')  # directory with our validation zebra pictures
validation_horses_dir = os.path.join(validation_dir, 'horses')  # directory with our validation horse pictures
test_zebras_dir = os.path.join(test_dir, 'zebras')  # directory with our training zebra pictures
test_horses_dir = os.path.join(test_dir, 'horses')  # directory with our training horse pictures

num_zebras_tr = len(os.listdir(train_zebras_dir))
num_horses_tr = len(os.listdir(train_horses_dir))

num_zebras_val = len(os.listdir(validation_zebras_dir))
num_horses_val = len(os.listdir(validation_horses_dir))

num_zebras_test = len(os.listdir(test_zebras_dir))
num_horses_test = len(os.listdir(test_horses_dir))

total_train = num_zebras_tr + num_horses_tr
total_val = num_zebras_val + num_horses_val
total_test = num_zebras_test + num_horses_test

batch_size = 128
epochs = 19
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our test data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=test_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')


model = creatingModelWithDropout()

train_data_gen = trainAugmentation()
val_data_gen = validationAugmentation()


# compile and train the model with validation
# history = compile_and_train_the_model(model, val_data_gen, total_val)
# plotVSVal(history)

# compile and train the model with test
history = compile_and_train_the_model(model, test_data_gen, total_test)
plotVStest(history)