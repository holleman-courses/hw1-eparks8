#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image

def build_model1():
    """4-layer fully-connected model"""
    model = keras.Sequential([
        layers.Flatten(input_shape=(32, 32, 3)),
        layers.Dense(128, activation='leaky_relu'),
        layers.Dense(128, activation='leaky_relu'),
        layers.Dense(128, activation='leaky_relu'),
        layers.Dense(10)
    ])

    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def build_model2():
    """Convolutional Neural Network"""
    model = Sequential([
        layers.Conv2D(32,(3,3),strides=2,padding='same',activation='relu',input_shape=(32,32,3)),
        layers.BatchNormalization(),

        layers.Conv2D(64,(3,3),strides=2,padding='same',activation='relu'),
        layers.BatchNormalization(),

        layers.Conv2D(128,(3,3),padding='same',activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128,(3,3),padding='same',activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128,(3,3),padding='same',activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128,(3,3),padding='same',activation='relu'),
        layers.BatchNormalization(),

        layers.Flatten(),

        layers.Dense(10)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def build_model3():
    """Seperable Convolutional Neural Network"""
    model = Sequential([
        layers.Conv2D(32,(3,3),strides=2,padding='same',activation='relu',input_shape=(32,32,3)),
        layers.BatchNormalization(),

        layers.SeparableConv2D(64,(3,3),strides=2,padding='same',activation='relu'),
        layers.BatchNormalization(),

        layers.SeparableConv2D(128,(3,3),padding='same',activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128,(3,3),padding='same',activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128,(3,3),padding='same',activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128,(3,3),padding='same',activation='relu'),
        layers.BatchNormalization(),

        layers.Flatten(),

        layers.Dense(10)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def build_model50k():
    """Best Model with 50k Parameter Limit"""
    model = Sequential([
        layers.Conv2D(32,(3,3),strides=2,padding='same',activation='relu',input_shape=(32,32,3)),
        layers.BatchNormalization(),

        layers.SeparableConv2D(64,(3,3),strides=2,padding='same',activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(64,(3,3),padding='same',activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(64,(3,3),strides=2,padding='same',activation='relu'),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(10)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

<<<<<<< HEAD
=======

# In[9]:

model1 = build_model1()
model2 = build_model2()
model3 = build_model3()
model50k = build_model50k()

>>>>>>> 60780f0dddd95790890cd5f2e627e1fe598fa593
if __name__ == '__main__':
    # 1. Load Data
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

    # 2. Preprocess (Normalize to 0-1)
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    # 3. Create Validation Split (e.g., last 10,000 images)
    val_images = train_images[-10000:]
    val_labels = train_labels[-10000:]
    train_images = train_images[:-10000]
    train_labels = train_labels[:-10000]

    # 4. Build and Train Model 1
    model1 = build_model1()
    print(model1.summary())

    # Set verbose=1 to see progress, but remember to comment out plt.show()
    history1 = model1.fit(
        train_images, train_labels, 
        epochs=30, 
        validation_data=(val_images, val_labels)
    )

    # Evaluate on test set
    test_loss1, test_acc1 = model1.evaluate(test_images, test_labels)
    print(f"Model 1 Test Accuracy: {test_acc1}")

    model2 = build_model2()
    print(model2.summary())
    
    # Set verbose=1 to see progress, but remember to comment out plt.show()
    history2 = model2.fit(
        train_images, train_labels, 
        epochs=30, 
        validation_data=(val_images, val_labels)
    )
    
    # Evaluate on test set
    test_loss2, test_acc2 = model1.evaluate(test_images, test_labels)
    print(f"Model 2 Test Accuracy: {test_acc2}")
    
    model3 = build_model3()
    print(model3.summary())
    
    # Set verbose=1 to see progress, but remember to comment out plt.show()
    history3 = model3.fit(
        train_images, train_labels, 
        epochs=30, 
        validation_data=(val_images, val_labels)
    )
    
    # Evaluate on test set
    test_loss3, test_acc3 = model1.evaluate(test_images, test_labels)
    print(f"Model 3 Test Accuracy: {test_acc3}")
    
    model50k = build_model50k()
    print(model50k.summary())
    
    # Set verbose=1 to see progress, but remember to comment out plt.show()
    history50k = model50k.fit(
        train_images, train_labels, 
        epochs=30, 
        validation_data=(val_images, val_labels)
    )
    
    # Evaluate on test set
    test_loss50k, test_acc50k = model1.evaluate(test_images, test_labels)
    print(f"Model 3 Test Accuracy: {test_acc50k}")
    model50k.save("best_model.h5")