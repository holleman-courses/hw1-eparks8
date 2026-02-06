#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[7]:


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


# In[8]:


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


# In[14]:


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


# In[26]:


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


# In[9]:

model1 = build_model1()
model2 = build_model2()
model3 = build_model3()
model50k = build_model50k()

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


# In[10]:


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


# In[15]:


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


# In[27]:


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


# In[33]:


test_img = np.array(tf.keras.utils.load_img(
    'OneDrive/UNCC/ECGR4127/test_image.jpg',
    grayscale=False,
    color_mode='rgb',
    target_size=(32, 32))
)

# 2. Preprocess the image
# Scale pixels to [0, 1] and add the 'batch' dimension
input_arr = test_img.astype('float32') / 255.0
input_arr = np.expand_dims(input_arr, axis=0)  # Becomes (1, 32, 32, 3)

# 3. Run the prediction
# Use model2 (or whichever model you want to test)
predictions = model3.predict(input_arr)

# 4. Interpret the result
# Since the model uses from_logits=True, we find the index of the highest logit
predicted_class_idx = np.argmax(predictions[0])

# CIFAR-10 class labels in order
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

print(f"The model predicts this image is a: {classes[predicted_class_idx]}")


# In[37]:


model50k.save("best_model.h5")


# In[ ]:




