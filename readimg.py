#!/usr/bin/env python3
import tensorflow as tf

from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import glob

import numpy as np
import cv2
import random
import math

path_blocked = "dataset/blocked/"
path_free = "dataset/free/"
get_paths  = lambda path:  [f for f in glob.glob(path + "**/*.*", recursive=True)]

blocked = [(cv2.imread(x), 1) for x in get_paths(path_blocked)]

free  = [] 
for i in get_paths(path_free):
    try:
        im = cv2.imread(i)
        free.append((im, 0))
    except:
        print("failed to read the file")
# free = np.array(free)

dataset = free + blocked
random.shuffle(dataset)

slice_index = math.floor(len(dataset)/15)
train_set = dataset[:-slice_index]
test_set = dataset[-slice_index:]

def seperate_data_labels(data_with_labels):
    data = [] 
    label = []
    for d, l in data_with_labels:
        data.append(d)
        label.append(l)
    return np.array(data), np.array(label)

train_images, train_labels = seperate_data_labels(train_set) 
test_images, test_labels = seperate_data_labels(test_set) 


# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
print(type(train_images))
print(train_images.shape)


# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(480, 640, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))


model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
