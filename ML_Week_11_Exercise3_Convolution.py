# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:41:07 2021
@author: sila
"""

# Import necessary libraries
import numpy as np
from sklearn.datasets import load_sample_image
import tensorflow as tf
import matplotlib.pyplot as plt

# Load two sample images (China and Flower) and normalize their pixel values to be in the range [0, 1]
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255

# Combine the two images into a single array to create a batch of images
images = np.array([china, flower])
batch_size, height, width, channels = images.shape

# Define a set of convolutional filters
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)

# Set the values in the filters to define specific patterns (vertical and horizontal lines)
filters[:, 3, :, 0] = 1  # Vertical line filter
filters[3, :, :, 1] = 1  # Horizontal line filter

# Perform a 2D convolution operation on the input images using the defined filters
outputs = tf.nn.conv2d(images, filters, strides=2, padding="SAME")

# Display one of the output feature maps (the second feature map of the first image)
plt.imshow(outputs[0, :, :, 1])
plt.imshow(outputs[1, :, :, 1])

numbers = []
plt.show()
