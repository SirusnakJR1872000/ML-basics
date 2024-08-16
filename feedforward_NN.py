# Implementing a Simple Feedforward Neural Network using TensorFlow

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# consider an example array
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR problem

# build a simple feedforward neural network
model = Sequential()
model.add(Dense(4, input_dim = 2, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# compile the model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# train the model
model.fit(x, y, epochs = 100, verbose = 1)

# make prediction
prediction = model.predict(x)
print("predictions :", prediction)



