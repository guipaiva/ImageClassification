from loader import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD


model = Sequential()
model.add(Dense(125, input_shape=(784,), activation="sigmoid"))
model.add(Dense(10, activation="sigmoid"))

model.compile(optimizer= SGD(0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(x,labels,epochs=5)

