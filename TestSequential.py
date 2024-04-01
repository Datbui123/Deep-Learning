import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
# Prepare the training dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# Reserve 10,000 samples for validation.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)



model=Sequential()
model.add(Dense(256, activation='relu'))
model.add(Dense(32, activation='relu'))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=5, verbose=1)
score=model.evaluate(x_test, y_test, verbose=0)
y_pred=model.predict(x_test)


print("Scoreeeeeeeeeeeee: ",score)