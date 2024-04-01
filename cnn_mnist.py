# Include all the required libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
# import np_utils as np_utils
from keras.datasets import mnist

# Load data MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_val, y_val = X_train[50000:60000,:], y_train[50000:60000]
X_train, y_train = X_train[:50000,:],y_train[:50000]
print(X_train.shape)

# Reshape data to fit the size that keras require
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# One hot encoding label (Y)
# Y_train = np_utils.to_categorical(y_train, 10)
Y_train = to_categorical(y_train, 10)
# Y_val = np_utils.to_categorical(y_val,10)
Y_val = to_categorical(y_val,10)
# Y_test = np_utils.to_categorical(y_test, 10)
Y_test = to_categorical(y_test, 10)
print("Dữ liệu y ban đầu ", y_train[0])
print("Dữ liệu y sau one hot encoding ", y_train[0])

# Model definition
model = Sequential()
# Add Convolutional layer with 32 kernel, kernel size 3*3
# Use sigmoid as activation and desire input_shape for the first layer
model.add(Conv2D(32, (3,3), activation = "sigmoid", input_shape=(28,28,1)))
# Add Convolutional layer
model.add(Conv2D(32, (3,3), activation = "sigmoid"))
# Add Max Pooling Layer
model.add(MaxPooling2D(pool_size=(2,2)))
# Flatten layer from tensor to vector
model.add(Flatten())
# Add Fully Connected Layer with 128 nodes and using sigmoid
model.add(Dense(128,activation="sigmoid"))
# Output layer with 10 node and using softmax function to get probability
model.add(Dense(10, activation="softmax"))

# Compile model, determing which loss_function is used, method to optimize loss function
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# Train model with data
H = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),batch_size=32, epochs=10, verbose=1)

# Evaluate model with test set
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)

# Predict image
plt.imshow(X_test[0].reshape(28,28), cmap="gray")
y_predict = model.predict(X_test[0].reshape(1,28,28,1))
print("Giá trị dự đoán: ", np.argmax(y_predict))
