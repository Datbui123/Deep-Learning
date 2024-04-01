import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout,Activation
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras.datasets import cifar10

# Load du lieu CIFAR10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_val, y_val = X_train[40000:50000,:], y_train[40000:50000]
X_train, y_train = X_train[:40000,:],y_train[:40000]
# print(X_train.shape)

# Reshape lai du lieu
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_val = X_val.reshape(X_val.shape[0], 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)

#Image Data Augmentation
from keras.preprocessing.image import ImageDataGenerator
train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True,zoom_range=.1 )
val_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True,zoom_range=.1)
test_generator = ImageDataGenerator(rotation_range=2, horizontal_flip= True,zoom_range=.1)

#Fitting the augmentation defined above to the data
train_generator.fit(X_train)
val_generator.fit(X_val)
test_generator.fit(X_test)

# One hot encoding label (Y)
Y_train = np_utils.to_categorical(y_train, 10)
Y_val = np_utils.to_categorical(y_val,10)
Y_test = np_utils.to_categorical(y_test, 10)
print("Dữ liệu y ban đầu ", y_train[0])
print("Dữ liệu y sau one hot encoding ", y_train[0])

# Model definition
model = Sequential()
# Add Convolutional layer with 32 kernel, kernel size 3*3
# Use sigmoid as activation and desire input_shape for the first layer
model.add(Conv2D(32, (3,3), activation = "sigmoid", input_shape=(32,32,3)))
# Add Convolutional layer
model.add(Conv2D(32, (3,3), activation = "sigmoid"))
# Add Max Pooling Layer
model.add(MaxPooling2D(pool_size=(2,2)))
# Flatten layer from tensor to vector
model.add(Flatten())
# Add Fully Connected Layer with 128 nodes and using sigmoid
model.add(Dense(256,activation="sigmoid"))
model.add(Dense(128,activation="sigmoid"))
# Output layer with 10 node and using softmax function to get probability
model.add(Dense(10, activation="softmax"))

# Compile model, determing which loss_function is used, method to optimize loss function
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# Train model with data
H = model.fit(train_generator.flow(X_train, Y_train, batch_size=32), validation_data=(X_val, Y_val), epochs=10, verbose=1)

# Evaluate model with test set
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)

# Predict image
plt.imshow(X_test[0].reshape(32,32,3), cmap="jet")
y_predict = model.predict(X_test[0].reshape(1,32,32,3))
print("Giá trị dự đoán: ", np.argmax(y_predict))