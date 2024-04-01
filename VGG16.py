import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.applications import VGG16, MobileNet
from keras.optimizers import Adam

#Load du lieu MNIST
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_val, y_val = X_train[50000:60000,:], y_train[50000:60000]
X_train, y_train = X_train[:50000,:], y_train[:50000]
print(X_train.shape)

# Resape lại dữ liệu cho đúng kích thước mà keras yêu cầu
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_val = X_val.reshape(X_val.shape[0], 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)

#One hot encoding label (Y)
Y_train = np_utils.to_categorical(y_train, 10)
Y_val = np_utils.to_categorical(y_val, 10)
Y_test = np_utils.to_categorical(y_test, 10)
print("Dữ liệu ban đầu ", y_train[0])
print("Dữ liệu y sau one-hot encoding ",Y_train[0])


# base_network = MobileNet(input_shape=(32, 32, 3), include_top = False, weights = 'imagenet')
# flat = Flatten()
# den = Dense(1, activation='sigmoid')

# Tải mô hình VGG16 đã được huấn luyện trước và bỏ đi lớp fully connected cuối cùng
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
 # Đóng băng các trọng số của các lớp Convolutional trong mô hình VGG16 
for layer in base_model.layers: 
	layer.trainable = False 


# model = Sequential([base_network, flat, den])
# model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics = ['accuracy'])
# model.summary()


# # Định nghĩa model
# model = Sequential()
# model.add(base_network)
# model.add(Flatten())
# model.add(Dense(150, activation='sigmoid'))
# model.add(Dense(120, activation='sigmoid'))
# model.add(Dense(84, activation='sigmoid'))
# model.add(Dense(10, activation='softmax'))
# model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics = ['accuracy'])

# Xây dựng mô hình mới trên cơ sở mô hình VGG16 đã tải 
model = Sequential() 
model.add(base_model) 
model.add(Flatten())
model.add(Dense(256, activation='relu')) 
model.add(Dense(10, activation='softmax'))
 # Compile và huấn luyện mô hình mới với dữ liệu mới 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))


# Thực hiện train model với dữ liệu
H = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=32, epochs=10, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)
print(score)

plt.imshow(X_test[0], cmap='gray')
y_predict = model.predict(X_test[0].reshape(1,32,32,3))
print('Giá trị dự đoán: ', np.argmax(y_predict))