
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split

data = pd.read_csv('USA_Housing.csv')

dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=False, random_state=14)
x_train = dt_Train.iloc[:, :5]
y_train = dt_Train.iloc[:, 5]
x_test = dt_Test.iloc[:, :5]
y_test = dt_Test.iloc[:, 5]

#-----------------------------------------------------------------------------
model=Sequential()
model.add(Dense(150, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse',metrics=['mae'])
model.fit(x_train, y_train,epochs=350, batch_size=32, verbose=1, validation_data=(x_test,y_test))


#-----------------------------------------------------------------------------

# model = Sequential()
# model.add(Dense(120,activation='relu'))
# model.add(BatchNormalization())
# # model.add(Dropout(0.2, input_shape=(60,)))
# model.add(Dense(32,activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(12,activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(10,activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(5,activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(3))
# model.add(BatchNormalization())
# model.add(Dense(1))
# model.add(BatchNormalization())
# model.compile(optimizer='adam',loss='mse',metrics=['mae'])
# model.fit(x_train, y_train, epochs=250, batch_size=32, verbose=1, validation_data=(x_test,y_test))


#-----------------------------------------------------------------------------

# model = Sequential()
# model.add(Dense(128,activation='relu'))
# model.add(Dense(64,activation='relu'))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(1))

# model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# model.fit(x_train, y_train, epochs=250, batch_size=32, verbose=0, validation_data=(x_test, y_test))

#-----------------------------------------------------------------------------

# model = Sequential()
# model.add(Dense(200,input_shape=(x_train.shape[1],)))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(100, activation='relu'))

# model.add(Dropout(0.2))
# model.add(Dense(50, activation='relu'))

# model.add(Dense(1, activation='linear'))
# # optimizer = Adam(learning_rate=0.001)
# model.compile(loss='mean_squared_error',optimizer='adam')
# history = model.fit(x_train,y_train,validation_split=0.1,batch_size=32,epochs=250,verbose=1)
# score = model.evaluate(x_test, y_test, verbose=0)


mse, mae=model.evaluate(x_test, y_test, verbose=0)
y_predict=model.predict(x_test)
# print(score)
print(mse, mae)
print(r2_score(y_test, y_predict))

