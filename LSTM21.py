import pandas as pd
from numpy import asarray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras import Sequential
from keras.layers import LSTM

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix, :-1], sequence[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return asarray(X), asarray(y)

data = pd.read_csv('data/winequality-red.csv')
values = data.values.astype('float32')
n_steps = 1
X, y = split_sequence(values, n_steps)
# print(X)
# print(y)

from keras.utils import np_utils
for i in range(0,len(y)):
    y[i] = y[i] - 1 

X = X.reshape((X.shape[0], X.shape[1], 11))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
y_train = np_utils.to_categorical(y_train,8)
y_test = np_utils.to_categorical(y_test,8)

model = Sequential()
model.add(BatchNormalization())
model.add(LSTM(500))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(200, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(8, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Đánh giá mô hình trên tập kiểm tra
loss, accuracy = model.evaluate(X_test, y_test)
print(accuracy)