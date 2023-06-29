from numpy import sqrt
from numpy import asarray
from pandas import read_csv
from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix , -1]
        X.append(seq_x)
        y.append(seq_y)
    return asarray(X), asarray(y)

filename = "jena_climate_2009_2016.csv"
all_attributes = ['p (mbar)', 'VPmax (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'rho (g/m**3)', 'wv (m/s)','T (degC)']
data = read_csv(filename, index_col=False, usecols=all_attributes, encoding='utf-8')[all_attributes]
print(data)

from sklearn import preprocessing
# le = preprocessing.LabelEncoder()
# data = data.apply(le.fit_transform)
# retrieve the values
values = data.values.astype('float32')
# specify the window size
n_steps = 6
# split into samples
X, y = split_sequence(values, n_steps)
print(X)
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=16)
print(X_train.shape, X_test.shape,X_val.shape, y_train.shape, y_test.shape,y_val.shape)

from keras.layers import Dense, Activation, Dropout, BatchNormalization
model = Sequential()
model.add(LSTM(120, activation='relu', kernel_initializer='he_normal', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(120, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=1, validation_data=(X_val, y_val))

y_pred = model.predict(X_test)
y_pred=y_pred.reshape(y_pred.shape[0])
print(y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("R2 score:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)