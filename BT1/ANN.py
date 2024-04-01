import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
data = pd.read_csv('drug.csv')

# Split features and labels
X = data.drop('Drug', axis=1)
y = data['Drug']

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X = X.apply(label_encoder.fit_transform)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=16)

# Scale the input features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# Define the model architecture
model = Sequential()
model.add(BatchNormalization())
model.add(Dense(285, activation='relu', input_shape=(X.shape[1],)))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(5, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=1)
# Evaluate the model on the test set
loss,accuracy  = model.evaluate(X_test, y_test)

y_pred = np.argmax(model.predict(X_test), axis=1)
#print(y_pred, y_test)
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
# accuracy = model.score(X_test, y_test)
print('Accuracy:', accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
