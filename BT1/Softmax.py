from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
drug = pd.read_csv('drug.csv')
le = preprocessing.LabelEncoder()
drug = drug.apply(le.fit_transform)
dt_Train, dt_Test = train_test_split(drug, test_size=0.3, shuffle = True, random_state=16)
X_train = dt_Train.iloc[:, :5]
y_train = dt_Train.iloc[:, 5]
X_test = dt_Test.iloc[:, :5]
y_test = dt_Test.iloc[:, 5]
model = LogisticRegression(C=1e5,multi_class='multinomial', solver='lbfgs', max_iter=4000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
# accuracy = model.score(X_test, y_test)
print('Accuracy:', accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)




