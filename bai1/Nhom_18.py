import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
# import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('./cars.csv')
dt_Train, dt_Test = train_test_split(data, test_size = 0.3, shuffle = False)

k = 5
kf = KFold(n_splits = k, random_state = None)
def error(y, y_pred):
    l = []
    for i in range(0, len(y)):
        l.append(np.abs(np.array(y[i:i+1]) - np.array(y_pred[i:i+1])))
    return np.mean(l)
max = 9999999
i = 1
for train_index, test_index in kf.split(dt_Train):
    X_train, X_test = dt_Train.iloc[train_index, :5], dt_Train.iloc[test_index, :5]
    y_train, y_test = dt_Train.iloc[train_index, 5], dt_Train.iloc[test_index, 5]
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_train = lr.predict(X_train)
    y_pred_test = lr.predict(X_test)
    
    sum = error(y_train, y_pred_train) + error(y_test, y_pred_test)
    
    if sum < max:
        max = sum
        last = i
        regr = lr.fit(X_train, y_train)
    i = i + 1
y_predict = regr.predict(dt_Test.iloc[:, :5])
y = np.array(dt_Test.iloc[:, 5])

print("Coefficient of determination: %.2f" %error(y_test, y_predict))
print("Thực tế \t Dự đoán \t Chênh lệch")
for i in range (0, len(y)):
    print("%.2f" % y[i], " ", y_predict[i], " ", abs(y[i] - y_predict[i]))