import pandas as pd
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import numpy as np

def NSE(y_pred, y_test):
    return (1-(np.sum((y_pred-y_test)**2)/np.sum((y_test-np.mean(y_test))**2)))

data = pd.read_csv('USA_Housing.csv')

dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=False)
x_train = dt_Train.iloc[:, :5]
y_train = dt_Train.iloc[:, 5]
x_test = dt_Test.iloc[:, :5]
y_test = dt_Test.iloc[:, 5]

# model = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.001, learning_rate_init=0.001, max_iter=200)
model = Ridge(alpha=0.0001, fit_intercept=True, copy_X=True, max_iter=None, tol=0.000001, solver='auto', positive=False, random_state=None)

# reg = LinearRegression().fit(x_train,y_train)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# y_pred = reg.predict(x_test)
y=np.array(y_test)

print("Coefficent of determination:%.2f" % r2_score(y_test,y_pred))
print("Thuc te        Du doan           Chenh lech")
for i in range(0,len(y)):
    print("%.2f"%y[i]," ",y_pred[i]," ",abs(y[i]-y_pred[i]))
print('NSE:',NSE(y,y_pred))