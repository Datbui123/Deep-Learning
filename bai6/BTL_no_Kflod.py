import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('D:\hocmay/WA_Fn-UseC_-HR-Employee-Attrition (1).csv')
data = np.array(df[['Age', 'DailyRate', 'Education', 'HourlyRate', 'JobLevel', 'MonthlyRate', 'PercentSalaryHike']].values)
#chia mô hình thành 2 tập dữ liệu: 70% cho tập train, 30% cho tập test
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=False)

X_train = dt_Train[:,:6]
Y_train = dt_Train[:,6]
X_test = dt_Test[:,:6]
Y_test = dt_Test[:,6]

#Biến reg chứa một đối tượng LinearRegression trong bộ thư viện Scikit-learn 
reg = LinearRegression().fit(X_train,Y_train)

#y dự đoán
y_pred = reg.predict(X_test)
#y thực tế
y = np.array(Y_test)
print("Hiệu suất của mô hình hồi quy tuyến tính: %.2f" % r2_score(Y_test, y_pred))
print("Thuc te  \tDu doan  \tChenh lech")
dung=sai=0
for i in range (0, len(y)):
    print("%.2f"%y[i], "  ", y_pred[i],"  ", abs(y[i]-y_pred[i]))
    if(abs(y[i]-y_pred[i])<=1):
        dung+=1
    else:
        sai+=1
print("Số mẫu dự đoán đúng: ", dung)
print("Tỉ lệ dự đoán đúng: ", round(dung*100/len(y)),"%")
print("Số mẫu dự đoán sai: ", sai)
print("Tỉ lệ dự đoán sai: ", round(sai*100/len(y)),"%")