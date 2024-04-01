import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
import numpy as np


df = pd.read_csv('D:\hocmay/WA_Fn-UseC_-HR-Employee-Attrition (1).csv')
data = np.array(df[['Age', 'DailyRate', 'Education', 'HourlyRate', 'JobLevel', 'MonthlyRate', 'PercentSalaryHike']].values)
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=False)
k=5
kf=KFold( n_splits = k, random_state=None)

# tinh sai so trung binh
def error(y, y_pred):
    thucTe = np.array(y)
    duDoan = np.array(y_pred)
    hieuHaiMaTran = np.subtract(thucTe, duDoan)
    binhPhuong = np.square(hieuHaiMaTran)
    return binhPhuong.mean()

max = 999999
#train_index: 4 phan cua dt_Train, tra ve 1 mang gam so hang
#test_index: phan con lai cua dt_Train, tra ve 1 mang gam so hang
for train_index, test_index in kf.split(dt_Train):
    # print("test index:", test_index)
    X_train = dt_Train[train_index,:6]
    X_test =  dt_Train[test_index,:6]
#     print(X_train)
    y_train = dt_Train[train_index,6]
    y_test = dt_Train[test_index,6]

    lr=LinearRegression()
    #k train model ==> k du doan dc
    lr.fit(X_train, y_train)
    Y_pred_train=lr.predict(X_train)
    Y_pred_test=lr.predict(X_test)
    #tong sai so tren 2 tap du lieu train va test
    sum = error(y_train,Y_pred_train)+error(y_test, Y_pred_test)
    # sum = mean_squared_error(y_train,Y_pred_train)+mean_squared_error(y_test, Y_pred_test)
    #tim mo hinh co sai so trung binh nho nhat
    if(sum < max):
        max= sum     
        #Biến reg chứa một đối tượng LinearRegression trong bộ thư viện Scikit-learn 
        regr = lr.fit(X_train, y_train)

    

y_pred= regr.predict(dt_Test[:,:6])
y=np.array(dt_Test[:,6])

c=[]
for i in range (0, len(y)):
    c.append(abs(y[i]-y_pred[i]))
sum_c=0
for i in c:
    sum_c+=i

print("Thực tế  Dự đoán  Chênh lệch")
dung=sai = 0
for i in range (0, len(y)):
    print("%.2f"%y[i], "  ", round(y_pred[i], 2),"  ", round(abs(y[i]-y_pred[i]), 2))
    if(abs(y[i]-y_pred[i])<=1):
        dung+=1
    else:
        sai+=1
print("Số mẫu dự đoán đúng: ", dung)
print("Tỉ lệ dự đoán đúng: ", round(dung*100/len(y)),"%")
print("Số mẫu dự đoán sai: ", sai)
print("Tỉ lệ dự đoán sai: ", round(sai*100/len(y)),"%")
print("Giá trị chênh lệch trung bình: ",sum_c/len(c))
#Coefficient of determination
print("Hiệu suất của mô hình hồi quy tuyến tính: %.2f" % r2_score(y, y_pred))
