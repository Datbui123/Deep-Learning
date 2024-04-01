from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def NSE(Y_predict, Y_test):
    return (1-(np.sum((Y_predict-Y_test)**2)/np.sum((Y_test-np.mean(Y_test))**2)))

#hàm chuẩn hóa các chuỗi kí tự có trong dữ liệu
def DataEncoder(dataStr):
	dataConvert = dataStr
	for i, j in enumerate(dataConvert):
		for k in range(0,len(dataConvert[0])):
			if(j[k]=="none"):
				j[k] = 0
			elif(j[k]=="low"):
				j[k] = 1
			elif(j[k]=="med"):
				j[k] = 2
			elif(j[k]=="high"):
				j[k] = 3
	return dataConvert # trả về dữ liệu đã được chuẩn hóa

#đọc dữ liệu từ file
Data = pd.read_csv('Data.csv')

#Chia dữ liệu thành 2 phần: DataX là các thuộc tính, DataY là nhãn của dữ liệu
DataX = DataEncoder(np.array(Data[['Age','Number of vaccinations','Heart disease','Cancer','Lung disease','HIV / AIDS']].values))
DataY = np.array(Data['Mortality rate (%)'].values)

#chia dữ liệu 70% để train 30% để test, có xáo trộn dữ liệu
X_train, X_test, Y_train, Y_test = train_test_split(DataX, DataY, test_size = 0.3, shuffle = True)

#bắt đầu huấn luyện
reg = LinearRegression().fit(X_train,Y_train)

#bắt đầu dự đoán
Y_predict = reg.predict(X_test)

print(f"Vector trọng số W = {reg.coef_}\nHệ số tự do W0 = {reg.intercept_}")

print(f"r2_score = {r2_score(Y_test,Y_predict)}")
print(f"NSE = {NSE(Y_test,Y_predict)}")
print(f"MAE = {mean_absolute_error(Y_test,Y_predict)}")
print(f"RMSE = {mean_squared_error(Y_test,Y_predict,squared=False)}")



