from tkinter import * 
from tkinter import messagebox , ttk
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn import decomposition
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def DataEncoder(dataStr):
	dataConvert = dataStr
	for i, j in enumerate(dataConvert):
		for k in range(0,1):
			if(j[k]=="unacc"):
				j[k] = 0
			elif(j[k]=="acc"):
				j[k] = 1
	return dataConvert # trả về dữ liệu đã được chuẩn hóa

df = pd.read_csv('data.csv')
X = np.array(df[['price','area','rooms','loca','envi','safety']].values)
y = df['acceptability'].values
v = DataEncoder(np.array(df[['acceptability']].values))

def NSE(Y_predict, Y_test):
    return (1-(np.sum((Y_predict-Y_test)**2)/np.sum((Y_test-np.mean(Y_test))**2)))

max  = 0   #max(id3)
max1 = 0   #max(linear)

for j in range(1, 7):                                   # chạy j từ 1 đến 8(7 mẫu)
    pca = decomposition.PCA(n_components=j)             # khai báo mô hình pca lấy j mẫu 
    print("PCA-components =",j )
    pca.fit(X)                                          # lấy mô hình huấn luyện
    Xbar = pca.transform(X)                             # biến đổi data thành j thuộc tính 
    X_train, X_test, y_train, y_test = train_test_split(Xbar,y, test_size=0.3, shuffle=False)
    
    #sử dụng thuật toán id3    
    id3 = DecisionTreeClassifier(criterion='entropy')   # khai báo mô hình 
    id3 = id3.fit(X_train,y_train)                      # lấy ra mô hình tốt nhất 
    y_id3 = id3.predict(X_test)                         # dự đoán

    d  = 0  #id3
    for i in range(len(y_test)):           # tính phần trăm của 2 thuật toán 
          if(y_id3[i] == y_test[i]):       # id3 dự đoán đúng thì d+1
             d = d+1  
    rate  = d / len(y_id3)                 # tỷ lệ dự đoán đúng của id3
   
    print("phan tram(id3):", rate,"\n","phan tram(svm):""\n")
    if (rate > max):                 # lấy mô hình j thành phần có tỷ lệ dự đoán đúng cao nhất   
        num_pca   = j                # lấy số thuộc tính trong mô hình dự đoán cao nhất
        pca_best  = pca              # lấy mô hình pca có tỷ lệ đoán đúng cao nhất 
        max       = rate             # lấy tỷ lệ đoán đúng cao nhất 
        modelmax_id3  = y_id3        # lấy mô hình id3 dự đoán đúng nhất    
        id3_max = id3                # lấy mô hình id3 
       
       
for j in range(1, 7):                                   # chạy j từ 1 đến 8(7 mẫu)
    pca = decomposition.PCA(n_components=j)             # khai báo mô hình pca lấy j mẫu 
    pca.fit(X)                                          # lấy mô hình huấn luyện
    Xbar = pca.transform(X)                             # biến đổi data thành j thuộc tính 
    X_train, X_test, v_train, v_test = train_test_split(Xbar,v, test_size=0.3, shuffle=False)
    
    lr = LinearRegression()
    lr.fit(X_train, v_train)
    v_pred = lr.predict(X_test)    
   
    if (r2_score(v_test,v_pred) > max):                 # lấy mô hình j thành phần có tỷ lệ dự đoán đúng cao nhất   
        num_pca   = j                # lấy số thuộc tính trong mô hình dự đoán cao nhất
        pca_best  = pca              # lấy mô hình pca có tỷ lệ đoán đúng cao nhất 
        max1       = r2_score(v_test,v_pred)             # lấy tỷ lệ đoán đúng cao nhất 
        modelmax  = lr        
        bestPred  =	v_pred									# lấy mô hình id3 dự đoán đúng nhất

def NSE(v_predict, v_test):
    return (1-(np.sum((v_predict-v_test)**2)/np.sum((v_test-np.mean(v_test))**2)))

print("max(id3)",max, ' d =',num_pca)

#form
form = Tk()
form.title("Dự đoán chất lượng nhà ở:") # tên của cửa sổ form 
form.geometry("750x450")       # kích cỡ của form dài x rộng


lable_ten = Label(form, text = "Nhập thông số kỹ thuật :", font=("Arial Bold", 10), fg="red") # tiêu đề form
lable_ten.grid(row = 1, column = 1, padx = 40, pady = 10)

lable_price = Label(form, text = "Giá :")                      # chỉ dẫn thông tin cần nhập
lable_price.grid(row = 2, column = 1, padx = 40, pady = 10)  # vị trí 
textbox_price = Entry(form)                                  # nhập thông tin 
textbox_price.grid(row = 2, column = 2)                      # vị trí 

lable_area = Label(form, text = "Diện tích :")
lable_area.grid(row = 3, column = 1, pady = 10)
textbox_area = Entry(form)
textbox_area.grid(row = 3, column = 2)

lable_rooms = Label(form, text = "Số phòng :")
lable_rooms.grid(row = 4, column = 1,pady = 10)
textbox_rooms = Entry(form)
textbox_rooms.grid(row = 4, column = 2)

lable_loca = Label(form, text = "Vị trí :")
lable_loca.grid(row = 2, column = 5, pady = 10)
textbox_loca = Entry(form)
textbox_loca.grid(row = 2, column = 6)

lable_envi = Label(form, text = "Môi trường:")
lable_envi.grid(row = 3, column = 5, pady = 10 )
textbox_envi = Entry(form)
textbox_envi.grid(row = 3, column = 6)

lable_safety = Label(form, text = "Độ an toàn:")
lable_safety.grid(row = 5, column = 1, pady = 10 )
textbox_safety = Entry(form)
textbox_safety.grid(row = 5, column = 2)

#=========================================================================================================
#id3             
lbl3 = Label(form, fg="red")              # tạo bảng 
lbl3.grid(column=1, row=9)                # vị trí 
lbl3.configure(text="Tỉ lệ dự đoán đúng của PCA: "+'\n' # tính và in ra các độ đo dựa vào mô hình dự đoán đúng nhất
                           +"Precision: "+str(precision_score(y_test, modelmax_id3, average='micro')*100)+"%"+'\n'
                           +"Recall: "   +str(recall_score(y_test, modelmax_id3, average='micro')*100)+"%"+'\n'
                           +"F1-score: " +str(f1_score(y_test, modelmax_id3, average='micro')*100)+"%"+'\n')

def dudoanid3(): # nhập thông tin mẫu mới 
    price          = textbox_price.get()
    area           = textbox_area.get()
    rooms          = textbox_rooms.get()
    loca           = textbox_loca.get()
    envi           = textbox_envi.get()
    safety         = textbox_safety.get()
    if((price == '') or (area == '') or (rooms == '') or (loca == '') or (envi == '')or (safety == '')): # thông báo nếu có 1 thuộc tính trống
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([price,area,rooms,loca,envi,safety]).reshape(1, -1) # reshape(1,-1):khai báo lại mảng có 1 dòng
        sample_id3 = pca_best.transform(X_dudoan)                                    # lấy ra mô hình tốt nhất có pca_best thành phần
        y_kqua = id3_max.predict(sample_id3)                                         # id3 dự đoán nhãn với đầu vào là X_dudoan
        lbl2.configure(text= y_kqua)                                                 # lưu kết quả dự đoán 
 
button_id3 = Button(form, text = 'KQ dự đoán theo PCA', command = dudoanid3)         # dùng để thao tác chuột với form để kích hoạt hàm dudoanid3
button_id3.grid(row = 10, column = 1, pady = 20)                                     # vị trí của buttom 
lbl2 = Label(form, text="...")                                                       # khai báo nơi hiện kết quả
lbl2.grid(column=2, row=10)

def khanangid3():                    # hàm tính tỷ lệ đoán đúng của mô hình id3
    dem=0
    for i in range (len(modelmax_id3)):            
        if(modelmax_id3[i] == y_test[i]):
            dem= dem+1
    count = (dem/len(modelmax_id3))*100
    lbl3.configure(text= count)
button_id31 = Button(form, text = 'Khả năng dự đoán đúng ', command = khanangid3)     # dùng để thao tác chuột với form để kích hoạt hàm khanangid3
button_id31.grid(row = 11, column = 1, padx = 30)                                     # vị trí của buttom
lbl3 = Label(form, text="...")                                                        # khai báo nơi hiện kết quả
lbl3.grid(column=2, row=11)                                                           # vị trí hiện 

#=========================================================================================================
#Linear
lbl5 = Label(form, fg="red")              # tạo bảng 
lbl5.grid(column=5, row=9)                # vị trí 
lbl5.configure(text="Tỉ lệ dự đoán đúng của Linear: "+'\n' # tính và in ra các độ đo dựa vào mô hình dự đoán đúng nhất
                           +"r2_score : "+str(r2_score(v_test,v_pred))+"%"+'\n'
                           +"NSE : "   +str(NSE(v_test,v_pred))+"%"+'\n'
                           +"MAE : " +str(mean_absolute_error(v_test,v_pred))+"%"+'\n'
                           +"RMSE : " +str(mean_squared_error(v_test,v_pred,squared=False))+"%"+'\n')

def LinearRegressionPredict(): # nhập thông tin mẫu mới 
    price          = textbox_price.get()
    area           = textbox_area.get()
    rooms          = textbox_rooms.get()
    loca           = textbox_loca.get()
    envi           = textbox_envi.get()
    safety         = textbox_safety.get()
    if((price == '') or (area == '') or (rooms == '') or (loca == '') or (envi == '')or (safety == '')): # thông báo nếu có 1 thuộc tính trống
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([price,area,rooms,loca,envi,safety]).reshape(1, -1) # reshape(1,-1):khai báo lại mảng có 1 dòng
        sample_id3 = pca_best.transform(X_dudoan)                                    # lấy ra mô hình tốt nhất có pca_best thành phần
        y_kqua = id3_max.predict(sample_id3)                                         # id3 dự đoán nhãn với đầu vào là X_dudoan
        lbl4.configure(text= y_kqua)                                                 # lưu kết quả dự đoán 
 
button_id3 = Button(form, text = 'KQ dự đoán theo Linear Regression', command = LinearRegressionPredict)         # dùng để thao tác chuột với form để kích hoạt hàm dudoanid3
button_id3.grid(row = 10, column = 5, pady = 20)                                     # vị trí của buttom 
lbl4 = Label(form, text="...")                                                       # khai báo nơi hiện kết quả
lbl4.grid(column=6, row=10)

def khananglinear():                    # hàm tính tỷ lệ đoán đúng của mô hình id3
    dem=0
    for i in range (len(modelmax_id3)):            
        if(modelmax_id3[i] == y_test[i]):
            dem= dem+1
    count = (dem/len(modelmax_id3))*100
    lbl6.configure(text= count)
button_svc1 = Button(form, text = 'Khả năng dự đoán đúng ', command = khananglinear)     # dùng để thao tác chuột với form để kích hoạt hàm khanangsvm
button_svc1.grid(row = 11, column = 5, padx = 30)                                     # vị trí của buttom
lbl6 = Label(form, text="...")                                                        # khai báo nơi hiện kết quả
lbl6.grid(column=6, row=11)                                                           # vị trí hiện 

form.mainloop()