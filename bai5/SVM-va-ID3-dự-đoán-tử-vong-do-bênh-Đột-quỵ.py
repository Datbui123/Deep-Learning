import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score #thêm vào để tính chỉ số của mô hình
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from tkinter import * #thêm vào thư viện để tạo form
from tkinter import messagebox
from tkinter import ttk

def tyledung(y_test, y_pred):
    d = 0
    for i in range(len(y_pred)): #đánh giá tỉ lệ mẫu
        if (y_pred[i] == y_test[i]): 
            d = d + 1
    rate = d / len(y_pred)  # tỉ lệ % dự đoán đúng
    return rate
#doc du lieu tu file
data = pd.read_csv('data1.csv')
data.pop('id') #loai bo cot id ra khoi du lieu
le = LabelEncoder() #ham chuyen chuoi thanh so
data['gender'] = le.fit_transform(data['gender']) 
data['ever_married'] = le.fit_transform(data['ever_married']) 
data['work_type'] = le.fit_transform(data['work_type'])
data['Residence_type'] = le.fit_transform(data['Residence_type'])
data['smoking_status'] = le.fit_transform(data['smoking_status'])
X_data = np.array(data[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']].values)
y = np.array(data['stroke'])

max = 0
for j in range(1, 11):
    print("lan", j)
    pca = decomposition.PCA(n_components=j) #với mỗi j thì dùng PCA để chọn ra thành phần chính tốt nhất
    pca.fit(X_data) #sử dụng thành phần chính tốt nhất để huấn luyện mô hình

    Xbar = pca.transform(X_data)  # Áp dụng giảm kích thước cho X. #dùng hàm transform trên X để chuyển X ban đầu thành X 1 chiều tốt nhất
    X_train, X_test, y_train, y_test = train_test_split(Xbar, y, test_size=0.3, shuffle=False)#chia dữ liệu thành tập train-test

    id3 = DecisionTreeClassifier(criterion='entropy')# sử dụng mô hình ID3 để huấn luyện trên tập dữ liệu có 1 thành phần chính tốt nhất
    id3.fit(X_train, y_train) #Truyền X train,y train vào mô hình id3
    svc = SVC(kernel='linear')
    svc.fit(X_train, y_train) #Truyền X train,y train vào mô hình svm
    y_pred_svm= svc.predict(X_test) #sd mô hình svm để dự đoán X test
    rate = tyledung(y_test, y_pred_svm)
    print('Ty le du doan dung: ', rate)

    if (rate > max): # số mẫu gán là đúng ở thời điểm hiện tại > max thì mô hình hiện tại là moo hình tốt
        num_pca = j
        pca_best = pca #lưu lại PCA tốt nhất
        max = rate
        modelmax_svm = svc #lưu lại mô hình
        modelmax_id3 = id3
print("max", max, "d=", num_pca)
#sample_encoder = data.head(1)
#sample_test = np.array(sample_encoder[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']].values)
#print(sample_test)
#sample_pca=pca_best.transform
# (sample_test)
#y_pred = modelmax_svm.predict(sample_pca)
#print('Nhan cua sampel_test:', y_pred)
#tao form
form = Tk()
form.title("Dự đoán tử vong do bệnh đột quỵ:")
form.geometry("1000x1000")



lable_ten = Label(form, text = "Nhập thông tin:", font=("Arial Bold", 10), fg="red")
lable_ten.grid(row = 1, column = 1, padx = 40, pady = 10)

lable_gender = Label(form, text = "Giới tính:")
lable_gender.grid(row = 2, column = 1, padx = 40, pady = 10)
textbox_gender = Entry(form)
textbox_gender.grid(row = 2, column = 2)

lable_age = Label(form, text = "Tuổi:")
lable_age.grid(row = 3, column = 1, pady = 10)
textbox_age = Entry(form)
textbox_age.grid(row = 3, column = 2)

lable_hypertension = Label(form, text = "Tăng huyết áp:")
lable_hypertension.grid(row = 4, column = 1,pady = 10)
textbox_hypertension = Entry(form)
textbox_hypertension.grid(row = 4, column = 2)

lable_heart_disease = Label(form, text = "Bệnh tim:")
lable_heart_disease.grid(row = 5, column = 1, pady = 10)
textbox_heart_disease = Entry(form)
textbox_heart_disease.grid(row = 5, column = 2)

lable_ever_married = Label(form, text = "Tình trang hôn nhân:")
lable_ever_married.grid(row = 6, column = 1, pady = 10 )
textbox_ever_married = Entry(form)
textbox_ever_married.grid(row = 6, column = 2)

lable_work_type = Label(form, text = "Công việc:")
lable_work_type.grid(row = 2, column = 3, pady = 10 )
textbox_work_type = Entry(form)
textbox_work_type.grid(row = 2, column = 4)

lable_Residence_type = Label(form, text = "Khu vực sống:")
lable_Residence_type.grid(row = 3, column = 3, pady = 10 )
textbox_Residence_type = Entry(form)
textbox_Residence_type.grid(row = 3, column = 4)

lable_avg_glucose_level = Label(form, text = "Mức độ đường trung bình:")
lable_avg_glucose_level.grid(row = 4, column = 3, pady = 10 )
textbox_avg_glucose_level = Entry(form)
textbox_avg_glucose_level.grid(row = 4, column = 4)

lable_bmi = Label(form, text = "BMI:")
lable_bmi.grid(row = 5, column = 3, pady = 10 )
textbox_bmi = Entry(form)
textbox_bmi.grid(row = 5, column = 4)

lable_smoking_status = Label(form, text = "Tình trạng hút thuốc:")
lable_smoking_status.grid(row = 6, column = 3, pady = 10 )
textbox_smoking_status = Entry(form)
textbox_smoking_status.grid(row = 6, column = 4)


#du doan 
#y_svm = cart.predict(X_test)
lbl1 = Label(form)
lbl1.grid(column=1, row=8)
lbl1.configure(text="Tỉ lệ dự đoán đúng của SVM: "+'\n'
                           +"Precision: "+str(precision_score(y_test, y_pred_svm, average='macro')*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_test, y_pred_svm, average='macro')*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_test, y_pred_svm, average='macro')*100)+"%"+'\n')
def dudoansvm():
   # gender = textbox_gender.get()
    age = textbox_age.get()
    hypertension = textbox_hypertension.get()
    heart_disease = textbox_heart_disease.get()
    #ever_married =textbox_ever_married.get()
    #work_type =textbox_work_type.get()
    #Residence_type =textbox_Residence_type.get()
    avg_glucose_level =textbox_avg_glucose_level.get()
    bmi =textbox_bmi.get()
    #smoking_status =textbox_smoking_status.get()
    le = LabelEncoder() #ham chuyen chuoi thanh so
    gender = le.fit_transform([textbox_gender.get()]) 
    ever_married = le.fit_transform([textbox_ever_married.get()]) 
    work_type = le.fit_transform([textbox_work_type.get()])
    Residence_type = le.fit_transform([textbox_Residence_type.get()])
    smoking_status = le.fit_transform([textbox_smoking_status.get()])
    if((gender == '') or (age == '') or (hypertension == '')  or (heart_disease == '')or (ever_married == '')or (work_type == '')or (Residence_type == '')or (avg_glucose_level == '')or (bmi == '')or (smoking_status == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]).reshape(1, -1)
        sample_svm=pca_best.transform(X_dudoan)
        y_kqua = modelmax_svm.predict(sample_svm)
        lbl.configure(text= y_kqua)
        print(X_dudoan)
button_cart = Button(form, text = 'Kết quả dự đoán theo SVM', command = dudoansvm)
button_cart.grid(row = 9, column = 1, pady = 20)
lbl = Label(form, text="...")
lbl.grid(column=2, row=9)

def khanangsvm():
    #y_cart = cart.predict(X_test)
    dem=0
    for i in range (len(y_pred_svm)):
        if(y_pred_svm[i] == y_test[i]):
            dem= dem+1
    count = (dem/len(y_test))*100
    lbl1.configure(text= count)
button_cart1 = Button(form, text = 'Khả năng dự đoán đúng ', command = khanangsvm)
button_cart1.grid(row = 10, column = 1, padx = 30)
lbl1 = Label(form, text="...")
lbl1.grid(column=2, row=10)
#Cay quyet dinh
#dudoanid3test
y_id3 = id3.predict(X_test)
lbl3 = Label(form)
lbl3.grid(column=3, row=8)
lbl3.configure(text="Tỉ lệ dự đoán đúng của ID3: "+'\n'
                           +"Precision: "+str(precision_score(y_test, y_id3, average='macro')*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_test, y_id3, average='macro')*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_test, y_id3, average='macro')*100)+"%"+'\n')
def dudoanid3():
    # gender = textbox_gender.get()
    age = textbox_age.get()
    hypertension = textbox_hypertension.get()
    heart_disease = textbox_heart_disease.get()
    #ever_married =textbox_ever_married.get()
    #work_type =textbox_work_type.get()
    #Residence_type =textbox_Residence_type.get()
    avg_glucose_level =textbox_avg_glucose_level.get()
    bmi =textbox_bmi.get()
    #smoking_status =textbox_smoking_status.get()
    le = LabelEncoder() #ham chuyen chuoi thanh so
    gender = le.fit_transform([textbox_gender.get()]) 
    ever_married = le.fit_transform([textbox_ever_married.get()]) 
    work_type = le.fit_transform([textbox_work_type.get()])
    Residence_type = le.fit_transform([textbox_Residence_type.get()])
    smoking_status = le.fit_transform([textbox_smoking_status.get()])
    if((gender == '') or (age == '') or (hypertension == '')  or (heart_disease == '')or (ever_married == '')or (work_type == '')or (Residence_type == '')or (avg_glucose_level == '')or (bmi == '')or (smoking_status == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]).reshape(1, -1)
        sample_id3=pca_best.transform(X_dudoan)
        y_kqua = modelmax_id3.predict(sample_id3)
        lbl2.configure(text= y_kqua)
        
button_id3 = Button(form, text = 'Kết quả dự đoán theo ID3', command = dudoanid3)
button_id3.grid(row = 9, column = 3, pady = 20)
lbl2 = Label(form, text="...")
lbl2.grid(column=4, row=9)

def khanangid3():
    y_id3 = id3.predict(X_test)
    dem=0
    for i in range (len(y_id3)):
        if(y_id3[i] == y_test[i]):
            dem= dem+1
    count = (dem/len(y_test))*100
    lbl3.configure(text= count)
button_id31 = Button(form, text = 'Khả năng dự đoán đúng ', command = khanangid3)
button_id31.grid(row = 10, column = 3, padx = 30)
lbl3 = Label(form, text="...")
lbl3.grid(column=4, row=10)
form.mainloop()
