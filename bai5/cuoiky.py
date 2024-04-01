from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



df = pd.read_csv('datacar2.csv')
X = np.array(df[['price','area','rooms','loca','envi','safety']].values)    
y = np.array(df['acceptability'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , shuffle = False)
#cart
cart = DecisionTreeClassifier(criterion='gini',max_depth=4, random_state=42)
cart.fit(X_train, y_train)
#id3
id3 = DecisionTreeClassifier(criterion='entropy',max_depth=4, random_state=42)
id3.fit(X_train, y_train)

#form
form = Tk()
form.title("Dự đoán chất lượng nhà ở:")
form.geometry("1000x500")



lable_ten = Label(form, text = "Nhập thông tin cho nhà ở:", font=("Arial Bold", 10), fg="red")
lable_ten.grid(row = 1, column = 1, padx = 40, pady = 10)

lable_buying = Label(form, text = " Giá:")
lable_buying.grid(row = 2, column = 1, padx = 40, pady = 10)
textbox_buying = Entry(form)
textbox_buying.grid(row = 2, column = 2)

lable_maint = Label(form, text = "Diện tích:")
lable_maint.grid(row = 3, column = 1, pady = 10)
textbox_maint = Entry(form)
textbox_maint.grid(row = 3, column = 2)

lable_doors = Label(form, text = "Số phòng:")
lable_doors.grid(row = 4, column = 1,pady = 10)
textbox_doors = Entry(form)
textbox_doors.grid(row = 4, column = 2)

lable_persons = Label(form, text = "Vị trí:")
lable_persons.grid(row = 5, column = 1, pady = 10)
textbox_persons = Entry(form)
textbox_persons.grid(row = 5, column = 2)

lable_lug_boot = Label(form, text = "Môi trường:")
lable_lug_boot.grid(row = 6, column = 1, pady = 10 )
textbox_lug_boot = Entry(form)
textbox_lug_boot.grid(row = 6, column = 2)

lable_safety = Label(form, text = "Độ an toàn:")
lable_safety.grid(row = 7, column = 1, pady = 10 )
textbox_safety = Entry(form)
textbox_safety.grid(row = 7, column = 2)



#cart
#cart
#dudoancarttheotest
y_cart = cart.predict(X_test)
lbl1 = Label(form)
lbl1.grid(column=1, row=8)
lbl1.configure(text="Tỉ lệ dự đoán đúng của CART: "+'\n'
                           +"Precision: "+str(precision_score(y_test, y_cart, average='macro')*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_test, y_cart, average='macro')*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_test, y_cart, average='macro')*100)+"%"+'\n')
def dudoancart():
    buying = textbox_buying.get()
    maint = textbox_maint.get()
    doors = textbox_doors.get()
    persons = textbox_persons.get()
    lug_boot =textbox_lug_boot.get()
    safety =textbox_safety.get()
    if((buying == '') or (maint == '') or (doors == '') or (persons == '') or (lug_boot == '')or (safety == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([buying,maint,doors,persons,lug_boot,safety]).reshape(1, -1)
        y_kqua = cart.predict(X_dudoan)
        lbl.configure(text= y_kqua)
button_cart = Button(form, text = 'Kết quả dự đoán theo CART', command = dudoancart)
button_cart.grid(row = 9, column = 1, pady = 20)
lbl = Label(form, text="...")
lbl.grid(column=2, row=9)

def khanangcart():
    y_cart = cart.predict(X_test)
    dem=0
    for i in range (len(y_cart)):
        if(y_cart[i] == y_test[i]):
            dem= dem+1
    count = (dem/len(y_cart))*100
    lbl1.configure(text= count)
button_cart1 = Button(form, text = 'Khả năng dự đoán đúng ', command = khanangcart)
button_cart1.grid(row = 10, column = 1, padx = 30)
lbl1 = Label(form, text="...")
lbl1.grid(column=2, row=10)

#id3
#dudoanid3test
y_id3 = id3.predict(X_test)
lbl3 = Label(form)
lbl3.grid(column=3, row=8)
lbl3.configure(text="Tỉ lệ dự đoán đúng của ID3: "+'\n'
                           +"Precision: "+str(precision_score(y_test, y_id3, average='macro')*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_test, y_id3, average='macro')*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_test, y_id3, average='macro')*100)+"%"+'\n')
def dudoanid3():
    buying = textbox_buying.get()
    maint = textbox_maint.get()
    doors = textbox_doors.get()
    persons = textbox_persons.get()
    lug_boot =textbox_lug_boot.get()
    safety =textbox_safety.get()
    if((buying == '') or (maint == '') or (doors == '') or (persons == '') or (lug_boot == '')or (safety == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([buying,maint,doors,persons,lug_boot,safety]).reshape(1, -1)
        y_kqua = id3.predict(X_dudoan)
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
    count = (dem/len(y_id3))*100
    lbl3.configure(text= count)
button_id31 = Button(form, text = 'Khả năng dự đoán đúng ', command = khanangid3)
button_id31.grid(row = 10, column = 3, padx = 30)
lbl3 = Label(form, text="...")
lbl3.grid(column=4, row=10)


form.mainloop()
