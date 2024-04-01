import numpy as np 
import pandas as pd 

from sklearn import tree 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_recall_fscore_support , f1_score
from sklearn.preprocessing import LabelEncoder

# Xử lí data

df = pd.read_csv('./Data.csv')

le = LabelEncoder()

print(df)
print("=======================")

# outlook = le.fit_transform(df['outlook'].values)

Year = le.fit_transform(df['Year'].values)
Type = le.fit_transform(df['Type'].values)
Color = le.fit_transform(df['Color'].values)
Price = le.fit_transform(df['Price'].values)
Number = le.fit_transform(df['Number'].values)
Users = le.fit_transform(df['Users'].values)

data = np.array([Year, Type, Color, Price, Number, Users])

dt_Train , dt_Test = train_test_split(data.T, test_size=0.3 , shuffle=False)

X_train = dt_Train[:,:5]
X_test = dt_Test[:,:5]

y_train = dt_Train[:,5]
y_test = dt_Test[:,5]


# Khai báo các hàm tính lỗi

def percent(y_pred , y_test):
	count = 0
	for i in range(0, len(y_test)):
		if( y_pred[i] == y_test[i]):
			count = count + 1
	return round(count/len(y_test), 4)





# ID3

id3_cls = tree.DecisionTreeClassifier(criterion="entropy")

id3_cls.fit(X_train, y_train)

y_id3_predict = id3_cls.predict(X_test)

# Tính lỗi ID3
id3_percent = percent(y_id3_predict, y_test)
id3_score  = precision_recall_fscore_support(y_test, y_id3_predict, average="macro")
id3_f1 = f1_score(y_test, y_id3_predict , average="macro")
# precision - recall - fscore - support

print("ID3 COUNT" , id3_percent )
print("ID3 PRECISION" , round(id3_score[0],4))
print("ID3 RECALL" , round(id3_score[0],4))
print("ID3 F1" , round(id3_f1,4))

print("=======================")
# CART

cart_cls = tree.DecisionTreeClassifier(criterion='gini')

cart_cls.fit(X_train , y_train)
y_cart_predict = cart_cls.predict(X_test)


# Tính lỗi cart
cart_percent = percent(y_cart_predict, y_test)
cart_score = precision_recall_fscore_support(y_test, y_cart_predict, average="macro")
cart_f1 = f1_score(y_test, y_cart_predict, average="macro")
print("COUNT CART" , cart_percent)
print("CART PRECISION" , round(cart_score[0],4))
print("CART RECALL" , round(cart_score[0],4))
print("CART F1" , round(cart_f1,4))


print("=======================")

# Per


per_cls = Perceptron()

per_cls.fit(X_train , y_train)

y_per_predict = per_cls.predict(X_test)

per_score = precision_recall_fscore_support(y_test, y_per_predict, average="macro")
per_f1 = f1_score(y_test, y_per_predict, average="macro")
print("COUNT PER" , percent(y_per_predict,y_test))
print("CART PRECISION" , round(per_score[0],4))
print("CART RECALL" , round(per_score[0],4))
print("CART F1" , round(per_f1,4))

