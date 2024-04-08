# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
   
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: THIRISHA.S

RegisterNumber:  212222230160


```python

import pandas as pd
data=pd.read_csv('/content/Employee.csv')

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

1.Data Head

<img width="566" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121222763/b859bb3e-cd91-4fa3-99bb-27106b68b378">


2.Dataset Info

<img width="235" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121222763/7bb7896d-ba17-4667-ac8d-6e9d34314d16">

3.Null dataset

<img width="125" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121222763/055e3ec4-6fa8-410c-9994-666b9118a12d">

4.Values Count in Left Column

<img width="122" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121222763/506a68dd-4bb6-43f7-a447-a1dc3967baab">

5.Dataset transformed head

<img width="566" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121222763/0bc6955d-011b-4e7f-81dd-f34453ae0d46">

6.x.head()

<img width="521" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121222763/06cc0174-da96-44e5-a968-5e471a7fe48e">

7.Accuracy

<img width="90" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121222763/f1a8ac5c-983e-4bd6-891d-8e7e0d58c5f8">

8.Data Prediction

<img width="559" alt="image" src="https://github.com/TejaswiniGugananthan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121222763/390fd95f-0089-42ef-9519-559e1842e532">



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
