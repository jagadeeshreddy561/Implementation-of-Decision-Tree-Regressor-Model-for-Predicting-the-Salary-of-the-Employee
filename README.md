# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## STEP 1 :
Import the standard libraries. 2.Upload the dataset and check for any null values using .isnull() function.

## STEP 2 :
Import LabelEncoder and encode the dataset.

## STEP 3 :
Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

## STEP 4 :
Predict the values of arrays.

## STEP 5 :
Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset 7.Predict the values of array.

## STEP 6 :
Apply to new unknown values.

## Program:
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
## Developed by: Jagadeeshreddy 
## RegisterNumber: 212222240059  
```python
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
## data.head() :



![ml-7 1](https://github.com/jagadeeshreddy561/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120623104/43b3cc1f-55bf-405c-921a-3c253c590c2d)


## data.info() :



![ml-7 2](https://github.com/jagadeeshreddy561/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120623104/57a465d8-a8a1-4482-af42-33bed30a4a73)



## isnull() & sum() function :


![ml-7 3](https://github.com/jagadeeshreddy561/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120623104/86d97236-6834-4f0e-814c-0bf8e57a71f3)



## data.head() for Position :


![ml-7 4](https://github.com/jagadeeshreddy561/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120623104/e780a3ed-e970-441b-89af-ad12d5d3cc2f)



## MSE value : 


![ml-7 5](https://github.com/jagadeeshreddy561/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120623104/fcdbf50f-70e1-482b-81dd-6fd6c5142629)



## R2 value :


![ml-7 6](https://github.com/jagadeeshreddy561/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120623104/55779004-76c9-420a-b0d5-afdcb982665d)


## Prediction value :


![ml-7 7](https://github.com/jagadeeshreddy561/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120623104/5e674629-533f-4d67-9c7c-7372a6cab8db)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
