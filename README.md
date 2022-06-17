# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: M.KIRUTHIGA
RegisterNumber:  212219040061
*/
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
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:

## Data Head:

![image](https://user-images.githubusercontent.com/98682825/174348749-5e071b65-654f-4914-9c46-7f7b8d985d05.png)


## Data Info:

![image](https://user-images.githubusercontent.com/98682825/174348776-f3520a2e-611c-4bed-96ca-f2e08ffc7647.png)


## Data Isnull:

![image](https://user-images.githubusercontent.com/98682825/174348803-3cb41e42-5208-47a5-80ab-df6db418764c.png)


## Data Head:

![image](https://user-images.githubusercontent.com/98682825/174348844-18b5f661-2cd0-437d-a2ab-600d7e2ab201.png)


## MSE:

![image](https://user-images.githubusercontent.com/98682825/174348886-ee24c18d-58c0-430b-9eb3-76f8b44a595b.png)


## R2:

![image](https://user-images.githubusercontent.com/98682825/174348926-b589a1ec-1a83-4b7e-99fc-247e89d504ea.png)


## Predicted Value:

![image](https://user-images.githubusercontent.com/98682825/174348952-2687512e-a2a5-4425-a1b5-8df16b8d59ca.png)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
