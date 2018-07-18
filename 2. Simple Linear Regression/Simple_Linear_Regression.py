
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
#Creating matrix of features of independent variable x (30,1) and the vector y of dependent variable (30,)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#*** Here Python will take care of Feature Scaling here The libraries are gonna take of that ***
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting simple linear regression model to training set
# Here the object will be simple linear regressor because it is the regressor we are going to fit to the training set
# To fit to the training set we will use a method because linear regression class has several methods and one of the method is the fit method is just like a tool of function
# Importing linear regression class
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# Fitting -> Our code will already have learned the correlation of the training set to learn how to predict the y_train
regressor.fit(X_train, y_train)

# Predicting the Test Set Results
# Here we create a vector of predctive values -> We will create a vector of predctive values of the Test Set salaries and we will put all these predicted salaries into a single vector y_pred
# y_pred -> Vector of predicted dependent variables
y_pred = regressor.predict(X_test)

# Visualizing the Training Set
plt.scatter(X_train, y_train, color = 'red')
# Here we are gonna compare the real salaries and the predicted salaries based on the same observations. That is the observations of the training set
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test Set
plt.scatter(X_test, y_test, color = 'red')
# Here we are gonna compare the real salaries and the predicted salaries based on the same observations. That is the observations of the test set
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()