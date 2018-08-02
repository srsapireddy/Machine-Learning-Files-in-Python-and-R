# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# We always want to consider matrix of features as a matrix and not as a vector
# Here python will not consider the upper bound index 2 in X
# (10,1) -> Matrix of feature of 10 lines and 1 column
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Here we need the shole data as training set to predict the accurate predictions
# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# We dont need Feature Scaling because the polynomial regression means adding some polynimials into multiple linear regression equation
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting linear regression to the dataset
# Here 2nd polynomial regressor is based on linear regressor on which we add some polynomial terms
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Fitting polynomial regression to the dataset
# Here we include a new class to include polynomial terms into linear regression equation -> polynomial features class
from sklearn.preprocessing import PolynomialFeatures
# Here poly_reg object is an transformation tool that will transform matrix of features X into a new matrix of features Xpoly (creates new independent varibles according to the formula X^2, X^3 ..... X^n)
# We call the PolynomialFeatures class with the parameter degree.
# degree -> Specifies the degree of polynomial features in the future feature scale matrixs Xpoly
poly_reg = PolynomialFeatures(degree = 3)
# As degree = 2 we are adding only 1 additional polynomial term
# Creating Xpoly matrix  
X_poly = poly_reg.fit_transform(X)
# The pupose of 2nd linear regressor lin_reg2 is to include this fit made with poly_reg object into our regression model
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y) 

# Visualizing the Linear Regression Results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Level of Position ()')
plt.ylabel('Salaries')
plt.show()

# Visualizing the Polynomial Regression Results
# To get a curve with a higher resolution
# X_grid contains all the levels plus incremental steps between the levels with a resolution of 0.1
# mix(X) -> lower bound
# max(X) -> upper bound
# Incrementation = 0.1
X_grid = np.arange(min(X), max(X), 0.1) # This gives a vector
# To reshap vector into a matrix
# We reshape X_grid into a matrix where the number of lines is the number of elements of X_grid [len(X_grid] and number of columns is 1
X_grid = X_grid.reshape((len(X_grid)), 1)
plt.scatter(X, y, color = 'red')
# Our model predicts imaginary salaries of 90 levels from 1 to 10 with a resolution of 0.1
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Level of Position ()')
plt.ylabel('Salaries')
plt.show()

# Adding a degree to Polynomial Regression Model
# Make degree = 3
# Here we can see that it is not convex anymore
# Our model is much better now

# Adding another degree
# Making degree = 4 we get a perfect model

# Predicting a new result with Linear Regression
lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))





