# Regression Template

# Polynomial Regression

# Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Regression to the dataset
# Create your regressor

# Predicting the new result
# Here y_pred is not going to be the vector of predictions but the predicted salary of 6.5 level
y_pred = regressor.predict(6.5)

# Visualizing the Regression Results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Level of Position ()')
plt.ylabel('Salaries')
plt.show()

# Visualizing the Regression Results (For higher resolution and smoother curve)
# X_grid -> All the levels from 1 to 10 incrmnted by 0.1
# arange -> to create an array of levels (with lower bound, upper bound and the step)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Level of Position ()')
plt.ylabel('Salaries')
plt.show()