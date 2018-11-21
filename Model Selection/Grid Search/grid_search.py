# Grid Search --> To improve the model performance.
# This is done by finding the optimal values of the hyper parameters.
# In every machine learning algorithm we have two types of parameters.
# [1] The parameters that are learnt by the machine learning model.
# [2] The parameters that we choose.

# How to choose the machine learning model

# Step 1
# If we dont have the dependent variable --> Clustering problem.
# If we have the dependent variable or a categorical problem --> Classification problem.
# If it is continuous profblem --> Regression problem.

# Step 2
# Find if it is an linear problem or non linear problem.
# Linear like SVM, non-linear like Kernal-SVM.
# This can be done by grid search. That is finding if the problem is linear or non-linear.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Classifier to Training Set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test Set Results
# y_pred --> Vector of predictions
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
# Confusion Matrix -> To see weather our Logistic Regression made correct prediction or not.
# This confusion matrx will contain the correct predictions made on the Test Set as well as the incorrect predictions,
# For this we are importing a function and not a class.
# Distinction --> Class contains the captial letters at the beginning.
# Parameters of cnfusion matrix -> (1) y_true = Real values thats the values of the data set, (2) y_pred.
# 65, 24 = 89-> Correct Predictions, 8,3 = 11-> Incorrect Predictions
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k - fold cross validation.
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
# Standard Deviation --> That is the average of the different accuracies that we will get when evluating the model performance and the average accuracy that is 90 percentage is 6 percentage. That means we are in low bias and low variance category. That is not too high variance.

# Applying the grid search for finding the better model and better performance.
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001, 0.0001]}]
# The different options that will be investigated by grid search to find the best set of parameters.
# For linear option -->
# C --> Penality Parameter --> The more the penality parameter the more the more it will prevent overfitting.
# gamma function --> coefficient of the kernel function.
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
# CV --> K in the k - fold cross validation.

# Parameters
# [1] estimator --> classifier.
# [2] X --> training set.
# [3] y --> Dependent variable vector of the training set.
# [4] cv --> Number of folds we want to split the training set into.
# [5] n_jobs --> -1 (Set this to -1 if we have the larger dataset. This will enable all the CPUs in the computer.)

# Visualizing the Training Set Results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()