# K-fold cross validation --> Optimizing way to evluate the models.
# Here we evaluate the model performance and improve the model performance.
# That is choosing the best parameters of the machine learning model.
# There are two types of parameters
# [1] --> Parameters that the model learns and the parameters that were changed and found the optimal values by running the model.
# [2] --> The parameters that we choose ourselves.
# These parameters are called the hybrid parameters.
# In previous models we split the data in to training set and the test set. Then we train the data on the training set and test the data in the test set. But here we get the variance problem. This can be explained by when we get the accuracy on the test set. So if we run the model and test the model on the other test set we would get another accuracy.
# So judjing our model only on one accuracy on one test set is not super relevant. This is not the mot relevant way to test the model performance.
# The k-fold cross validation will fix this problem. This is fixed by spinning the training set in to ten folds when k = 10.
# Then we train our model on 9 folds and test is on the remaining fold.
# Thus with the 10 folds we get 9 combinations of 9 folds to train the model and 1 fold to test the model.
# Thus we can train and test the model on 10 sets of combinations.
# Then we take the accuracy of different 10 evaluations and also compute the standard deviation to see the variance.
# Bias-Variance Trade off
# [1] Good accuracy and small variance --> Low Bias Low Variance
# [2] Large accuaracy and high variance --> Low Bias High Variance
# [3] Small accuracy and low variance --> High Bias Low Variance
# [4] Low accuaracy and high variance --> High Bias High Variance

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