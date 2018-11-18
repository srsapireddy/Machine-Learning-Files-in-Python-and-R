# Model Selection --> k - fold cross validation
# Here we evluate the model performance and improve the model performance. This inclues choosing the best parameters of the machine learning model.
# Everytime we build a machine learning model we have two types of parameters.
# [1] The parameer the model learnt. That is the parametes taht were changed and found the optimal values by running the model.
# [2] The parameters that we choose ourselves. These parameters are called the hyper parameters.Thus there is way of choosing the optimal values for these hyper parameters. This can be done through the grid search.
# Upto now we have the training set and test set. We train the data on the training set and we test the performance of the model on the test set.
# But doing this we have the variance problem. The variance problem can be explained by when we get the accuracy on the test set and test the model on a different test set we get a different accuracy.
# So judging our model on only one accuracy and one test set is not super relevant.
# This can be solved by k-fold cross validation. This can is done by splitting the training set into 10 folds when k = 10.
# Then we train the model on 9 folds and test the model on last fold.
# Since with 10 folds we can make 10 different combinations of 9 folds and 1 fold to test it. That is we can train and test the model on 10 different combinations.
# Hence we can take the average of the accuracies of the different 10 observations and also compute the standard deviation to look at the variance.
# The bias - variance tradeoff categories
# [1] Good accuracy and small variance --> Low bias low variance.
# [2] Large accuracy and high variance --> Low bias high variance.
# [3] Small accuracy and low variance --> High bias low variance.
# [4] Low accuaracy and high variance --> High bias high variance.

# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# Fitting classifier to the Training set
install.packages('e1071')
library(e1071)
classifier = svm(formula = Purchased ~ ., data = training_set, type = 'C-classification', kernel = 'radial')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)

# Applying the K - Fold Cross Validation
install.packages('caret')
library(caret)
folds = createFolds(training_set$Purchased, k = 10)
# lapply function --> To apply the function to the different elements of the list.
# Here function --> Function used to find the accuracy of each of these 10 folds.
# list --> folds.
cv = lapply(folds, function(x){
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = svm(formula = Purchased ~ ., data = training_fold, type = 'C-classification', kernel = 'radial')
  y_pred = predict(classifier, newdata = test_fold[-3])
  cm = table(test_fold[, 3], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[2,1] + cm[1,2])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))
# x --> Each one of the 10 folds.
# Inthis function we are going to implement k - fold cross validation. This contains:
# [1] training_set --> This is the whole training set from which we withdraw the test fold.

# Visualising the Training set results
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Kernel SVM (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'Kernel SVM (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


