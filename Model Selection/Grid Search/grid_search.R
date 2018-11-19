# Grid Search --> This gonna improve the models performance. By finding the optimal values of the hyper parameters.
# Hyper parameters --> The parameters that we choose.
# This will answer which of the machine learning model we are going to choose for our model. The most optimal value for our machine learning model.
# Here we gonna take the example of kernarl parameters ---> penality parameer, gamma parameter.

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

# Applying grid search for finding the optimal parameters.
# Caret package is used for parameter 
# Radial Basis Function Kernel is a gaussian kernal and a common kernal for building the kernal svm nodel.
# Train parameters -->
# [1] svmRadial (Method parameter) --> By this parameter the train function will know which model to build and which model to choose.
# [2] form --> formula
library(caret)
classifier = train(form = Purchased ~., data = training_set, method = 'svmRadial')
classifier
classifier$bestTune

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



