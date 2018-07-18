# SVR

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting the SVR to the dataset
install.packages('e1071')
library(e1071)
regressor = svm(formula = Salary ~ ., data = dataset, type = 'eps-regression')

# Predicting a new Result
# Adding polynomial regression terms -> Level1, Level2, Level3 and Level4
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualizing the SVR Results
install.packages('ggplot2')
library(ggplot2)
ggplot() + geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)), colour = 'blue') +
  ggtitle('Truth or Bluff (SVR)') +
  xlab('Level') +
  ylab('Salaries')