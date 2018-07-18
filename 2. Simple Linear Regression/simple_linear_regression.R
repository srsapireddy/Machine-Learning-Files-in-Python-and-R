# Simple Linear Regression

# Importing the Dataset
dataset = read.csv('Salary_Data.csv')
# dataset = dataset[, 2:3]

# Splitting the data into Training Set and Test Set
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling -> R gonna take care of it
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

# Fitting Simple Linear Regression to our Training Set
# Here we are using lm function
# formula -> dependeent variaBle expressed as a linear combination of independent variable
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)
# summary(regressor)
# It tells about statistical significance of the cofficients
# The 3 stars at the end tells that the independent variable and dependent variable are highly significant (strong linear relationship)
# Lower the P value is the high the significance will be
# P value - When we are below 5 % the independent variable will be highly significant
# Multiple R - squared and Adjusted R - squared tells about the evulating the model

# Predicting the Test Set results
y_pred = predict(regressor, newdata = test_set)

# Visualizing Training Set Results
# install.packages('ggplot2')
library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), colour = 'blue') +
  ggtitle('Years of Experience vs Salary (Training Set)') +
  xlab('Years of Experience') +
  ylab('Salary')

# Visualizing Test Set Results
# install.packages('ggplot2')
library(ggplot2)
ggplot() + 
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)), colour = 'blue') +
  ggtitle('Years of Experience vs Salary (Training Set)') +
  xlab('Years of Experience') +
  ylab('Salary')
# As our regressor is already trained on the Training Set there is no need to change the geom_line. We obtain the same Simple 
# Linear Regression Line whether we keep training_set or the test_set here
