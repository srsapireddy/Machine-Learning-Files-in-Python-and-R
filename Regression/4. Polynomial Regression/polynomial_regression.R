# Polynomial Regression

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
# Here we wont need feature scaling as a polynomial regression model is actually a multiple rgression model with polynomial terms 
# Instead of having different features like features that represent something very different
# Here we are taking the first feature that is the position levels form 1 to 10 and as in the case of multiple line regression we take the squares of the levels here
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Linear Regression to the dataset
lin_reg = lm(formula = Salary ~ ., data = dataset)

# Fitting Polynomial Regression to the dataset
# Here the independent variables are actually the polynomial features and here we have only one independent variable
# Adding polynomial features  -> These are some aditional independent variables that are Level^2, Level^3 and so on
# These compose our new matrix of features in some way which will be the matrix on which we apply our multiple linear regression
# This will make the whole model into Polynomial Regression Model
# In short a polynomial regression model is a multiple linear regression that composed of one independent variable and additional independent variables that are the polynomial terms of the first independent variables
# Adding a new independent variable Level 2 (Level^2). To add a column in a dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
# Level -> orginal independent Variables
# Level2 -> Polynomial Terms of first independent variables (Level)
poly_reg = lm(formula = Salary ~ ., data = dataset)

# Visualizing the Linear Regression Results
#install.packages('ggplot2')
library(ggplot2)
ggplot() + geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)), colour = 'blue') +
              ggtitle('Truth or Bluff (Linear Regression)') +
              xlab('Level') +
              ylab('Salaries')

# Visualizing the Polynomial Regression Results
ggplot() + geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)), colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salaries')

# Predicting a new Result with Linear Regression Model
y_pred = predict(lin_reg, data.frame(Level = 6.5))

# Predicting a new Result with Polynomial Regression Model
y_pred = predict(poly_reg, data.frame(Level = 6.5,
                                      Level2 = 6.5^2,
                                      Level3 = 6.5^3,
                                      Level4 = 6.5^4))
