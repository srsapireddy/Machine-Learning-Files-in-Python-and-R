# Eclat
# Support -> Here we get the diferent products which are frequently bought together
# Input is the sparse matrix - same as the eclat model
# Density -> The proportion of the non-zero values in the matrix

# Data Preprocessing
dataset = read.csv("Market_Basket_Optimisation.csv", header = FALSE)

library(arules)
dataset = read.transactions("Market_Basket_Optimisation.csv", sep = ',', rm.duplicates = TRUE)
summary(dataset)

# Frequency plot of different products bought by different customers in store during this whole week
itemFrequencyPlot(dataset, topN = 10)

# Training Eclat Algorithm on the dataset
rules = eclat(data = dataset, parameter = list(support = 0.004, minlen = 2))

# Sorting the rules by decreasign the support
inspect(sort(rules, by = 'support')[1:10])