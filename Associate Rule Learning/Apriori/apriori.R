# Apriori
# Apriori Algorithm is trained on the transactions dataset
# We want to optimize the sales but we also want to optimize the revenue.

# Data Preprocessing
dataset = read.csv("Market_Basket_Optimisation.csv", header = FALSE)

# This algorithm takes the input as the Sparse Matrix
# Sparse Matrix -> It is a matrix which contains lot of zeros (Very few number of non-zero values)
# We gonna take all the different products of the dataset (120 products) and we are going to attribute one column to each of 120 products.
# That means we get 120 columns
# Columns -> 1 (If the product is in the basket of the customer), 0 (If the product is not in the basket)
# To train the Apriori Algorithm we dont need to have duplicates
#install.packages('arules')
library(arules)
dataset = read.transactions("Market_Basket_Optimisation.csv", sep = ',', rm.duplicates = TRUE)
summary(dataset)
# Comment ->
# 1
# 5
# There are 5 transactions containing 1 duplicate
# Density -> The proportion of Non-zero values
# Information of distribution of baskets of all 7500 transcations
# This 1 associated to 1754 means there are 1754 baskets containing only one product#
# On an average people put 4 products in there basket when they go to shop. (Quantiles)

# Frequency plot of different products bought by different customers in store during this whole week
itemFrequencyPlot(dataset, topN = 10)

# Training Apriori Algorithm on the dataset
# Here we need to choose a value for the Apriori Algorithm itself that is support and we use frequency plot to choose a good support
# Apriori Function -> 2nd argument => parameter argument (That will contain the choice of minimum support and a choice of confidence support)
# The support of set of items I = numbr of transactions contained in this set of items I divided by total number of trnasactions I.
# Support -> Numbr of transactions contained in this product over total number of transactions. That is the minimum support we need to have in our rules
# Since we have 7500 transactions the minimum support is equal to 7*3/7500 (Product is purchased 3 times a day, 7 times a week(Total number of transactions registered over a week) and 7500 -> total number of transactions)
# Support of a product purchased 3 times a day over one week -> 3*7/7500 = 0.003 (Minimum support of products that will be considered by our rules)
# Confidence -> That is the minimum confidence we need to have in our rules. This means that are going to appear in your rules will have a higher support than this support here and same for the confidence.
# Confidence is a kind of a arbitary choice (Start with a default value => 0.8 and tune accordingly). Until we get the relevant rules
# Since revenue is the linear combination of different number of products where the coefficients of these products ar the prices of these products.
# Here if we need to optimize the revenue of these products we need to optimize the sales of these products that are purchased very often rather than these products which are less purchased.
# Thus we need to choose the support left to the minimum support.
# To choose the support we need to choose the products which are purchased frequently (more than three of four times a day)
# minlen (minlen of the basket) -> According to the basket of rules the basket will contain atleast one product.
# Number of rules -> 0 rules here (When we train our apriori model we found our zero rules). It is due to the confidence level we choose to have. That means each rule should be correct atleast 80 percentage of the transactions. That means the rule must be true aleast out of 5 times.
# For the products bought atleast four times a day. Support = 0.004
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Sorting the rules by decreasign the lift (Best metric to measure the relevance of the rule)
# (confidence = 0.4 -> Most purchased products) -> These are the rules associated to the most purchased products that fall in the same basket. Thats why we need to change the confidence to get (most people who bought also bought)
# (confidence = 0.2 -> Here we will get the most rules relevant to the Associated Rule related to the principle people who bought also bought)
# Visualising the results
# 10 fist rules having the high lift
# Confidence -> How much percntage of cases pople buy the products according to the association rule.
inspect(sort(rules, by = 'lift')[1:10])



