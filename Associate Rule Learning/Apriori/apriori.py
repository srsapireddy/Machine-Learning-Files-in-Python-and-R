# Apriori
# To find the strongest rules which optimize the sales
# Optimizing the sales in the grocery store by optimizing the sales of their products
# Takes input as a list of lists but not a dataframe


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

# Here we are using a for loop for all the transactions in the dataset and to loop all the products over each transaction
# The apriori algorithm will expect different transactions and different products as strings
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
# Training Apriori on the dataset
# Here we import the apriori function from apyori.py file
# minimum support, confidence and maximum lift (strength and the relevence of t he rule) depends upon number of transactions
# min_support -> The support of the set of items I = Number of transactions contained in this set of items in I / Total number of transactions
# -> support = Number of transactions containing this product / Total number of transactions 
# For products purchased atleast three or four times a day (This depends on our business problem) (support = 7*3/7500)
# Confidnce of 0.8 means the rules has to be correct 80% of the time that is 80% of the cases that is 4 out of 5. (Here we get some rules that are obvious in nature because of the high confidence and having most purchased items in the basket without any logical association) because there is no rule correct 80% of the time. Which mans all the rules we obtain will be true atleast 20% of time.
# minimum lift -> Here we need to have some rules which have the lift higher than 3. (To get some relevant rules) <-> lift deside the strength of the rules.
# maximum length -> To set the maximum lenght of items (products) in our rules. We are using it as we want to make atleast association rule atleast two different products (minimum/ maximum number of products in the transactions)
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualizing the results
# The rules here are already sorted by there relevance. Relevance -> Its a combination of support, confidence and the list. So this is the relevance cretrion made by the apriori function that consider support, confidence and the lift.
# The relevance cretrion of the apriori function is very close to the lift crtrion. Thus we get the top relevance rules as we get in R.
# Support -> The proportion of the set of this products amoung all the transactions.
# Example -> 0.0045 support value for light cream and chicken -> obtained by number of transactions containing some light cream and chicken divided by total number of transactions. So 0.45% set of the transactions contain light cream and chicken.
# Confidence = 29%(0.29) -> Example -> People buying light cream have 29% of chances of buying chicken.
# So we have left hand side of the rule light cream and right hand side of the rule chicken.
# Lift = 4.8 -> Shows the relevance of the rule. Which makes it a relevant rule. So it is relevant to add light cream with chicken.
results = list(rules)




