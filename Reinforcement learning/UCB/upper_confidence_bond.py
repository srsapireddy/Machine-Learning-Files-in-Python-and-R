# Upper Confidence Bond
# This algorithm will decide which version of the ad should be showed to the user. (To find the ad with more clicks.)
# Here we are gonna start with no data.
# Round -> Each time the user connects to the account it will be a round.
# Reward = 1 -> If user clicks on the ad.
# Reward = 0 -> If the user does not click on the ad.
# It will decide which ad to show to the user according to the previous observations (depends upon previous rounds).
# The goal of the algorithm is to maximize the total rewards.
# Total Reward = Sum of all different rewards at each round obtained by the different selections of the ads.
# Reinfornment Learning -> Online Learning -> Interactive Learning
# Here the strategy is dynamic and it depends on from the observations from the beginning of the experiment to the present time.
# CTR -> Click To Rate (Conversion Rates)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# This is just a dataset for simulation
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Implementing UCB
import math
N = 10000
d = 10
ads_selected = []
# Vector of size d containing only zeros
# Ni(n) -> the number of times the ad i was selected upto round n
# Ri(n) -> the sum of rewards of the ad i upto round n.
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
# ri(vector)(n) -> the average reward of ad i upto round n.
# Calculate the confidence interval. delta i(n)
# n -> Total number of rounds = 10000.
# Here we compute only the upper bound of the confidence interval.
# 1st for loop -> To loop over 10000 rounds (N).
# 2nd for loop -> To loop over 10 versions of the ad (d).
for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range(0, d):
        if(numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = delta_i + average_reward
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
        
# Here the Upper Confidence Bound is -> Average reward + delta(i).
# The indexes of python start with zero.
# Step 3 -> Select the ad version i which has the maximum upper bound
# We need to create a huge vector that contains different versions of the ad selected at each round.
# Here we append the different ads that we selected up to the last round round 10000.
# Since the maximum upper bound variable is going to be different at each round. So we need to initilize it at each round.
# Then we will compute upper bounds for each these 10 ads. Then we will compare these upper bounds to maximum upper bound.
# At each time the upper bound computed is higher than the maximum upper bound. Then we will set maximum upper bound equal to upper bound.
# Here we also need to select the ad with max_upper_bound.
# We also need to track the index of the max_upper_bound.
# Here i corresponds to the value that corresponds to the specific ad.
# We will select first 10 ads without using the strategy here. We will use these strategy as soon as we have some information of rewards of each of 10 ads. During these first 10 rounds we select the first ad that is 1 for 1st ad for round 1, 2 for the 2nd for round 2 ...... 10 for the 10th round.
# After round 10 the number of selections will be one for each of the 10 ads
# We also get some information about sum of rewards and number of selections of the first 10 rounds.
# The strategy will be applied after the first 10 rounds

# Why are we giving a very large number to upper bound in else condition?
# The if condition will never be true as no ads were selected in the first round. Therefore we directly go to the else condition where the upper bound will be set equal to 1e400.
# And in the second if condition we get ad = i. As we are beguinning of the for loop as i = 0 we get the ad = 0.
# The second for loop executes for second time and then here we select the ad = 0 (first ad).
# Then we go to the next step in first if condition where i = 1 and we select the ad = 1 and so on (Corresponds to the second ad).
# In this way we select the first 10 ads in the first 10 rounds that is 1, 2, 3...... up to 10 (for first 10 rounds).
# After round 10 we use the strategy of first if condition to select the ads.

# Then we need to append the ad selected to the ad_selected vector.

# Visualizing the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('ads')
plt.ylabel('Number of times each ad was selected')
plt.show()




