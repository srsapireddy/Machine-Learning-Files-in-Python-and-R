# K-Means Clustering

# Importing the mall dataset
dataset <- read.csv("Mall_Customers.csv")
X <- dataset[4:5]

# Using the elbow method to find the optimal number of clusters
set.seed(6)
wcss <- vector()
for (i in 1:10) wcss[i] <- sum(kmeans(X,i)$withinss)
plot(1:10, wcss, type = 'b', main = paste('Clusters of Clients'), xlab = 'Number of Clusters', ylab = 'WCSS')

# Applying K-Means to our mall dataset
set.seed(29)
kmeans <- kmeans(X, 5, iter.max = 300, nstart = 10)

# Visualizing the Clusters
# Clus <- Vector of clusters (The vector that returns the each observations which cluster it belongs to)
# lines <- distance of lines between the clusters
# Shade <- So that the clusters are shaded according to their density
# Labels <- So that all points in the clusters are labeled in the plot
# plotchar <- To have different symbols for points in the different clusters
# SPAN <- To plot ellipse around the clusters
install.packages('cluster')
library(cluster)
clusplot(X, clus = kmeans$cluster, lines = 0, shade = TRUE, color = TRUE, label = 2, plotchar = FALSE, span = TRUE,
         main = paste('Clusters of Clients'), xlab = 'Annual Income', ylab = 'Spending Score')

