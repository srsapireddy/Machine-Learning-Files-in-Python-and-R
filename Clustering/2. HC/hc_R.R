# Hierarchical Clustring

# Importing mall dataset
dataset <- read.csv("Mall_Customers.csv")
X <- dataset[4:5]

# Using dendrogram to find the optimal number of clusters
# hclust -> Class
# d -> Distance matrix of our dataset X. Which tells that for each pair of customers the euclidean distance between the two.
# So that for each pair of customers we take the two coordinates annual income and spending score and we compute the euclidean distance between the two based on these coordinates.
# Method -> Simply to find the clusters
# Ward.D -> To minimize the variance between each cluster.
dendrogram = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
plot(dendrogram, main = paste('dendrogram'), xlab = 'Customers', ylab = 'Euclidean Distance')

# Fitting the hierarchical clustering to the mall dataset
# Buiding the vector of clusters (y_hc)
hc = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc, 5)

# Visualizing the Clusters
library(cluster)
clusplot(X, clus = y_hc, lines = 0, shade = TRUE, color = TRUE, label = 2, plotchar = FALSE, span = TRUE,
         main = paste('Clusters of Clients'), xlab = 'Annual Income', ylab = 'Spending Score')

