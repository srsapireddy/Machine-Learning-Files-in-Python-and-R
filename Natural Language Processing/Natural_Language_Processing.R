# Natural Language Processing --> Analysing Texts

# Importing the dataset
# Factor --> A single entity having a single meaning regardless of different meanings of the different words of the review.
# quote --> To ignore the quotes.
dataset_orginal = read.delim("Restaurant_Reviews.tsv", quote = "", stringsAsFactors = FALSE)

# Cleaning the Texts
# The goal here is to create some independent variables.
# This table will have 1000 rows because we have 1000 reviews. One row for each review and the columns will simply be all the words that we can find in the reviews.
# Thus we have one column for each word and then each cell in the table will correspond to that is one row that is one review and one column that is one word in all these words and these 1000 reviews.
# And then the each cell is going to be the number of times the word appears in the review.
# Bag of Words Model --> Sparse Matrix because we get huge zeros in the sparse matrix and that we will get a table having lot of independent variables and one dependent variable.
# And then we will be using our Machine Learning Classification Models to predict the class of a new review the model not have seen yet.
# Spep 1 --> Initializing a Corpus. Because we will not clean the review directly in the dataset. We will instead create a Corpus which will contain all the reviews and that will be in this Corpus we will clean all the reviews.
install.packages("tm")
library(tm)
corpus = VCorpus(VectorSource(dataset$Review))
# Corpus is a new dataset. Containing only the reviews only texts of the reviews.
# Sparse Matrix --> Matrix of Features.

# Step 1 --> Putting all the reviews in the lowercases. We are doing this because we dont want the same word starting with the capital letter and the uncapital letter. Thus to havve the one version of the same word. The one with the lowercase.
corpus = tm_map(corpus, content_transformer(tolower))

# Step 2 --> Removing all numbers of the reviews. As they are not relevant to tell if the review is positive or negative. And this could add lot more columns.
corpus = tm_map(corpus, removeNumbers)

# Step 3 --> Removing any kind of punctuation in the reviews.
# Because in the sparse matrix we dont want to have a colon in place of a comma or an other colon for a colon or other colon for a dot or for a semicolon or any kind of punctuation.
corpus = tm_map(corpus, removePunctuation)

# Step 4 --> Removing all the non-relevant words in the reviews.
# Here common and non-relevant words are removed.
install.packages("SnowballC")
library(SnowballC)
corpus = tm_map(corpus, removeWords, stopwords())

# Step 5 --> Stemming Step
# Stemming --> Getting the root of the each word.
corpus = tm_map(corpus, stemDocument)

# Step 6 --> Removing the extra spaces.
corpus = tm_map(corpus, stripWhitespace)

# Creating the Bag of Words Model.
# Its time to create the sparse matrix of features containing all the different reviews in the rows and all the different words of the reviews in the columns.
# We gonna have one row for each review and one column for each word in the table.
# Then for each cell in this table each cell corresponds to one review corresponding to the row and one word corresponding to the column. 
# So the value contained in the cell is the number of times the word appears in the review.
# Sparse Matrix --> Sparse matrix is table containing lot of zeros.
# Sparsity --> A situation where we have lot of zeros.
# We are creatin this table to have the framework of classification Models. That is to have the independent variables and a dependent variable.
dtm = DocumentTermMatrix(corpus)

# Only taking the words that are more frequent. For example if the word appear in only one review and these words are not frequent and appear only once.
dtm = removeSparseTerms(dtm, 0.999)

# Making  Machine Learning Classification Model.
# We need to create the independent variables and one dependent variable which will be the input for our Machine Learning Classification Model.
# But here the dtm is a matrix and well in R the classification models works on dataframes.
dataset = as.data.frame(as.matrix(dtm))
# But here we need a dataframe of dependent variables and an independent variable. So we need to add the dependent variable to the dataset.
dataset$Liked = dataset_orginal$Liked
# Here we use Random Forest Classification as our Machine Learning Classification Model.

# Random Forest Classification Model.
# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set
install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692], y = training_set$Liked, ntree = 10, random_state = 0)
# Here x is the training set without the dependent variable.

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)

# So from the Confustion Matrix we have 51 incorrect predictions. Out of 200 new reviews.
# Accuracy = Number of correct predictions / Total number of predictions (Test set reviews --> 200 here)


