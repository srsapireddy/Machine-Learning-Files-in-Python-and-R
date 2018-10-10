# Natural Language Processing - Analysing Text.

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)

# Here we create the bag of words model representation - This consists of getting the rlevant words in the reviews here. That is we get to reduce the unused words. And we also get rid of punctuation marks. We also get rid of numbers unless numbers have a significant impact.
# We also do whats called as stemming - which consists of taking the routes of the different versions of a same word. (Its just like taking the tenses from the words [ex - love --> loved or loves]). We apply this not to have too many words in the end. Its like regrouping the same versions of same word.
# We also get rid of the capitals so that we only have the texts in the lower case.
# Last Step - Creating the bag of words model -> Which is tokenization process. -> It splits all the different reviews into different words and only take the relevant words (Also known as text pre - processing)
# And then we will attribute one column for the each word.
# Then for each review each column will contain the number of times the associated word appears in the review.
# We will have lot of zeros because a lot of words dont appear in the rview and have ones for the words that appear once in the review and so on.And here we get a sparce matrix as we are getting a lot of zeros.Because for most reviews most words will not be in the reviews.

# Cleaning the Texts - That is cleaning each one of the review.
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
# Third Step -> To remove the non significant word.
# This include all the articles and prepositions.

from nltk.stem.porter import PorterStemmer


corpus = []
for i in range(0, 1000):
# We will build a new list of 1000 reviews. We create a list as corpus as we create a corpus of clean 1000 reviews.
# Corpus --> Is a collection of text of anything. Collection of the same type.

    review = re.sub("[^a-zA-Z]", " " ,dataset["Review"][i])
# Sub --> For only keeping the letters.
# Forrlwing the ^ is the words we dont want to remove.

    review = review.lower()
# Second Step -> Putting the review words in the lower case.

    review = review.split()
# Here our goal is to remove too much sparcity in the sparce matrix.This sparse matrix has each word has its own column.
# We will make a for loop to go through all the words through this review. And remove all the insignificant words.
# Here the review is a string and we need to split this string into different words to go through.
# After this review will become a list of different words.
# We are using set for much faster search of stopwords. This is faster than words in a list.

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english "))]

# Fourth Step -> Stemming.
# Stemming --> Taking the route of the words.
# Ex-> Loved is the past tense of the love word.
# In order to use this stem function we need to import a class and then we will create an object of this class which will use the stem function on each word in the review.

# Last Step --> Joining back the different words of the review list so that the list again becomes the string.
    review = ' '.join(review)
    
# To apply cleaned review to our corpus.
    corpus.append(review)

# Creating Bag of Words Model --> Take all the words of the 1000 reviews.Here we will take the different unique words without duplicates or triplicates of these 1000 reviews and create one column for each word and we will put all these columns in a table where the rows are nothing the less the 1000 reviews.
# Basically what we will get is a table of 1000 rows corresponds to the reviews and columns corresponds to the different words in this corpus.
# So each cell of this table will correspond to one specific review  and one specific word of the corpus. And in this cell we gonna have a number and this number is going to be the number of times the word corresponding to the column appears in the review.
# And here we get a matrix with lot of zeros and a matrix containing lot of zeros is called as a sparse matrix.
# Creating this sparse matrix is the bag of words model. --> Creating the bag of words model through tokenization.
# Tokenization --> Is the process of taking all the words of the review and creating one column for each of these words.
# Then we will train the machine learning model on all these words in the review. The machine learning model will create correlation between the words and reviews and the real result.
# In order to train the machine learning model if the review is posotove or negative if needs to have some independent variables and one dependent variable because here we are doing here is classification. Because the outcome the dependent variable is a catagorical variable a binary outcome 1 if the review is positive and 0 if the review is negative.
# In bag of words model each of the column corresponds to one specific word is one independendent variable itself.
# So when the word appears in the review the column gets a 1 and if it dosent appear in review the column gets a 0.
# Columns -> Words
# Rows -> Reviews
# At the end we will have the matrix of features of matrix of independent variables which will be the different words appearing in all the reviews here that will be columns of the matrix and we will have the dependent variable vector which will be the result which will be 1 --> positive or 0 --> negative.


# Max_features --> To keep only most frequent words.
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()

# To train our machine learning models which are classification models on bag of words model we need a dependent variable.
# iloc --> To take the index of the dependent variable column.
y = dataset.iloc[:,1].values

# Independent Variables (1000 rows, 1500 columns)
# 0 ->> If word appears in the review
# 1 --> If word dosent appears in the review.
# Here now we try to understand the correlation between the presence of words and the reviews and the outcome 0 if it is negative review and 1 if it is positive review.

# Sparcity --> Lot of zeros.
# Confusion Matrix --> Number of correct and incorrect predictions.

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Classifier to Training Set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predicting the Test Set Results
# y_pred --> Vector of predictions
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy
(55+91)/200

