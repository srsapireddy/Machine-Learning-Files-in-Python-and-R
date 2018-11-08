# Convolutional Neural Network -> A type of Artificial Neural Network
# To classify some images and videos.
# Our goal is to classify the image and to tell the class of the image.

# Why do we use convolutional layers?
# To preserve the spatial structure in images and therefore to be able to classify some images.

# Independent Variables --> pixels distributed in 3-D arrays representing the images.
# Here our dataset contains the images in two different folders. One for the training set and one for the test set.
# Here we use keras to import the images.

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 --> Building the CNN

# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Sequential --> To initialize our neural network in to sequence of layers.
# Convolution2D --> To add the convolutional layers to deal with images.
# Dense --> To add a classic ANN. A fully connected layers.

# Intitializing the CNN
classifier =    Sequential()

# Step 1 --> Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Convolution2D Function
# 1. nb_filter --> Number of filters. That is number of feature detectors we are going to apply on our input image to get the same number of feature maps. There will be one feature map created for each filter used.
# 2. Number of rows in the convolutional kernal (Other name for the feature detector)
# 3. Number of columns of our feature detector
# 4. border_mode --> To specify how the feature detectors will handle the borders of the input image.
# 5. Input_Shape --> The shape of our input image. To specify what is the expected format of our input images.That is the format that our images will be converted.
# input_shape(number of channels, dimensions of the 2D array in the each of the channel) --> This order is based on the theano backend but we are using the tensorflow backend.
# 6. Activation Function --> In ANN we use a activation function to activate the neurons. But here we use an activation function just not to have any negative value pixels in the feature maps in order to have non-linearity in our CNNs. This activation function helps to get the non-linearity.

# Step 2 --> Pooling --> Reducing the size of the feature maps. --> We will get half of the size of the orginal image that is half the size.
# We apply this inorder to reduce the number of nodes in the next step that is flatterning and fully connected layers.
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding second convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 --> Flatterning --> Taking all the pixels in our feature map and put them in our single dimension vector.
classifier.add(Flatten())

# Step 4 --> Full Connection
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(1, activation = 'sigmoid'))

# output_dim --> Number of nodes in the hidden layer.
# Activation Function --> The nodes in the hidden layers are like neurons that we need to activate according to how much they can pass on the signal.
# Rectifier activation function returns the probabilities of each class. Here we use the sigmoid activation function as we have a binary outcome.
# If we have more than two classes we need to use Softmax Function.

# Compiling the CNN
# We need to compile our CNN by choosing a stocastic gradient descent algorithm that is a loss function and eventually a performance metric.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Optimize parameter --> To choose the stochastic gradient descent algorithm.  
# loss --> A Parameter to choose the loss function.
# Metrics Parameter --> To choose the performance metrics.

# Part 2 --> Fitting the CNN to the images.
# We use keras documentation --> It is basically for image augementation that is basically pre-processing the images to prevent overfitting.
# If we dont do this we will get a great accuracy on the training set and very low accuracy on the test set. This is due to overfitting on the training set.
# One of the situations that can lead to overfitting is when we have few data to train our model.
# Data augumentation --> Will create many batches of our images and in each batch it will apply some random transformations (Thus our model does not find the same pictures in our model.) on random selection of our images like rotating them, shifting them and flipping them.So we gat many more training images in these batches and therefore lot more material to train our machine learning model. So image augufmentation reduces the overfitting with small amount of images.
# Transformations applied in image augumentation are:
# 1. rescale --> This corresponds to the feature scaling part of data pre-processing.
# Scaling all the pixel values from 0 and 1. Because our pixel values are between 0 and 255.
# 2. sheer range --> geometrical transformation --> Where the pixels are moved to a fixed direction over a proportional distance from a line that is parallel to the direction they are moving to.
# 3. zoom range --> This is some kind of random zoom that we apply to our images. 
# 4. horizontal flip --> This flips the image horizontally.
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(64, 64), batch_size=32, class_mode ='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

# To fit the training set and test set to the CNN
classifier.fit_generator(training_set, steps_per_epoch=8000, epochs=25, validation_data=test_set, validation_steps=2000)

# Making deeper Concolutional Neural Network --> Increase the accuracy on the test set and reduce the overfitting
# [1] Adding another convolutional layer
# [2] Adding another fully connected layer

# To increase the accuracy more we need to choose an higher target size than 64 so that we get more information about our pixel patterns. As we get lot more pixels in the rows and lot more pixels in the columns. Therefore we have more information to take on the pixels.
