# **Traffic Sign Recognition** 

## Writeup, Joseph Rogers

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/josephRog/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

After I loaded the images from the data set, I chose to convert them to grayscale, and then normalize them to the -0.5 to 0.5 range before sending them to the networks. I did this because I wanted to make sure all of the images had the mean of their color values centered around 0. Additionally I wanted to have all of the pixtures be grayscale so that signs with odd colors or even filters could still be recognized for the signs they represent.

I experimented with augmenting the data by adding random rotation, translation and scale to the signs, but it did not end up making a noticable difference in the accuracy of the network. It did however make it take significantly longer to run the calculations when training the network. At a result, I did not include these operations in the final version of the project.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, padding = 'VALID', Output = 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6				|
| Convolution 3x3	    | 1x1 stride, padding = 'VALID', Output = 28x28x6		|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  Output = 5x5x6				|
|Flatten     |     Input = 5x5x16, Output = 400|
| Fully connected		| Input = 400, Output = 200	|
| RELU					|	
| Fully connected		| Input = 200, Output = 120	|
| RELU					|	
| Fully connected		| Input = 120, Output = 84	|
| RELU					|	
| Dropout Layer | keep_prob = 0.5			|	
| Fully connected		| Input = 84, Output = 43	|
| Softmax				|      									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

In order to train the model I used a batch size of 64, learing rate of 0.001, and epoch count of 100. I switched to a batch size of 64 since it worked fine on my computer and allowed me to train using more data at once. I experimented with changing the learning rate from the default value of 0.001 with mixed results. Initially I thought about using a slower learning rate with more epochs but this led to the model finding local maxima too easily. Using a higher learing rate made the model converge much faster but to a worse accuray than before. I experimented with many different values of epochs and eventually settled on 100. This allowed the network to be completely trained in about 5 minutes on an NVIDIA GTX 980ti. When training the network I noticed that the validation accuracy always seemed to stop improving around epoch 60 to 80. I decided to let it go out to 100 on the off chance that it discovered some better weight but didn't see much point in continuing after that.

As I trained the network trying to find better values for hyperparameters, I noticed that the results sometimes fluctuated quite a bit, and often the best network was actually NOT produced during the final epoch. This gave me the idea of only saving the network that had given the best validation accuracy instead of saving the network produced during the final epoch. I expanded on this idea but training the network from scratch several times and comparing the best results for each of those training runs. This allowed me to train a network with a high accuracy much more reliably.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of ? 
* test set accuracy of ?

#### If an iterative approach was chosen:
#### * What was the first architecture that was tried and why was it chosen? 
I began by using the default LeNet architecture. I changed it slighly so as to accept grayscale images and to output to 43 different class types.

#### * What were some problems with the initial architecture?
Some problems with the initial architecture was that it didn't have any dropout. I was concered that this would lead to overfitting on the training data. 

#### * How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
As I mentioned before, the lack of a dropout layer made me worry about overfitting to the training data. I added a dropout layer just before the final output to address this. When training it had a 50% chance to drop nodes. I also added an extra fully connected layer to reduce the rate at which the number of layer outputs was decreased. This did not conclusively chance the performance of the network for better or for worse.

#### * Which parameters were tuned? How were they adjusted and why?
I experimented at length with tuning all of the models hyperparameters. Learning rate ended staying at the default of 0.001 but I reduced batch size to 64, and increased epochs to 100. This was to give the model more time to converge and allow it to train with more of the data at once during a given epoch.

#### * What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
I think it makes a lot of since to use convolution layers to help make a good model for this problem. Having a convolutional nearual network should allow for much more flexibility with being able to identify and classify a sign in a picture even if the picture is not ideal. Specifically, a sign should able to be able to be recognized and classified independent of its location, rotation or scale within the image frame. The images provided as training for this project seemed to all be centered on the sign and of the same scale and orientation. However this will likely not always be the case.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


