# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./figures/raw_image_plot.png "Visualization"
[image2]: ./figures/class_distribution.png "Class Distribution"
[image3]: ./figures/raw.png "Raw Images"
[image4]: ./figures/grayscale.png "Grayscaling"
[image5]: ./figures/aug.png "Data Augmentation"
[image6]: ./figures/loss_acc_raw.png "Loss and Accuracy Plot"
[image7]: ./figures/new_images.png "New Test Images"
[image8]: ./figures/predictions_0.png "Predictions of Image 0"
[image9]: ./figures/predictions_1.png "Predictions of Image 1"
[image10]: ./figures/predictions_2.png "Predictions of Image 2"
[image11]: ./figures/predictions_3.png "Predictions of Image 3"
[image12]: ./figures/predictions_4.png "Predictions of Image 4"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/zexihan/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Below is the summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4000
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The first plot shows some raw training images in the data set. Images in the same row belongs to the same class.

![alt text][image1]

The second plot is a histogram showing the distribution of the 43 traffic sign categories. We can see that the imbalance exists in the classes. Traffic signs have varying brightness and rotation angles. 

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert all the images to grayscale so as to correct the varying contrast and brightness of the images.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

![alt text][image4]

As a following step, I normalized the image data because in the process of training our network, we're going to be multiplying (weights) and adding to (biases) these initial inputs in order to cause activations that we then backpropogate with the gradients to train the model. We'd like in this process for each feature to have a similar range so that our gradients don't go out of control (and that we only need one global learning rate multiplier).

At last, I performed data augmentation. I decided to generate additional data in order to improve the performance and ability of the model to generalize.

To add more data to the the data set, I used random rotation and cropping. Here is an example of some augmented images:

![alt text][image5]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer (type)        	|     Output Shape	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   				    | 
| Convolution 32, 5x5   | 1x1 stride, same padding, relu activation, outputs 32x32x32 	|
| Max pooling	      	| 2x2 stride, outputs 16x16x32   				|
| Convolution 32, 5x5   | 1x1 stride, same padding, relu activation, outputs 16x16x32    |
| Max pooling    		| 2x2 stride, outputs 8x8x32     				|
| Flatten				| outputs 2048        							|
| Fully connected		| relu activation, outputs 120					|
| Dropout				| 0.6 keep rate									|
| Fully Connected		| relu activation, outputs 84					|
| Dropout				| 0.6 keep rate									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer. Adam is a replacement optimization algorithm for stochastic gradient descent for training deep learning models. Adam combines the best properties of the AdaGrad and RMSProp algorithms to provide an optimization algorithm that can handle sparse gradients on noisy problems.

I experimented with various combination of batch size and number of epochs. One hundred epochs is surely enough or over-killed for the task. I found small batch sizes usually quickly eat up most of the loss. As I have four powerful GPUs, the model trained with batch size of one performed the best.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 95.87%
* validation set accuracy of 96.03% 
* test set accuracy of 92.48%

First I applied the original LeNet architecture for the task. Then I focused on tuning the # of filters for the convolutions. The significance # of filters (or feature detectors) intuitively is the number of features (like edges, lines, object parts etc...) that the network can potentially learn. Also note that each filter generates a feature map. Feature maps allow you to learn the explanatory factors within the image, so the more # filters means the more the network learns (not necessarily good all the time - saturation and convergence matter the most). Finally I increased the # of filters of the first convolution from 6 to 32.

Next problem is the overfitting. Overfitting is bad because the model has extra capacity to learn the random noise in the observation. To accommodate noise, an overfit model overstretches itself and ignores domains not covered by data. Consequently, the model makes poor predictions everywhere other than near the training set. I added dropout layer following each fully connected layer. Dropout is a simple way to prevent neural networks from overfitting. Because the outputs of a layer under dropout are randomly subsampled, it has the effect of reducing the capacity or thinning the network during training.

In the end, by looking at validation metrics, loss and accuracy curves, we can see that the model converged to an ideal status.

![alt text][image6] 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7] 

The last image might be difficult to classify because it is not photographed from the real world like the images in the data set.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (120km/h) | Speed limit (120km/h)   						| 
| Road work     		| Road work 									|
| Priority road			| Priority road									|
| No vehicles	   		| Yield			        		 				|
| Vehicles over 3.5 metric tons prohibited | Go straight or left    	|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is relatively sure that this is a Speed limit (120km/h) sign (probability of 0.95), and the image does contain a Speed limit (120km/h) sign. The top five soft max probabilities were

![alt text][image8] 

For the second image, the model is sure that this is a Road work sign (probability of almost 1.), and the image does contain a Road work sign. The top five soft max probabilities were

![alt text][image9] 

For the third image, the model is sure that this is a Priority road sign (probability of almost 1.), and the image does contain a Priority road sign. The top five soft max probabilities were

![alt text][image10] 

For the fourth image, the model is sure that this is a Yield sign (probability of almost 1.), but the image contains a No vehicles sign. The top five soft max probabilities were

![alt text][image11] 

For the last image, the model is not sure that this is a Go straight or left sign (probability of almost 0.11), and the image contains a Vehicles over 3.5 metric tons prohibited sign. The top five soft max probabilities were

![alt text][image12] 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


