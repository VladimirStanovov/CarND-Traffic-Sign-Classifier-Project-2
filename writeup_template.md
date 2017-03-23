#**Traffic Sign Recognition** 

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

[image1]: ./barchart.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./(1).png "Traffic Sign 1"
[image5]: ./(2).png "Traffic Sign 2"
[image6]: ./(3).png "Traffic Sign 3"
[image7]: ./(4).png "Traffic Sign 4"
[image8]: ./(5).png "Traffic Sign 5"
[image9]: ./(6).png "Traffic Sign 6"
[image10]: ./(7).png "Traffic Sign 7"
[image11]: ./(8).png "Traffic Sign 8"
[image12]: ./(9).png "Traffic Sign 9"
[image13]: ./(10).png "Traffic Sign 10"
[image14]: ./(11).png "Traffic Sign 11"
[image15]: ./(12).png "Traffic Sign 12"
[image16]: ./(13).png "Traffic Sign 13"
[image17]: ./(14).png "Traffic Sign 14"
[image18]: ./(15).png "Traffic Sign 15"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
##Writeup

Here is a link to my [project code](https://github.com/VladimirStanovov/CarND-Traffic-Sign-Classifier-Project-2/blob/master/Traffic_Sign_Classifier.ipynb)

##Data Set Summary & Exploration

After loading the data in the first code cell, I used the .shape of each array to get the size of the trainig, validation and test sets, as well as the image shape. To figure out what is the number of unique classes, I used a little function that created a dictionary of unique class numbers.

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

After this I have also calculated the number of instances in each class.

##2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

Here I have imported the sign names from signnames.csv file and created an array signNames filled with text interpretation of each class number. Next, I used the matplotlib to plot a bar chart of number of instances for each class:

![alt text][image1]

The dataset appears to be imbalanced, with maximum imbalance ration of around 10.
After this I have also added a visualization of a random sign - just to be sure that everything is good.

###Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because several papers on image classification with convolutional neural networks appear to show better performance on grayclaed images.
The grayscale images are appended to the original data, so that each image now has 32x32x4 size.

####2. I have also tried artificial balancing of the dataset by generating additional images of minority classes, so that the number of instance for each class equal to 2010. This preprocessing step included jittering the image, as well as adding random noise. Howerer it appeard that this does not deliver any accuracy improvements (it actually makes it even worse), so this code was deleted from the final version.

####3. The model architecture is following: the image is split into two separate parts, i.e. color image and grayscale. For each of these two three layers of convolutions are used, folloed by three fully connected layers. First fully connected layer includes conntections to both second and third convolution layers.

My final model consisted of the following layers:

| Layer         		|     Description	        					|    
|:---------------------:|:---------------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|     
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Convolution 3x3	    | 14x14x24, valid padding, outputs 10x10x32      									|
| RELU					|												|     
| Max pooling	      	| 2x2 stride,  outputs 6x6x32 				|
| Convolution 3x3	    | 5x5x32, valid padding, outputs 2x2x64      									|
| RELU					|												|     
| Input         		| 32x32x1 grayscale image   							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|     
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Convolution 3x3	    | 14x14x24, valid padding, outputs 10x10x32      									|
| RELU					|												|     
| Max pooling	      	| 2x2 stride,  outputs 6x6x32 				|
| Convolution 3x3	    | 5x5x32, valid padding, outputs 2x2x64      									|
| RELU					|												|     
| Fully connected		| input: 2816 (layer-2 and layer3, RGB and gray), output: 352 	|  
| RELU					|												|     
| Dropout					|						0.5						|     
| Fully connected		| input: 352, output: 176      									|  
| RELU					|												|     
| Dropout					|						0.5						|     
| Fully connected		| input: 176, output: 43      									|  
| Softmax				| etc.        									|  
|						|												|  
|						|												|  
 
I used the dropout only for two fully connected layers, as this appeared to be the best choice.

####4. Model training

The code for training the model is located in cellы 15-19 of the ipython notebook. I used the AdamOptimizer with rate = 0.001, and also included the L2 loss for reguralization with beta = 0.001. The rest of the code is similar to the one used in LeNet lab. I calculated the training and validation accuracy after each epoch. After the training is finished, i calculated the test accuracy.

####5. The approach taken for finding a solution

The architecture used includes 2 separate convolutional nets of three layers, which receive RGB and grayscale images as inputs. This looked like a good idea, because this could combine advantages of each image encoding, providing more features to the network. Moreover, the fully connected layer gets inputs not only from last convolutional layer, but also from previous one, so that the network is capable of getting both high-level and low-level features - some of them could be of much importance for classification.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.974
* test set accuracy of 0.961

The iterative structure tuning procedure was the following:
* First I tried the classical LeNet architecture, which resulted in accuracy of around 0.89.
* This looked like underfitting, so I have increased the width of the model - made deeper convolutions, but kept the number of layers same.
* Various experiments with LeNet architecture got me to the conclusion that it is not cabable enough - the problem is more complicated, so larger architectures should be used. First experiments were including more fully connected and convolutional layers - this gave a little performance boost. I ended up with 3 convolutions and 3 fully connected layers. Adding more parameters to these, as well as adding connections from conv2 and conv3 layers to first fully connected by Yann LeCun's recomendation in his paper on German traffic signs detection resulted in 100% trainig accuracy, however, validation accuracy was still quite low - around 96%. So the next step was to introduce regularisation - I tried to apply L2 loss, which gave a little result, but including dropout was much more effective - I was able to get around 97.5% accuracy on validation set. However, dropout appeared to be good only for two fully connected layers - adding it to convolutions decreased the performance. After all this, I have added the grayscale images and made a second 'branch' of the network to process them, shich resulted in validation accuracy reaching 98%. At the end, the test accuracy that I got was around 0.961.
* Testing each architecture included trying different sigmas for the initialization, but at the end 0.05 appeared to be a good choise. Larger values gave a lot of overfitting, while smaller values resulted in a much longer training.


###Testing a Model on New Images

####1. Get different German traffic signs

I have asked my colleague living in Germany to make some photos of traffic signs. Here is what I got:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11] ![alt text][image12] 
![alt text][image13] ![alt text][image14] ![alt text][image15] 
![alt text][image16] ![alt text][image17] ![alt text][image18]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

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
