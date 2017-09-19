# Project 3: Behavioral Cloning Project**

## Overview
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[nvidiaCNN]: ./image_references/nvidia_model_CNN.png "nvidiaCNN"
[left]: ./image_references/leftImage.jpg "left"
[center]: ./image_references/centerImage.jpg "center"
[right]: ./image_references/rightImage.jpg "right"

[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Introduction
The goal of this project was to explore the Behavioral Cloning techniques within Deep Learning. Behavioral Cloning refers to adopting existing Deep Neural Network architecture for use in a similar but slightly different purpose than the network was originally designed for. The objective of this project is to adopt and fine tune a Deep Neural Network to train a simulated car to drive autonomously around a simulated track. 

First, the user generated training data by driving the car around the track via remote control. The training data consisted of 3 on-board camera images from the car and the corresponding steering angle. Next, the camera images and steering angle acted at the training data and labels, respectively, as input into the Deep Neural Network. The network was trained to accept camera images and predict a single output, steering angle, to keep the car within the bounds of the track. After the network was trained, the model was implemented in the same simulator, this time with the network autonomously driving the car. 

There were many challenges that ensued within this project. Because the only output is the steering angle, its tough to gauge exactly where the data augmentation, data pre-processing, or network training needs to be adjusted. There were many iterations 

## Data Collection, Preparation and Augmentation

### Data Collection
This project showed that the Deep Learning is all about the data. In the words of my coworker, [Sunil Bharitkar](<https://www.linkedin.com/in/sunil-bharitkar-3293832>), "Garbage in is garbage out." There were different methods of collecting data. At first, it was easy to think that driving the car down the center of the track multiple times teaches the DNN the identical process. Upon further examination, the car needed to learn how to recover from all sorts of situations. Therefore, the data included recoveries and windy track laps. Also, the track primarily included left hand turns, but rather than drive the track in reverse, the images and corresponding steering angles were flipped prior to sending to the network for training. 

### Data Preparation
#### Original Images
At each given moment, three camera images were collected. Images were categorized as left, center, right images and had dimension 320x160x3.

Left Camera Image: ![alt text][left] 

Center Camera Image: ![alt text][center] 

Right Camera Image: ![alt text][right]

#### Cropped Images
The images were cropped 20% from the top and bottom. This was performed in order to help the neural network to focus on recognizing features on the road and eliminate features from the sky, trees, hood of the car, etc. Below is the center camera image after cropping. 

Cropped Camera Image: ![alt text][cropped]

#### Resize Images
The images were then resized 

#### Match with drive.py
The same cropping and resizing was performed in `drive.py`, lines 70-71 , the python file which runs the trained model through the simulator.

### Data Augmentation


## Model Architecture and Training Strategy

### Model Architecture

The inspiration for my model was from the NVIDIA End to End Learning for Self Driving Cars [paper](<http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf>). 

![alt text][nvidiaCNN]

The input NVIDIA used is 66x200x3 RGB colored images with an output of the inverse of steering angle. The input to this project's network is 100x100x3 RGB colored image with the a single neuron output of steering angle. Three 5x5 convolution layers, followed by three 3x3 convolution layers, followed by 4 fully connected layers were implemented in this project. Subsampling or max pooling was used within each convolution. The activation function used in each neuron is RELU. Dropout of 50% was used after each set of convolutions and within each fully connected layer to combat over fitting. The Adam optimizer was chosen with default parameters and the chosen loss function was mean squared error (MSE). 

### 2. Model Training
The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


## UDACTIY SHIT
### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 



## Data Preparation and Augmentation

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was over fitting. 

To combat the over fitting, I modified the model so that ...

Then I ... 
The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
