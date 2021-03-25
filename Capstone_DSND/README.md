# Project: Write an Algorithm for a Dog Identification App 
## Convolutional Neural Networks

This project is completed as a partial requirement for the UDACITY’s Data Scientist nanodegree program.

# Table of Contents


### 1. [Project Motivation](#motivation)
### 2. [Installations](#installations)
### 3. [Project Overview](#overview)
### 4. [Results of the Algorithm on Sample Images](#results)
### 5. [Project Folders and Files](#tree)
### 6. [Acknowledgments](#ack)

<a id='motivation'></a>
# 1. Project Motivation

This project takes central place in learning Artificial Intelligence domain. Since it deals with a very complex task i.e., the identification of the dog breed. The classification problem such as classifying human or dog is somewhat easy as compared to classification problems that deal with identifying dog breed and human race. The latter problem requires complex algorithms, careful analysis and huge amount of training data. The results of this project are encouraging because it identifies correctly dog breeds. These algorithms could also be used in other situations such as identifying human races, categorizing male, female or classifying humans like kids, young and older by utilizing corresponding data. Today, we use to see the application of this project in every day’s life, security cameras might recognize the individuals and their background by just analyzing their faces. We also see several facial recognition apps such as Face ID in latest iPhone. All these reasons motivated me to choose this topic to dive deeper in this project and I literally enjoyed doing each step of the project.

<a id='installations'></a>

# 2. Installations

In order to run the project files properly, you need to install folowing python libraries.

* from sklearn.datasets import load_files       
* from keras.utils import np_utils
* import numpy as np
* from glob import glob
* import random
* import cv2                
* import matplotlib.pyplot as plt
* from keras.applications.resnet50 import ResNet50
* from keras.preprocessing import image                  
* from tqdm import tqdm
* from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
* from keras.layers import Dropout, Flatten, Dense
* from keras.models import Sequential



<a id='overview'></a>

# 3. Project Overview

This project mainly applies a CNN Convolutional Neural Network to Classify Dog Breed using Transfer Learning

### 1.	Obtain Bottleneck Features

It begins with the extracting the bottleneck features corresponding to the train, test, and validation sets.

### 2.	Model Architecture

In this project I developed the model named as (my_model), I used Convolutional Neural Network CNN Architecture by using Resnet50. my_model also follows the archetecture defnined above in the VGG16 model. I applied GlobalAveragePooling2D in order to flatten the features that could be fed into the fully connected layer to the pretrained selected model. The DENSE layer was implemented with 133 nodes (one for each dog category) alonwith softmax function. This new algorith with transfer learning performed much better and arrive at the accuracy (78%) that is above desired accuracy level (60%).


### 3.	Compile the Model

In the next step of the model building, I compile the model by using three parameters such as optimizer, loss and metrics.

### 4. Train the Model

Next step is training the model. I used model checkpointing to save the model that attains the best validation loss (suggested by the project notebook). In order to train the model we need to use the `fit()` function alogwith the required parameters.

### 5. Load the Model with the Best Validation Loss
In the next step, I loaded the best saved model in the previous step with the best validation loss.

### 6. Test the Model

Now, it's time to test your model. It was required to attain at least 60% accouracy as specified in the notebook "Try out your model on the test dataset of dog images. Ensure that your test accuracy is greater than 60%." The Test accuracy of my_model is: 78.2297%

### 7. Predict Dog Breed with the Model
In the next step, I predicted the dog breed by bottleneck features in the function.


<a id='results'></a>

# 4. Results of the Algorithm on Sample Images

In the previous step I developed an algorithm to predict whether the image (input) contains a dog, a human or neither. Possible outcomes of this algorithm are:

* If a dog is detected in the image, it returns its corresponding breed
* If a human is detected in the image, it returns most resembling dog breed.
* In third situation if neither is detected it may return an error message.

You can read the blog post of this project [here](https://webdevelopmentyschools.wordpress.com/2021/01/10/project-write-an-algorithm-for-a-dog-identification-app/).



![picture](https://github.com/Rizwanhcc/Data-Scientist-Nanodegree-Udacity/blob/main/Capstone_DSND/Images/1.png)
![picture2](https://github.com/Rizwanhcc/Data-Scientist-Nanodegree-Udacity/blob/main/Capstone_DSND/Images/2.png)
![picture3](https://github.com/Rizwanhcc/Data-Scientist-Nanodegree-Udacity/blob/main/Capstone_DSND/Images/3.png)
![picture4](https://github.com/Rizwanhcc/Data-Scientist-Nanodegree-Udacity/blob/main/Capstone_DSND/Images/4.png)
![picture5](https://github.com/Rizwanhcc/Data-Scientist-Nanodegree-Udacity/blob/main/Capstone_DSND/Images/5.png)
![picture6](https://github.com/Rizwanhcc/Data-Scientist-Nanodegree-Udacity/blob/main/Capstone_DSND/Images/6.png)
![picture7](https://github.com/Rizwanhcc/Data-Scientist-Nanodegree-Udacity/blob/main/Capstone_DSND/Images/Screenshot%202021-01-11%20at%2023.51.51.png)

<a id='tree'></a>

# 5. Project Files and Folders


    .
    ├── Images                          #contains images                
    ├── saved_models                    #contains saved models                
    │   ├── weights.best.VGG16.hdf5     #weights of VGG16 model      
    │   ├── weights.best.model.hdf5     #weights of the best saved model      
    ├── dog_app.ipynb                   #contains algorithms and python codes
    ├── extract_bottleneck_features.py  #contains bottleneck features
    └── README.md



<a id='ack'></a>

# 6. Acknowledgments
I accknlwoledge the support of Udacity instructors for making this project possible. In addition, i should commend the mentor's help and reviwer's comments to improve the quality of this project. Data was provided by Udacity in the project workspace.

