# Disaster Response Pipeline Project
This project is completed as a partial requirement for the UDACITY’s Data Scientist nanodegree program.

# Table of Contents


### 1. [Project Motivation](#motivation)
### 2. [Installations](#installations)
### 3. [Project Overview](#overview)
### 4. [Web App](#webapp)
### 5. [Project Folders and Files](#tree)
### 6. [Instructions](#instructions)
### 7. [Acknowledgments](#ack)

<a id='motivation'></a>
# 1. Project Motivation

This project is quite unique and interesting in a way that it includes all the aspects that a data scientist does in real world. The objective of this project is deploying a text classification web app by using data provided by Figure Eight. This web app should classify texts with respect to disaster situation. By entering a text in the app, a user could get the classification of that input text. This project provides a unique opportunity to learn and get your hands-on various data sciences processes under one umbrella. It involves ETL Extract, Transform and Load workflow, ML Machine Learning Workflow and finally deploying the model on a web-app. This web_app classifies disaster text messages, it can help speed up the process of accessing victims. The concerned authority would quickly assess which type of need victims need and which team/department should be alrerted as a quick response.

<a id='installations'></a>

# 2. Installations

In order to run the project files properly, you need to install folowing python libraries.

* pandas
* re
* sys
* sklearn
* nltk
* sqlalchemy
* pickle
* Flask
* sqlite3


<a id='overview'></a>

# 3. Project Overview

This project mainly contains three steps.

## 1.	ETL Pipeline

It begins with creating a data pipeline that includes Extract, Transform and Load (ETL), create a SQL table and saving in SQL database. 

## 2.	Machine Learning Pipeline

In the next step it prepares a Machine Learning pipeline (ML) by extracting the same data saved in the previous step in SQL database. In Machine Learning part, it loads data, apply Natural Language Processing (NLP) steps, build Machine Learning model, fine tune model and then evaluate model. Finally, store the model in a pickle file.

## 3.	Flask Web App

In the final step of this project, we have to build a Flask web app where a user enters the disaster message and gets the message classifications based on the entered text.

<a id='webapp'></a>

### 4. Web App images

![picture](https://github.com/Rizwanhcc/Data-Scientist-Nanodegree-Udacity/blob/main/Disaster_Response_Project/Images/result.png)
![picture2](https://github.com/Rizwanhcc/Data-Scientist-Nanodegree-Udacity/blob/main/Disaster_Response_Project/Images/one.png)
![picture3](https://github.com/Rizwanhcc/Data-Scientist-Nanodegree-Udacity/blob/main/Disaster_Response_Project/Images/two.png)
![picture4](https://github.com/Rizwanhcc/Data-Scientist-Nanodegree-Udacity/blob/main/Disaster_Response_Project/Images/three.png)

<a id='tree'></a>

# 5.	Project Files and Folders


    .
    ├── app     
    │   ├── run.py                      #web app python file                  
    │   └── templates   
    │       ├── go.html                 #classification results page of the web app              
    │       └── master.html             #main page of the web app              
    ├── data                   
    │   ├── disaster_categories.csv     #dataset containing all categories        
    │   ├── disaster_messages.csv       #dataset containing all messages     
    │   └── process_data.py             #clean data in python file     
    ├── models
    │   └── train_classifier.py         #train ML model in python file                  
    └── README.md

*Note: pickel file could not be uploaded due to it's file size*

<a id='instructions'></a>

# 6. Instructions:
Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python 

<code>data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db</code>

To run ML pipeline that trains classifier and saves 

<code>python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl</code>

Run the following command in the home's directory to run your web app. 

<code>python app/run.py</code>

Run the following command to generate the view web app link

<code> env | grep WORK </code>

Go to http://0.0.0.0:3001/
 
 or 

Go to: https://view6914b2f4-3001.udacity-student-workspaces.com/

<a id='ack'></a>

# 7. Acknowledgments
I accknlwoledge the support of Udacity instructors for making this project possible. In addition, i should comend the mentors help me to improve the quality of this project. Data was provided by Udacity and Figure Eight.

