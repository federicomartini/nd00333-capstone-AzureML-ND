# Machine Learning Engineer with Microsoft Azure Nanodegree - Capstone Project

In this project, I've created two models: one using **Automated ML** (denoted as AutoML from now on) and one customized model whose hyperparameters are tuned using **HyperDrive**. Then, I've compared the performance of both the models and deployed the best performing model.

This project to demonstrate how to use an external dataset in a workspace, train a model using the different tools available in the AzureML framework as well as how to deploy the model as a **web service**.

The data to study comes from an external source that is the **Heart Failure Prediction dataset** (from Kaggle) and the goal is to build a classification model to predict if data leads to a death event or not.

## Project Set Up and Installation

## Dataset

### Overview

The data source is available at this link: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure. 

Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

### Task
The main goal is to create a model to assess the likelihood of a death by heart failure event. This can be used to help hospitals in assessing the severity of patients with cardiovascular diseases. The features I'll use to predict the **death event** are:
* age
* anaemia
* creatinine_phosphokinase
* diabetes
* ejection_fraction
* high_blood_pressure
* platelets
* serum_creatinine
* serum_sodium
* sex
* smoking
* time


### Access
*TODO*: Explain how you are accessing the data in your workspace.
Data are loaded from a **csv** file into the **Azure ML Studio** environment and registered as a **Dataset**.

![](/Screenshots/dataset.png)

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
Link: https://youtu.be/RgqXz1rPk9o

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
