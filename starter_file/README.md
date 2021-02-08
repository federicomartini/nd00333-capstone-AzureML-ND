# Machine Learning Engineer with Microsoft Azure Nanodegree - Capstone Project

![](/starter_file/Screenshots/capstone-diagram.png)

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
Data are loaded from a **csv** file into the **Azure ML Studio** environment and registered as a **Dataset**.

![](/starter_file/Screenshots/dataset.png)

## Automated ML
This is a **classification** problem, because we have to predict a **DEATH_EVENT** which can be **True** or **False**, so the task has been set to *"classification"*. To evaluate the performance of the model, I've set the **accuracy** as the **primary metric**. Further, to avoid running out of time, I've set the **timeout** for the experimento to 30 minutes. Lastly, I've decided to use two folds for **cross-validation**, and a maximum number of **concurrent iteration** of 3 take advantage of the compute instance setup.

```
automl_settings = {
    "experiment_timeout_minutes": 30,
    "task": 'classification',
    "primary_metric": 'accuracy',
    "max_concurrent_iterations": 3,
    "training_data": ds,
    "label_column_name": "DEATH_EVENT",
    "n_cross_validations": 2
}

# TODO: Put your automl config here
automl_config = AutoMLConfig(**automl_settings)
```

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
The results obtained after a complete run with the **AutoML** is shown below:

![](/starter_file/Screenshots/AutoML_Child_Runs_UI.png)

The **Best Model** found is takes advantage of the **RobusScaler, ExtremeRandomTrees** algorithm that reached an **accuracy** of **0.85289**.

![](/starter_file/Screenshots/AutoML_Best_Run_Details_UI.png)
![](/starter_file/Screenshots/AutoML_Best_Run_Metrics_UI.png)

And below is the screenshot of the **id** and **model name** of the best model.

![](/starter_file/Screenshots/AutoML_Best_Run_Id_Log.png)

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

The image below shows the **experiment** completed successfully as indicated by the **RunDetails**.

![](/starter_file/Screenshots/AutoML_RunDetails_Complete.png)

And below is a screenshot of the **best model** trained with it's parameters.
![](/starter_file/Screenshots/AutoML_Best_Run_Parameters.png)

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
