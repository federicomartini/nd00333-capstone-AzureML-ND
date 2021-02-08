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
The results obtained after a complete run with the **AutoML** is shown below:

![](/starter_file/Screenshots/AutoML_Child_Runs_UI.png)

The **Best Model** found is takes advantage of the **RobusScaler, ExtremeRandomTrees** algorithm that reached an **accuracy** of **0.85289**.

![](/starter_file/Screenshots/AutoML_Best_Run_Details_UI.png)
![](/starter_file/Screenshots/AutoML_Best_Run_Metrics_UI.png)

And below is the screenshot of the **id** and **model name** of the best model.

![](/starter_file/Screenshots/AutoML_Best_Run_Id_Log.png)

The image below shows the **experiment** completed successfully as indicated by the **RunDetails**.

![](/starter_file/Screenshots/AutoML_RunDetails_Complete.png)

And below is a screenshot of the **best model** trained with it's parameters.
![](/starter_file/Screenshots/AutoML_Best_Run_Parameters.png)

To improve the model, we could try tweaking the **featurization** parameter in the **AutoML** configuration. At the moment, it's set to the defaul value that is *'auto'*. Instead, we could try setting it to *'FeaturizationConfig'* to specify the steps to take for the featurization without relying on the automatic system.

## Hyperparameter Tuning
The algorithm I've used is the **RandomParameterSampling** to pick up the values for each hyperparameter randomly, to avoid the risk of ending up with an experiment that takes lot of time to complete (considering the limited time applied to the VM provided by Udacity).

As for the Hyperparameters, I've used the **RandomParameterSampling** with a limited number of parameters to complete the experiment in a few minutes. I've decided to use the **Regularization Strength** to figure out how the penalization to the sum of squares affects the accuracy (values are 0.001, and 5.0), and the **maximum number of iterations** to understand if the model improves its performance with more iterations (values are 1, 50, and 150).

The primary metric is the **accuracy** to find the most accurate model in terms of predicting the **DEATH_EVENT**, and we want to **MAXIMIZE** the accuracy to select the best model. The termination policy is the **BanditPolicy** set to evaluate each interval using a **slack factor** (the ratio of the distance from the best performing run) equal to 0.1, to avoid losing time on those runs that are performing bad (too far from the best run).

### Results
The child runs obtained after running the experiment are shown in the image below.

![](/starter_file/Screenshots/HD_Child_Runs_UI.png)

The **Best Run** reached an **accuracy** of **0.8** with **C = 5** and **max_iter = 150**.

![](/starter_file/Screenshots/HD_Best_Run_Details_UI.png)
![](/starter_file/Screenshots/HD_Best_Run_Metrics_UI.png)

Lastly, the screenshot below shows the **RunDetails** log after completing the experiment successfully.

![](/starter_file/Screenshots/HD_RunDetails_Log.png)

A possible improvement could be to select the **Regularization Strength** as an **uniform** value in the range to see if it helps improve the performance of the model.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

The **best model** in terms of **accuracy** has been obtained using the **AutoML** tool, and we decided to deploy so that it can become a **web service**. Below are the steps I've taken to deploy the model:

* I've registered the best model obtained with the **AutoML**
![](/starter_file/Screenshots/AutoML_registered_model.png)

* The registered model already has an **outputs** folder that contains the **scoring_file** and **conda_env** files required to deploy the model
![](/starter_file/Screenshots/AutoML_Best_Model_Files.png)

* And below is how I've deployed the **best_model** to the **endpoint**
![](/starter_file/Screenshots/AutoML_deploy_success.png)

* We can also check the status of the **Endpoint** to make sure everything works as intended. In this case, the status is **Healthy**, that means the **service** is active.
![](/starter_file/Screenshots/AutoML_Endpoint_Healthy.png)

* Next, I've tested the **Endpoint** using code provided by Azure ML Studio in the **Endpoint** itself.
```
import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

data = {
    "data":
    [
        {
            'age': "0",
            'anaemia': "0",
            'creatinine_phosphokinase': "0",
            'diabetes': "0",
            'ejection_fraction': "0",
            'high_blood_pressure': "0",
            'platelets': "0",
            'serum_creatinine': "0",
            'serum_sodium': "0",
            'sex': "0",
            'smoking': "0",
            'time': "0",
        },
    ],
}

body = str.encode(json.dumps(data))

url = 'http://59d5671d-dd9d-4780-8fdc-154604b3202f.southcentralus.azurecontainer.io/score'
api_key = '' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(json.loads(error.read().decode("utf8", 'ignore')))
```

The **Data** submitted in the **request** are all at 0. you can change them to find out the prediction of the model when **data** passed as an input is different.

* Lastly, the answer from the server shows that we received a **True** from the model deployed to the **Endpoint**, based on the data used in the **request**

![](/starter_file/Screenshots/endpoint_answer.png)

## Screen Recording
Link: https://youtu.be/RgqXz1rPk9o

