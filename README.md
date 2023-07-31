# Bank Customer Churn Prediction - An MLOps Project 

![Bank Customers](https://trainingindustry.com/content/uploads/2021/03/Customer-Service-Training-for-Your-Bank-4.5.21-928x522.jpg)

## Table of Contents 
1. [Problem Definition](#problem-defintion)
2. [Project Outline](#project-outline)
3. [How to Run the Project](#how-to-run-the-project)
4. [References](#References)
5. [Contact Me](#contact-me)

## Problem Definiton 
It is a common occurrence for bank customers to default to other banks. This project involves exoploring the data of bank customers and try to establish churning patterns, and then use the knowledge gained to predict churn for other customers. Then we turn our solution into a full mlops project to automate and continually optimize the entire process <br>

More information on the data used for this project can be found [here](https://www.kaggle.com/datasets/santoshd3/bank-customers)

## Project Outline
- [Notebooks](https://github.com/Tobi-Ade/mlops_customer_churn_prediction/tree/main/notebooks) <br> 
Here we create notebooks where we clean and preprocess the data, before training our ml models. We train different models and select the best performing model. <br>
We spend minimal time building the model so as to focus on the major goal of the project (MlOps). 

- [Mlflow and Model Registry](https://github.com/Tobi-Ade/mlops_customer_churn_prediction/blob/main/notebooks/churn_prediction_mlflow.ipynb) <br>
We use Mlflow and Model Registry to track the performance of our models. This gives us a structured way to monitor and store our artifacts (models, preprocessors, etc).
We access our stored artifacts multiple times to train new models.

- [Workflow Orchestration with Prefect](https://github.com/Tobi-Ade/mlops_customer_churn_prediction/tree/main/workflow-orchestration) <br>
Prefect allows us to set a defined structure for our project. We can deploy our project and use work queues to track our deployments and run them whenever we wanr as prefect  flows. <br>
The prefect ui also provides an interface to see our flow runs and logs from every run.

- [Deployment](https://github.com/Tobi-Ade/mlops_customer_churn_prediction/tree/main/deployment) <br>
Here we deploy our model as a web service using flask. We also use docker o add an eextra layer of isolation for the web service. We build a version of the service using our artifacts stored in model registry.

- [Monitoring with Evidently, Grafana, Adminer]() <br>
We use evidently to track how model performance by defining metrics on our data. Adminer to manage por postgres database where store our metrics from evidently. <br>
Then we use Grafana to pull these data metrics and create dashboards to make monitoring these metrics easier.

- [Best-Practices]() <br>
Here we create tests for our scripts to make sure the result of every run is exactly what we intended. We use Pytest, DeepDiff, and Make to run and automate the tests.


## How to Run the Project
- After pulling the repo into your preferred directory, install all the necessary dependencies from the requirements file by running:
```bash
pip install -r requirements.txt
```
- You can now automatically run the tests and the flask service by running:
```bash
make
```
- Now the flask web service is running and you can send a request by running the sample request in the scripts-test directory: <br>
    first cd into the directory by running:
```bash 
cd scripts-test
```

- Now run the sample-request module by running: <br>
```bash
python sample-request.py
```

## References 
- [Data Source](https://www.kaggle.com/datasets/santoshd3/bank-customers)
- [Mlops Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)
- [Readme Image](https://trainingindustry.com/content/uploads/2021/03/Customer-Service-Training-for-Your-Bank-4.5.21-928x522.jpg)


## Contact Me 
 [<img src="https://img.shields.io/badge/tobi-ade-000000?style=flat-square&logo=github&logoColor=white" />](https://github.com/Tobi-Ade) [<img src="https://img.shields.io/badge/gabriel-adeleke-0A66C2?style=flat-square&logo=linkedin&logoColor=white" />](https://www.linkedin.com/in/gabriel-adeleke/) [<img src="https://img.shields.io/badge/Gmail-EA4335?style=flat-square&logo=Gmail&logoColor=white" />](mailto:themarveloustobi@gmail.com)

  

