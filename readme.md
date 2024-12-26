# Flight Satisfaction
Pytorch model to predict customer flight satisfaction based on survey results.

## Data
Data was pulled from [Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction). Code expects test.csv and train.csv to be placed in the "data" directory.

## Environment
Create a Python venv and pip install from "requirements.txt".

## Data Preprocessing
Before running the model, data preprocessing is done. The following happens:
* Missing values are filled in with 0.
    * These missing values only appear in the "Arrival Delay in Minutes" column and were assumed to be 0 minutes.
* The columns 'Customer Type', 'Type of Travel', and 'Class' are turned into multiple columns with one-hot encoding.
* The 'Gender' column is turned into Gender_Female, a boolean column representing if the customer is female or not.
* The 'satisfaction' column has its values turned into 1 for "satisfied" and 0 for "neutral or dissatisfied".