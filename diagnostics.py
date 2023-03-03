
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
from datetime import datetime
from sklearn import metrics
import subprocess


##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(os.getcwd(), config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(os.getcwd(), config['prod_deployment_path'])
model_file_name = 'trainedmodel.pkl'
output_file_name = 'finaldata.csv'
test_file_name = 'testdata.csv'

##################Function to get model predictions
def model_predictions(dataset):
    #read the deployed model and a test dataset, calculate predictions
    X = dataset[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)
    y = dataset['exited'].values.reshape(-1, 1).ravel()
    with open(os.path.join(prod_deployment_path, model_file_name), 'rb')  as file:
        model = pickle.load(file)

    preds = model.predict(X)
    return y, preds, model

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    training_data = pd.read_csv(os.path.join(dataset_csv_path, output_file_name))
    means = training_data.mean
    medians = training_data.median
    stds = training_data.std
    
    return [means, medians, stds]

##################Function to get missing data
def missing_data():
    #check missing data
    training_data = pd.read_csv(os.path.join(dataset_csv_path, output_file_name))
    nas = list(training_data.isna().sum())
    na_percents = [nas[i] / len(training_data.index) for i in range(len(nas))]
    return na_percents


##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    start_time = timeit.default_timer()
    os.system('python ingestion.py')
    ingestion_time = timeit.default_timer() - start_time
    start_time = timeit.default_timer()
    os.system('python training.py')
    training_time = timeit.default_timer() - start_time

    return [ingestion_time, training_time]

##################Function to check dependencies
def outdated_packages_list():
    #get a list of dependencies
    installed = subprocess.check_output(['pip', 'list', '-o'])
    # print(installed)
    return installed


if __name__ == '__main__':
    test_data = pd.read_csv(os.path.join(test_data_path, test_file_name))
    y, preds, model = model_predictions(test_data)
    statistics_summary = dataframe_summary()
    times = execution_time()
    installed = outdated_packages_list()
