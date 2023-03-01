from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from datetime import datetime

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(os.getcwd(), config['output_model_path']) 
test_data_path = os.path.join(os.getcwd(), config['test_data_path']) 
model_file_name = 'trainedmodel.pkl'
score_file_name = 'latestscore.txt'

#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    file_list = os.listdir(test_data_path)
    for file in file_list:
        #check if the file extension is '.csv'
        if os.path.splitext(file)[1].lower() == '.csv':
            test_data = pd.read_csv(os.path.join(test_data_path, file))

    with open(os.path.join(model_path, model_file_name), 'rb')  as file:
        model = pickle.load(file)

    X = test_data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']].values.reshape(-1, 3)
    y = test_data['exited'].values.reshape(-1, 1).ravel()
    preds = model.predict(X)
    f1score = metrics.f1_score(preds, y)

    #record the time and file of ingestion
    dateTimeObj = datetime.now()
    time_now = str(dateTimeObj.year) + '/' + str(dateTimeObj.month) + '/' + str(dateTimeObj.day)
    all_records = [time_now, f1score]
    test_score_file = open(os.path.join(model_path, score_file_name), 'a')
    for element in all_records:
        test_score_file.write(str(element) + ' ')
    test_score_file.write('\n')
    return f1score
    
if __name__ == '__main__':
    f1score = score_model()
