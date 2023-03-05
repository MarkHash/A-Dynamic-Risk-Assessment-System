from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import subprocess

##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

output_folder_path = os.path.join(os.getcwd(), config['output_folder_path']) 
prod_deployment_path = os.path.join(os.getcwd(), config['prod_deployment_path'])
model_path = os.path.join(os.getcwd(), config['output_model_path']) 
model_file_name = 'trainedmodel.pkl'
score_file_name = 'latestscore.txt'
step_record_name = 'ingestedfiles.txt'

####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    
    res1 = subprocess.run(['cp', os.path.join(model_path, model_file_name), os.path.join(prod_deployment_path, model_file_name)], capture_output=True)
    res2 = subprocess.run(['cp', os.path.join(model_path, score_file_name), os.path.join(prod_deployment_path, score_file_name)], capture_output=True)
    res3 = subprocess.run(['cp', os.path.join(output_folder_path, step_record_name), os.path.join(prod_deployment_path, step_record_name)], capture_output=True)
    # print(res1)
    # print(res2)
    # print(res3)    

if __name__ == '__main__':
    store_model_into_pickle()
