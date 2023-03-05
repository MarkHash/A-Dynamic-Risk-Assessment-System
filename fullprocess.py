
import os
import json
import requests
from training import train_model
from scoring import score_model
from ingestion import merge_multiple_dataframe
from deployment import store_model_into_pickle
import diagnostics
from reporting import report_model
from apicalls import api_test

with open('config.json', 'r') as f:
    config = json.load(f)

prod_deployment_path = os.path.join(os.getcwd(), config['prod_deployment_path'])
input_folder_path = os.path.join(os.getcwd(), config['input_folder_path'])
output_folder_path = os.path.join(os.getcwd(), config['output_folder_path'])
prod_deployment_path = os.path.join(os.getcwd(), config['prod_deployment_path'])
step_record_name = 'ingestedfiles.txt'
score_file_name = 'latestscore.txt'
model_file_name = 'trainedmodel.pkl'
X_columns = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']


##################Check and read new data
#first, read ingestedfiles.txt
list_of_list = []
with open(os.path.join(prod_deployment_path, step_record_name), 'r') as f:
    step_records = f.read().split('\n')
    ingested_files = [line.split(' ')[1] for line in step_records if line]
    # print(ingested_files)

f.close()

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
input_file = os.listdir(input_folder_path)
new_files = [value for value in input_file if value not in ingested_files and os.path.splitext(value)[1].lower() == '.csv']
print(new_files)

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if new_files:
    print("continue")
    #ingest new dataset
    merge_multiple_dataframe(input_folder_path)
    #read latest score
    with open(os.path.join(prod_deployment_path, score_file_name), 'r') as f:
        score_records = f.read().split('\n')
        scores = [float(line.split(' ')[1]) for line in score_records if line]
        latest_score = max(scores)
        print("latest: " + str(latest_score))

    new_score = score_model(trained_model_path=prod_deployment_path, data_path=input_folder_path, model_file=model_file_name)
    print("new: " + str(new_score))

    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    if  latest_score > new_score:
        print("model drift")
        ##################Deciding whether to proceed, part 2
        #if you found model drift, you should proceed. otherwise, do end the process here
        train_model(dataset_path=output_folder_path)
        ##################Re-deployment
        #if you found evidence for model drift, re-run the deployment.py script
        store_model_into_pickle()
        ##################Diagnostics and reporting
        #run diagnostics.py and reporting.py for the re-deployed model
report_model(matrix_file='confusionmatrix2.png')
api_test(response_file='apireturns2.txt')





