
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = os.path.join(os.getcwd(), config['input_folder_path'])
output_folder_path = os.path.join(os.getcwd(), config['output_folder_path'])

output_file_name = 'finaldata.csv'
step_record_name = 'ingestedfiles.txt'

#############Function for data ingestion
def merge_multiple_dataframe(data_path=input_folder_path):
    """
    Process the data in multiple CSV files and merge into one CSV file
    """
    #check for datasets, compile them together, and write to an output file
    if os.path.isfile(os.path.join(output_folder_path, output_file_name)):
        merged_df = pd.read_csv(os.path.join(output_folder_path, output_file_name))
    else:
        merged_df = pd.DataFrame()

    file_list = os.listdir(data_path)
    for f in file_list:
        #check if the file extension is '.csv'
        if os.path.splitext(f)[1].lower() == '.csv':
            #read CSV file into dataframe and merge into one
            tmp_df = pd.read_csv(os.path.join(data_path, f))
            merged_df = pd.concat([merged_df, tmp_df], ignore_index=True)

            #record the time and file of ingestion
            dateTimeObj = datetime.now()
            time_now = str(dateTimeObj.year) + '/' + str(dateTimeObj.month) + '/' + str(dateTimeObj.day)
            all_records = [data_path, f, len(tmp_df.index), time_now]
            step_record_file = open(os.path.join(output_folder_path, step_record_name), 'a')
            for element in all_records:
                step_record_file.write(str(element) + ' ')
            step_record_file.write('\n')

    #remove duplicate record and export to csv file
    merged_df = merged_df.drop_duplicates()
    merged_df.to_csv(os.path.join(output_folder_path, output_file_name))

if __name__ == '__main__':
    merge_multiple_dataframe()
