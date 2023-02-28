import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(os.getcwd(), config['output_folder_path']) 
test_data_path = os.path.join(os.getcwd(), config['test_data_path'])
output_model_path = os.path.join(os.getcwd(), config['output_model_path'])
test_file_name = 'testdata.csv'
matrix_file_name = 'confusionmatrix.png'


##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    test_data = pd.read_csv(os.path.join(test_data_path, test_file_name))
    y, preds, model = model_predictions(test_data)

    cm = confusion_matrix(y, preds, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.savefig(os.path.join(output_model_path, matrix_file_name))


if __name__ == '__main__':
    score_model()
