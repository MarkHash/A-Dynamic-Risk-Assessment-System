import os
import requests
import json

#Specify a URL that resolves to your workspace
# URL = "http://127.0.0.1:8000"
URL = "http://192.168.1.103:8000"

with open('config.json','r') as f:
    config = json.load(f)
output_model_path = os.path.join(os.getcwd(), config['output_model_path'])
test_data_path = os.path.join(os.getcwd(), config['test_data_path']) 
file_location = os.path.join(os.getcwd(), test_data_path, 'testdata.csv')
response_file_name = 'apireturns.txt'

def api_test(response_file=response_file_name):
    #Call each API endpoint and store the responses
    response1 = requests.post(URL + '/prediction?file_location=' + file_location).content
    response2 = requests.get(URL + '/scoring').content
    response3 = requests.get(URL + '/summarystats').content
    response4 = requests.get(URL + '/diagnostics').content

    #combine all API responses
    responses = [response1, response2, response3, response4]

    #write the responses to your workspace
    response_file = open(os.path.join(output_model_path, response_file), 'a')
    for element in responses:
        response_file.write(str(element) + ' ')
        response_file.write('\n')

if __name__ == '__main__':
    api_test()

