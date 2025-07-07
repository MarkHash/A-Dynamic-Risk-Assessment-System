# A-Dynamic-Risk-Assessment-System
A Dynamic Risk Assessment System


# Introduction

## Purpose of this project

Imagine that you're the Chief Data Scientist at a big company that has 10,000 corporate clients. Your company is extremely concerned about attrition risk: the risk that some of their clients will exit their contracts and decrease the company's revenue.

The company needs you to create, deploy, and monitor a risk assessment ML model that will estimate the attrition risk of each of the company's 10,000 clients. Creating and deploying the model isn't the end of your work, though. You need to set up regular monitoring of your model to ensure that it remains accurate and up-to-date. You'll set up processes and scripts to re-train, re-deploy, monitor, and report on your ML model, so that your company can get risk assessments that are as accurate as possible and minimize client attrition.

# Files and instructions

## Files in this repository

- `training.py` a Python script meant to train an ML model
- `scoring.py` a Python script meant to score an ML model
- `deployment.py` a Python script meant to deploy a trained model
- `ingestion.py` a Python script meant to ingest new data
- `diagnostics.py`a Python script meant tomeasure model and data diagnostics
- `reporting.py` a Python script meant to generate reports and model metrics
- `app.py` a Python script meant to contain API endpoints
- `apicalls.py` a Python script meant to call API endpoints
- `fullprocess.py` a Python script meant to determine whether a model needs to be re-deployed, and to call all other Python scripts when needed
- `requirements.txt` a text file and records the current versions of all the modules that scripts use
- `config.json` a data file that contains names of files that will be used for configuration of ML Python scripts
- `README.md` discusses this project

## Run instructions

1. Run `app.py` to set up API ready for use
2. Run `apicalls.py` to call each API endpoints.