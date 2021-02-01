# ### Automated ML
#
# This script setups and executes an Azure AutoML pipeline to let AzureML intelligently find the best regression algorithm and hyperparameter combination to predict the mortgage rate in a pre-engineered dataset we provided it.
#
# To do so we already provisioned the data in form of an AzureML dataset named **EngineeredMortgageSpread** using the script ProvisionDataSets.py to make it available for the training cluster. All we still need to do then is to define the target column, rescrictions in form of time limits, the primary metrics and the amount of compute power we want to provide and then AutoML basically executes the complete training process for us.
#
# After the process has finished it tells us the best run and it's metrics so we could right afterwards forward it e.g. for regression tests.

# TODO: Import Dependencies. In the cell below, import all the dependencies that you will need to complete the project.

import logging
import os
import sys
import csv
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
import pkg_resources
import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
from azureml.core.dataset import Dataset
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.pipeline.steps import AutoMLStep
from azureml.pipeline.core import Pipeline
from azureml.pipeline.core import PipelineData, TrainingOutput
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.webservice import AciWebservice
# Check core SDK version number
print("SDK version:", azureml.core.VERSION)
# add common directory as module search path
common_path = os.getcwd()+"/../common"
if not common_path in sys.path:
    sys.path.append(common_path)
# %load_ext autoreload
# %autoreload 2
from ml_principal_authenticate import AzureMLAuthenticator
from notebook_check import *
from seaborn_vis import *

# +
# configuration
experiment_timeout = 120 # Best result achieved after 65 minutes

main_test_set = "EngineeredMortgageSpread"
unclean_test_set = "UncleanedMortgageSpread"
import argparse
if check_isnotebook(): # calm down argparse when testing in a notebook
    sys.argv = ['']
parser = argparse.ArgumentParser()
parser.add_argument('--unclean', type=bool, default=False, help="Defines if the uncleaned, unengineered data set shall be used")
args = parser.parse_args()
if args.unclean:
    used_data_set = unclean_test_set
else:
    used_data_set = main_test_set
print(f"Using dataset {used_data_set}...")
# -

# ### Log into AzureML Workspace

# +
print("Connecting to AzureML Workspace...")
config_path = os.path.normpath(f"{os.getcwd()}/../Config")
service_authenticator = AzureMLAuthenticator(config_path=config_path)

ws = service_authenticator.get_workspace("aml_research")
if ws is not None:
    print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')
else:
    print("Workspace not available")
# -

# ### Set up training cluster

# +
print("Setting up compute cluster...")

amlcompute_cluster_name = "tmplpriocluster"

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=amlcompute_cluster_name)
    compute_target.update(min_nodes=5, max_nodes=5, idle_seconds_before_scaledown=600)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS12_V2',# for GPU, use "STANDARD_NC6"
                                                           vm_priority = 'lowpriority',
                                                           min_nodes=5,
                                                           max_nodes=5)
    compute_target = ComputeTarget.create(ws, amlcompute_cluster_name, compute_config)
compute_target.wait_for_completion(show_output=True, min_node_count = 5, timeout_in_minutes = 10)
# -

# ### Dataset
#
# ### Overview
#
# I am using the mortgage loan rate spread data set which was till 2019 used in Microsoft's Data Science Professional program. The goal of this project was to especially analyze the rate spread in the US between different groups of people such as male and female, ethical and religous groups to verify that there is **no discrimation** involved in the computation of the effective loan rate they were given. The dataset contains real information about Americans and the **rate_spread** to the current base interest rate defined at that point of time in state which is our target label in the experiments.
#
# A very detailed overview of the data you can find here:
# https://github.com/Alyxion/MPPDataScience/blob/master/MPP_DS_FinalReport.pdf
#
# I chose this dataset because it it's especially "dirty", this means that it contains large amounts of categorical data, missing values, outliers etc.. You can find the original data here:
# https://github.com/Alyxion/Udacity_AzureMLEngineer/tree/main/Project_03_Capstone/data
#
# As the original data is basically not usable by a classic algorithm I already curated it in the original project which you can find here:
#
# In this project both, the Azure ML as the HyperDrive approach, both use the curated data I named "engineered". This data has already been stored in the Azure ML Datasets using the **ProvisionDataSets.py** script you can find in this notebook's directory.
#
#
# TODO: Get data. In the cell below, write code to access the data you will be using in this project. Remember that the dataset needs to be external.

dataset = None
if used_data_set in ws.datasets.keys(): 
    print("Dataset found, downloading it...")
    dataset = ws.datasets[used_data_set]
df = dataset.to_pandas_dataframe()
visualize_nb_data(df.describe())

visualize_nb_data(dataset.take(5).to_pandas_dataframe())

# ### Setup experiment

experiment_name = 'AzureMLCapstoneExperiment'
if args.unclean:
    experiment_name = 'AzureMLCapstoneExperimentUnclean'
project_folder = './pipeline-project'
experiment = Experiment(ws, experiment_name)
if check_isnotebook():
    display(experiment)

# ### AutoML configuration
#
# TODO: Explain why you chose the automl settings and configuration you used below.
#
# The data we want to predict is the **rate_spread**, so basically the offset to the base interest rate in the US at this point of time. As the **rate_spread** is a continuous value our task is a **regression task**. As primary metric I chose the r2_score - the reason for this is that this was also the goal of a contest related to Microsoft's MPP project in the past on the one hand and I wanted a comparable result as outcome. Independent of this the r2 score is very well suited to regression tasks which large data variance such as ours. Further details you can find here: https://en.wikipedia.org/wiki/Coefficient_of_determination
#
# As our datasets has more than 70 columns and more than 100,000 lines of data and the problem itself is quite difficult I had to choose a quite large timeout of 120 minutes. Depending on the vm size it also needs at least 70 minutes to find the best algorithm. To minimize the waiting time and to use maximum use of the up to 20 low priority vCPUs cores available in an MSDN subscription I defined a maximum of 5 concurrent iterations.

# +
automl_settings = {
    "experiment_timeout_minutes": experiment_timeout,
    "max_concurrent_iterations": 5,
    "primary_metric" : 'r2_score'
}
automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "regression",
                             training_data=dataset,
                             label_column_name="rate_spread",   
                             path = project_folder,
                             enable_early_stopping= True,                             
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings
                            )
ds = ws.get_default_datastore()
metrics_output_name = 'metrics_output'
best_model_output_name = 'best_model_output'

metrics_data = PipelineData(name='metrics_data',
                           datastore=ds,
                           pipeline_output_name=metrics_output_name,
                           training_output=TrainingOutput(type='Metrics'))
model_data = PipelineData(name='model_data',
                           datastore=ds,
                           pipeline_output_name=best_model_output_name,
                           training_output=TrainingOutput(type='Model'))
automl_step = AutoMLStep(
    name='automl_module',
    automl_config=automl_config,
    outputs=[metrics_data, model_data],
    allow_reuse=True)
pipeline = Pipeline(
    description="pipeline_with_automlstep",
    workspace=ws,    
    steps=[automl_step])
# -

# ### Execute pipeline and wait for it to finish

pipeline_run = experiment.submit(pipeline)

# ### Run Details
#
# TODO: In the cell below, use the RunDetails widget to show the different experiments.

if check_isnotebook():
    from azureml.widgets import RunDetails
    RunDetails(pipeline_run).show()

# ### Clean up computing resources and test the model

print("Waiting for ML run to finish execution...")
pipeline_run.wait_for_completion()
print("Deleting computing resources...")
compute_target.delete()

metrics_output = pipeline_run.get_pipeline_output(metrics_output_name)
num_file_downloaded = metrics_output.download('.', show_progress=True)

# +
import json
with open(metrics_output._path_on_datastore) as f:
    metrics_output_result = f.read()
    
deserialized_metrics_output = json.loads(metrics_output_result)
metrics = pd.DataFrame(deserialized_metrics_output)
visualize_nb_data(metrics)
# -

# ### Best Model
#
# TODO: In the cell below, get the best model from the automl experiments and display all the properties of the model.

# Retrieve best model from Pipeline Run
best_model_output = pipeline_run.get_pipeline_output(best_model_output_name)
num_file_downloaded = best_model_output.download('.', show_progress=True)

# +
import pickle

with open(best_model_output._path_on_datastore, "rb" ) as f:
    best_model = pickle.load(f)
visualize_nb_data(best_model)
# -

visualize_nb_data(best_model.steps)

print(len(df))
print("Selecting sample")
df_test = df.sample(50)
print("Done")
y_test = df_test['rate_spread']
df_test = df_test.drop(['rate_spread'], axis=1)
X_test = df_test
print("Executing prediction")
y_pred = best_model.predict(X_test)
print("Predictions:")
print(y_pred)
print("Real values:")
print(y_test.tolist())
if seaborn_available:
    resid_qq(y_pred, np.array(y_test.tolist()), 'Best algorithm')

# +
import shutil
try:
    os.mkdir("temp")
except:
    pass

try:
    os.mkdir("aml_model")
except:
    pass

try:
    os.mkdir("temp/outputs")
except:
    pass
df_test.to_json("temp/test_sample.json")
shutil.copy(best_model_output._path_on_datastore, "aml_model/model_data")
# -

# ### Model Deployment
#
# Remember you have to deploy only one of the two models you trained.. Perform the steps in the rest of this notebook only if you wish to deploy this model.
#
# TODO: In the cell below, register the model, create an inference config and deploy the model as a web service.

# +
from azureml.core import Model
from azureml.core.resource_configuration import ResourceConfiguration
import sklearn

model = Model.register(workspace=ws,
                      model_name='mortgage_rate_automl_model',                # Name of the registered model in your workspace.
                      model_path=best_model_output._path_on_datastore)        # Local file to upload and register as a model.
# -

# ### Configure Python environment with all modules required to execute AutoML pickles

# +
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment

# Create the environment
myenv = Environment(name="mortgage_score_env_aml")
conda_dep = CondaDependencies()

# Define the packages needed by the model and scripts
conda_dep.add_conda_package("numpy")
conda_dep.add_conda_package("pip")
conda_dep.add_conda_package("scikit-learn=0.22.2.post1") # required for AutoML
# conda_dep.add_conda_package("scikit-learn=0.20.3")  # required for HyperDrive trained model
# You must list azureml-defaults as a pip dependency
conda_dep.add_pip_package("azureml-defaults==1.11.0")
conda_dep.add_pip_package("azureml-core")
conda_dep.add_pip_package("azureml-automl-runtime")
conda_dep.add_pip_package("packaging")
conda_dep.add_pip_package("azureml-explain-model==1.11.0")
conda_dep.add_pip_package("inference-schema")
conda_dep.add_conda_package("numpy")
# scikit-learn>=0.19.0,<=0.20.3
conda_dep.add_conda_package("pandas")
conda_dep.add_conda_package("py-xgboost")
# Save environment also locally to disk so we can test the score script directly by creating a local environment
conda_dep.save('temp/mortgage_score_env.yml')
myenv.python.conda_dependencies = conda_dep
# -

# ### Setup web service and inference configuration
#
# We are deploying the local script score.py as entry point and the environment defined above. Also we enabled authentication, even if our deployment is just for a short while.

webservice_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1, auth_enabled=True)
inference_config = InferenceConfig(entry_script='score.py', environment=myenv)

# ### Test in local docker container
#
# For faster pre-evaluation (as a web deployment can take up to 10 minutes) I test the script locally (which usually just takes 20 seconds) before deploying it to the web.

df_test.head(10)

partial_data = df_test[0:50].to_json()
run_data = json.dumps({'data':partial_data})
print(run_data)

print("\nTesting inference using local docker container before deploying it as web service\n")
from azureml.core.webservice import LocalWebservice
# This is optional, if not provided Docker will choose a random unused port.
deployment_config = LocalWebservice.deploy_configuration(port=6789)
local_service = Model.deploy(ws, "local-mortgage-service-test", [model], inference_config, deployment_config)
local_service.wait_for_deployment()
result = local_service.run(run_data)
print("Inference result")
print(result)
print("Success" if len(result)==len(df_test) else "Failed")
local_service.delete()

print("Deploying web inference service...")
web_service = model.deploy(workspace=ws, name="mortgage-service", models=[model], inference_config=inference_config,
    deployment_config=webservice_config, overwrite=True)
web_service.wait_for_deployment(show_output=True)
web_service.update(enable_app_insights=True)

# TODO: In the cell below, send a request to the web service you deployed to test it.

print("\n\nTesting web service via WebService class interface...")
result = web_service.run(run_data)
print("Inference result")
print(result)
print("Success" if len(result)==len(df_test) else "Failed")

if seaborn_available:
    resid_qq(np.array(result), np.array((df[0:50]).loc[:, 'rate_spread'].tolist()), 'Best algorithm')

print(f"Inference url is {web_service.scoring_uri}")

import requests
scoring_uri = web_service.scoring_uri
print(f"\n\nTesting web service directly via requests module. Calling URL {scoring_uri}...")
primary_key = web_service.get_keys()[0]
# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {primary_key}'
result = json.loads(requests.post(scoring_uri, headers=headers, data=run_data).text)
print("Inference result")
print(result)
print("Success" if len(result)==50 else "Failed")

# TODO: In the cell below, print the logs of the web service and delete the service

import time
logs = web_service.get_logs()
print(logs)
print("Cleaning up and deleting web service...")
web_service.delete()
print("Done")


