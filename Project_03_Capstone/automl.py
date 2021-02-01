# ### Azure AutoML training script
#
# This script setups and executes an Azure AutoML pipeline to let AzureML intelligently find the best regression algorithm and hyperparameter combination to predict the mortgage rate in a pre-engineered dataset we provided it.
#
# To do so we already provisioned the data in form of an AzureML dataset named **EngineeredMortgageSpread** using the script ProvisionDataSets.py to make it available for the training cluster. All we still need to do then is to define the target column, rescrictions in form of time limits, the primary metrics and the amount of compute power we want to provide and then AutoML basically executes the complete training process for us.
#
# After the process has finished it tells us the best run and it's metrics so we could right afterwards forward it e.g. for regression tests.

# +
# configuration
experiment_timeout = 180 # Best result achieved after 65 minutes

main_test_set = "EngineeredMortgageSpread"
unclean_test_set = "UncleanedMortgageSpread"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--unclean', type=bool, default=False, help="Defines if the uncleaned, unengineered data set shall be used")
args = parser.parse_args()
if args.unclean:
    used_data_set = unclean_test_set
else:
    used_data_set = main_test_set
print(f"Using dataset {used_data_set}...")
# -

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

dataset = None
if used_data_set in ws.datasets.keys(): 
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

# ### Setup AutoML pipeline

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
df = pd.DataFrame(deserialized_metrics_output)
visualize_nb_data(df)
# -

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

print("Preparing test set...")
df_test = dataset.to_pandas_dataframe()
df_test = df_test[pd.notnull(df_test['rate_spread'])]
y_test = df_test['rate_spread']
X_test = df_test.drop(['rate_spread'], axis=1)

ypred = best_model.predict(X_test)
