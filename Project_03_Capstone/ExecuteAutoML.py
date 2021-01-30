from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException
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

# +
service_authenticator = AzureMLAuthenticator(config_path=os.path.normpath(f"{os.getcwd()}/../Config"))

ws = service_authenticator.get_workspace("aml_research")
if ws is not None:
    print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')
else:
    print("Workspace not available")

# +
amlcompute_cluster_name = "lprio-cpu-clst"

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=amlcompute_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS12_V2',# for GPU, use "STANDARD_NC6"
                                                           vm_priority = 'lowpriority',
                                                           max_nodes=5)
    compute_target = ComputeTarget.create(ws, amlcompute_cluster_name, compute_config)

compute_target.update(min_nodes=5, max_nodes=5, idle_seconds_before_scaledown=600)
compute_target.wait_for_completion(show_output=True, min_node_count = 5, timeout_in_minutes = 10)

# +
main_test_set = "EngineeredMortgageSpread"  # TODO Export

# Try to load the dataset from the Workspace. Otherwise, create it from the file
# NOTE: update the key to match the dataset name
found = False

dataset = None
if main_test_set in ws.datasets.keys(): 
    found = True
    dataset = ws.datasets[main_test_set] 

df = dataset.to_pandas_dataframe()
df.describe()
# -

dataset.take(5).to_pandas_dataframe()

experiment_name = 'AzureMLCapstoneExperiment'
project_folder = './pipeline-project'
experiment = Experiment(ws, experiment_name)
experiment

# +
automl_settings = {
    "experiment_timeout_minutes": 180,
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

pipeline_run = experiment.submit(pipeline)

# +
from notebook_check import check_isnotebook

if check_isnotebook():
    from azureml.widgets import RunDetails
    RunDetails(pipeline_run).show()
# -

pipeline_run.wait_for_completion()

pipeline_run.wait_for_completion()

metrics_output = pipeline_run.get_pipeline_output(metrics_output_name)
num_file_downloaded = metrics_output.download('.', show_progress=True)

# +
# Download and visualize the performance of the single models tried

# +
import json
with open(metrics_output._path_on_datastore) as f:
    metrics_output_result = f.read()
    
deserialized_metrics_output = json.loads(metrics_output_result)
df = pd.DataFrame(deserialized_metrics_output)
df
# -

# Retrieve best model from Pipeline Run
best_model_output = pipeline_run.get_pipeline_output(best_model_output_name)
num_file_downloaded = best_model_output.download('.', show_progress=True)

# +
import pickle

with open(best_model_output._path_on_datastore, "rb" ) as f:
    best_model = pickle.load(f)
best_model
# -

import sklearn
print(sklearn.__version__)


