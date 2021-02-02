# # Hyperparameter tuning using Hyperdrive
#
# This script executes up to a hundred different regression model an hyperparameter combinations to find the best one to predict the effective mortgage a US resident has to pay for a house, an apartment or a trailer. The data is based upon the Microsoft Professional for Data Science Capstone project and was used in a global contest to achieve the highest r2_score where of a score of 0.72 was required to pass the exam.
#
# The model training intelligence is stored in the file **CustomModelTraining.py** which can also be executed locally. This script stores the model training script and all it's dependencies in a single folder and then uses Azure HyperDrive to intelligently iterate through a set of hyperparameter combinations on multiple machines in parallel. Each model's outcome, so it's model and it's metrics, are then stored in an archive in the Experiment.
#
# This script then chooses the best performing model and uploads it in the Azure Workspace's model zoo so it can be used in production.

print("Executing hyper-drive training run for US Mortgage Rate Spread dataset...")

# TODO: Import Dependencies. In the cell below, import all the dependencies that you will need to complete the project.

import os
import sys
import shutil
import json
from azureml.core import Model
from azureml.core.resource_configuration import ResourceConfiguration
from azureml.core.webservice import LocalWebservice
import azureml
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Experiment
from azureml.widgets import RunDetails
from azureml.train.sklearn import SKLearn
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.policy import BanditPolicy
from azureml.train.hyperdrive.sampling import BayesianParameterSampling
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.parameter_expressions import uniform, choice
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

# ### Log into Azure ML Workspace

# +
print("Connecting to AzureML Workspace...")
service_authenticator = AzureMLAuthenticator(config_path=os.path.normpath(f"{os.getcwd()}/../Config"))

ws = service_authenticator.get_workspace("aml_research")
if ws is not None:
    print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')
else:
    print("Workspace not available")
# -

# ### Dataset:
#
# TODO: Get data. In the cell below, write code to access the data you will be using in this project. Remember that the dataset needs to be external.
#
# Note that the dataset itself is provisioned as AzureML dataset and will directly be fetched from the training script so the code below is only required to fulfull this TODO and to verify the endpoint at the end of this notebook. The data is provisioned in ProvisionDataSets.py.

dataset = None
used_data_set = "EngineeredMortgageSpread"
if used_data_set in ws.datasets.keys(): 
    print("Dataset found, downloading it...")
    dataset = ws.datasets[used_data_set]
df = dataset.to_pandas_dataframe()
visualize_nb_data(df.describe())

# ### Define script components

script_dependencies = ["../common/ml_principal_authenticate.py", "../common/notebook_check.py", 
                       "../common/seaborn_vis.py", "../Config/ml_principal.json"]
base_directory = os.getcwd()
script_file = "CustomModelTraining.py"
script_path = "training_script"
local_test_dir = f"{os.getcwd()}/local_training_script"
local_script_dir = f"{os.getcwd()}/{script_path}"
print(f"Training scripts will be stored in {local_script_dir}")
print(f"Local test run script will be stored in {local_test_dir}")

# ### Wind up compute cluster for hyper drive training execution

# +
amlcompute_cluster_name = "tmplphdcluster"

print(f"Setting up compute cluster... {amlcompute_cluster_name}")

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

# ### Assemble training scripts and test script locally

# +
def provide_script_files_in_directory(target_dir):
    """
    Collects all files required for the remote training script execution in the local directory defined
    
    :param target_dir: The directory in which the script files shall be collected
    """
    try:
        shutil.rmtree(target_dir)
    except:
        pass
    os.mkdir(target_dir)
    print(f"Storing training script {script_file} in {target_dir}...")
    shutil.copy(script_file, f"{target_dir}/{script_file}")
    for dependency in script_dependencies:
        print(f"Storing dependency {dependency}...")
        shutil.copy(f"{base_directory}/{dependency}", f"{target_dir}/{os.path.basename(dependency)}")
    # create place holder for training configuration, is used to tell the script it is packaged
    with open(f"{target_dir}/hd_training_run_config.json", "w") as training_run_file:
        pass
    print("Done")    
    
provide_script_files_in_directory(local_test_dir)
# -

# ### Test script locally before executing it in parallel on HyperDrive

import subprocess
from subprocess import Popen, PIPE
os.chdir(local_test_dir)
current_python_environment = sys.executable
p = Popen([current_python_environment, script_file, "--models==linear, mlpregressor", "--complexity=0.3"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
output, err = p.communicate(b"input data that is passed to subprocess' stdin")
rc = p.returncode
os.chdir(base_directory)
print(output.decode("utf-8") )
if rc!=0:
    print("An error occured:")
    print(err.decode("utf-8") )

# ### Prepare scripts for containerization

provide_script_files_in_directory(local_script_dir)

# ### Hyperdrive Configuration
#
# TODO: Explain the model you are using and the reason for chosing the different hyperparameters, termination policy and config settings.
#
# My training script supports overall 3 methods as of now - a simple polynomial based linear regression ("--model==linear"), a neural network ("--model==mlpregressor") and a gradient boosting using a set of estimators ("--model==gradientboosting). Both the neural network and the gradient boosting are quite strong methods for such complicated problems as ours. The reason for this is that they can very effectively adjust to data such as ours where the regression's outcome is influences by many binary (originally categorical) values.
#
# I am a using a Bayesian sampling optimization to make the best use of the time, assumming it will quickly detect the strength of the boosting ensembling with quite high complexity grades. 
#
# In addition I am iterating through several different, reasonable learning rates, **lrf**. As the (effective) learning rates for the neural network and the boosting algorithms vary strongly I am providing this hyperparameter as factor to a "reasonable" base value defined in the script itself. The effective values chosen will be stored in the output pickle files for reproducability.
#
# Also I am iterating through different **complexity** grades. In case of the neural network they define the width of the hidden neuron layers, in case of the boosting variant they define the depth and count of estimators.
#
# And last but not least - though this value only affects the neural network - I am trying two different **iterations** counts, 100, 200. About 100 runs are the minimum needed to nearly converge, more runs just still slightly improve the model.
#
# After several tries a max_total_runs value of **50** turned out to be a good compromise, the best result is usually achieved after about 30 runs. Our target metric is - as in the AutoML variant - the r2_score is this was originally also the goal for of the contest this dataset has been used to and a strong indicator for a dataset such as this with a quite huge variance within the label data.

# +
# Specify parameter sampler, usnig Baysesian sampling to quickly choose the most promising combinations
ps = BayesianParameterSampling( {
        "--model": choice('linear', 'mlpregressor', 'gradientboosting'),
        "--lrf": choice(1.0, 0.1, 0.25, 0.5, 2.0),
        "--iterations": choice(100, 200),
        "--complexity": choice(1.0, 0.25, 0.5, 2.0)
    })

# Create a SKLearn estimator for use with train.py
est = SKLearn(source_directory=script_path, entry_script=script_file, compute_target=compute_target)
# Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy.
hyperdrive_config = HyperDriveConfig(estimator=est, hyperparameter_sampling=ps,
                            policy=None, primary_metric_name="r2_score",
                            primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                            max_total_runs=50,
                            max_concurrent_runs=5)
# -

# ### Setup experiment and submit run

experiment_name = 'AzureMLCapstoneExperiment_HyperDrive'
experiment = Experiment(ws, experiment_name)
if check_isnotebook():
    display(experiment)

hd_run = experiment.submit(hyperdrive_config)
if check_isnotebook():
    RunDetails(hd_run).show()

# ### Wait for completition and archive the best performing model in our model zoo

hd_run.wait_for_completion(show_output=True)

# ### Run Details
#
# OPTIONAL: Write about the different models trained and their performance. Why do you think some models did better than others?
#
# TODO: In the cell below, use the RunDetails widget to show the different experiments.

best_run = hd_run.get_best_run_by_primary_metric()
if check_isnotebook():
    from azureml.widgets import RunDetails
    RunDetails(best_run).show()    

print("Cleaning up compute...")
compute_target.delete()

# ### Best Model
#
# TODO: In the cell below, get the best model from the hyperdrive experiments and display all the properties of the model.

import joblib
model = best_run.register_model('mortgage_hd_prediction_model', f"outputs/model.pkl")

model.download('hd_model')

best_run.get_metrics()

# #### Prepare test data

df_test = df.sample(50)
df_test = df_test.drop(['rate_spread'], axis=1)
df_test.head()

# ### Model Deployment
#
# Remember you have to deploy only one of the two models you trained.. Perform the steps in the rest of this notebook only if you wish to deploy this model.
#
# TODO: In the cell below, register the model, create an inference config and deploy the model as a web service.
#
#

# +
# Prepare environment config
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
if not 'temp' in os.listdir():
    os.mkdir('temp')
df_test.to_json("temp/test_sample.json") # save data for external tests

# Create the environment
myenv = Environment(name="mortgage_score_env_hd")
conda_dep = CondaDependencies()

# Define the packages needed by the model and scripts
conda_dep.add_conda_package("numpy")
conda_dep.add_conda_package("pip")
conda_dep.add_conda_package("scikit-learn=0.20.3")
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

webservice_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1, auth_enabled=True)
inference_config = InferenceConfig(entry_script='score.py', environment=myenv)

# ### Test in local docker container

print("\nTesting inference using local docker container before deploying it as web service\n")
# This is optional, if not provided Docker will choose a random unused port.
deployment_config = LocalWebservice.deploy_configuration(port=6789)
local_service = Model.deploy(ws, "local-mortgage-service-test-hd", [model], inference_config, deployment_config)
local_service.wait_for_deployment()
single_row_data = df_test[0:10].to_json()
run_data = json.dumps({'data':single_row_data})
result = local_service.run(run_data)
print("Inference result")
print(result)
print("Success" if len(result)==10 else "Failed")
local_service.delete()

# ### Deploy real, externally accessable web service

print("Deploying web inference service...")
web_service = model.deploy(workspace=ws, name="mortgage-service-hd", models=[model], inference_config=inference_config,
    deployment_config=webservice_config, overwrite=True)
web_service.wait_for_deployment(show_output=True)
web_service.update(enable_app_insights=True)

# TODO: In the cell below, send a request to the web service you deployed to test it.

print("\n\nTesting web service via WebService class interface...")
result = web_service.run(run_data)
print("Inference result")
print(result)
print("Success" if len(result)==10 else "Failed")

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
print("Success" if len(result)==10 else "Failed")

# TODO: In the cell below, print the logs of the web service and delete the service

import time
logs = web_service.get_logs()
print(logs)
print("Cleaning up and deleting web service...")
time.sleep(120.0)  # wait a moment - otherwise we will receive an error as the deployment isn't yet finished 100%
web_service.delete()
print("Done")


