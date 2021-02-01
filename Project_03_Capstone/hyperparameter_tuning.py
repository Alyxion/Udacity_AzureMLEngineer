# ### Custom model Azure HyperDrive training script
#
# This script executes up to a hundred different regression model an hyperparameter combinations to find the best one to predict the effective mortgage a US resident has to pay for a house, an apartment or a trailer. The data is based upon the Microsoft Professional for Data Science Capstone project and was used in a global contest to achieve the highest r2_score where of a score of 0.72 was required to pass the exam.
#
# The model training intelligence is stored in the file **CustomModelTraining.py** which can also be executed locally. This script stores the model training script and all it's dependencies in a single folder and then uses Azure HyperDrive to intelligently iterate through a set of hyperparameter combinations on multiple machines in parallel. Each model's outcome, so it's model and it's metrics, are then stored in an archive in the Experiment.
#
# This script then chooses the best performing model and uploads it in the Azure Workspace's model zoo so it can be used in production.

print("Executing hyper-drive training run for US Mortgage Rate Spread dataset...")

import os
import sys
import shutil
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

# ### Setup hyperdrive config

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
                            max_total_runs=100,
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

import joblib
best_run = hd_run.get_best_run_by_primary_metric()
best_run.register_model('mortgage_prediction_model', f"outputs/model.pkl")

best_run.get_metrics()

if check_isnotebook():
    from azureml.widgets import RunDetails
    RunDetails(best_run).show()    

print("Cleaning up compute...")
compute_target.delete()


