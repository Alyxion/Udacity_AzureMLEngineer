print("Executing custom model training...")
print("Initializing base modules")
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
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
from sklearn import ensemble
import sklearn.metrics as sklm 
import math
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
print("Using Azure ML SDK version:", azureml.core.VERSION)
# add common directory as module search path
common_path = os.getcwd()+"/../common"
if not common_path in sys.path:
    sys.path.append(common_path)
common_path = os.getcwd()
if not common_path in sys.path:
    sys.path.append(common_path)    
# %load_ext autoreload
# %autoreload 2
from ml_principal_authenticate import AzureMLAuthenticator
from notebook_check import *
from seaborn_vis import *

print("Authenticating to Azure, logging into AzureML workspace")
service_authenticator = AzureMLAuthenticator(config_path=os.path.normpath(f"{os.getcwd()}/../Config"))
ws = service_authenticator.get_workspace("aml_research")
if ws is not None:
    print("Successfuly connected to worksapce")
    print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')
else:
    print("Workspace not available")

print("Downloading dataset...")
main_test_set = "EngineeredMortgageSpread"
dataset = None
if main_test_set in ws.datasets.keys(): 
    dataset = ws.datasets[main_test_set] 
if dataset is not None:
    print("Success")
df = dataset.to_pandas_dataframe()
visualize_nb_data(df.describe())

print("View sample data below")
visualize_nb_data(df.sample(10))

import scipy.stats as ss
import sklearn.decomposition as skde
from sklearn import feature_selection as fs

training_data_pd = df
training_data_pd = training_data_pd.drop('row_id', axis=1)
visualize_nb_data(training_data_pd.head(10))


# +
def convert_categorical_columns(data_set):
    """
    Removes all columns which are not part of the test data
    """
    reduced_data_set = pd.concat([data_set[['loan_amount', 'applicant_income', 'population', 'minority_population_pct', 
                                            'ffiecmedian_family_income', 'tract_to_msa_md_income_pct', 
                                            'number_of_owner-occupied_units','number_of_1_to_4_family_units', 
                                            'co_applicant']], \
                                  data_set.loc[:,'tract_income':'income_not_prov'],
                                   data_set.loc[:,'purp_purchase':'type_fsarhs'],
                                   data_set.loc[:,'msa_spread':'income_loan_rel']
                                 ], axis=1)
    return reduced_data_set

# Receive training data target labels
print("Removing label from dataset, converting categorical to binary data")
training_rate_spread = np.array(training_data_pd['rate_spread'])
prepared_training_set = convert_categorical_columns(training_data_pd)
features = np.array(prepared_training_set)
labels = np.array(training_rate_spread).reshape(-1,1)
print("Prepared training set")
visualize_nb_data(prepared_training_set.sample(5))

# +
scaler = preprocessing.StandardScaler().fit(features)
features = scaler.transform(features)
label_scaler = preprocessing.StandardScaler().fit(labels)
labels = label_scaler.transform(labels)

X_train, X_valid, y_train, y_valid = ms.train_test_split(features, labels, test_size=0.1, random_state=42)
X_train, X_test, y_train, y_test = ms.train_test_split(X_train, y_train, test_size=0.1, random_state=42)


# -

def print_metrics(y_true, y_predicted, n_parameters):
    ## First compute R^2 and the adjusted R^2
    r2 = sklm.r2_score(y_true, y_predicted)
    r2_adj = r2 - (n_parameters - 1)/(y_true.shape[0] - n_parameters) * (1 - r2)
    
    ## Print the usual metrics and the R^2 values
    print('Mean Square Error      = ' + str(sklm.mean_squared_error(y_true, y_predicted)))
    print('Root Mean Square Error = ' + str(math.sqrt(sklm.mean_squared_error(y_true, y_predicted))))
    print('Mean Absolute Error    = ' + str(sklm.mean_absolute_error(y_true, y_predicted)))
    print('Median Absolute Error  = ' + str(sklm.median_absolute_error(y_true, y_predicted)))
    print('R^2                    = ' + str(r2))
    print('Adjusted R^2           = ' + str(r2_adj))


# +
from sklearn.neural_network import MLPRegressor

potential_models = ['LinearRegression']

class BaseModel:
    """
    Defines the model base class
    """
    def __init__(self):
        """
        Initializer
        """
        self.model = None
        
    def fit(self, x, y):
        """
        Fits the model to the training data and target labels provided
        
        :param x: The training data
        :param y: The target labels
        """
        pass
        
    def predict(self, values):
        """
        Predicts the target label for the values provided
        
        :param values: A list of values for each row and feature
        :return: The predicted values
        """
        return self.model.predict(values)


class LinearRegressionModel(BaseModel):
    """
    Implements the training interface for a linear regression model
    """
    
    def __init__(self):
        super().__init__()
    
    def fit(self, x, y):
        lin_mod = linear_model.LinearRegression()
        lin_mod.fit(x, y.ravel())
        self.model = lin_mod
    

class MLPRegressorModel(BaseModel):
    """
    Implements the training interface for a neural network
    """
    
    def __init__(self, max_iterations=500, hidden_layer_sizes = (32,8), learning_rate=0.001):
        super().__init__()
        self.hidden_layer_sizes = (32,8)
        self.max_iterations = 500
        self.learning_rate = learning_rate
    
    def fit(self, x, y):        
        nn_mod = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iterations, learning_rate_init=self.learning_rate)
        nn_mod.fit(x, y.ravel())
        self.model = nn_mod
        
        
class GradientBoostingModel(BaseModel):
    """
    Implements the training interface for a gradient boosting model
    """
    
    def __init__(self, estimators = 100, learning_rate = 0.2, max_depth=6):
        super().__init__()
        self.estimators = estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
    
    def fit(self, x, y):        
        params = {'n_estimators': self.estimators, 'max_depth': self.max_depth, 'min_samples_split': 2,
                  'learning_rate': self.learning_rate, 'loss': 'ls'}
        clf = ensemble.GradientBoostingRegressor(**params)
        clf.fit(x, y.ravel())
        self.model = clf


# -

available_model_architectures = ['linear', 'mlpregressor', 'gradientboosting']
current_model = None

for model_architecture in available_model_architectures:
    if model_architecture=='mlpregressor':
        print("Using MLPRegressor model")
        current_model = MLPRegressorModel()
    elif model_architecture=='gradientboosting':
        print("Using GradientBoosting model")
        current_model = GradientBoostingModel()
    else:
        print("Using linear regression model")
        current_model = LinearRegressionModel()    

    current_model.fit(X_train, y_train)        
    y_score = current_model.predict(X_test)
    print_metrics(y_test, y_score, X_train.shape[1])

    if check_isnotebook:
        show_pred_vs_test(y_score, y_test, model_architecture)
        resid_qq(y_score, y_test, model_architecture)

# +
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException

amlcompute_cluster_name = "cpucluster"

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=amlcompute_cluster_name)
except:
    pass

compute_target.update(min_nodes=4)
compute_target.wait_for_completion(show_output=True, min_node_count = 1, timeout_in_minutes = 10)
compute_target.update(min_nodes=0)
# -


