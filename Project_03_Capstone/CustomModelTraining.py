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
import math
import pkg_resources
import azureml.core
import scipy.stats as ss
import sklearn.decomposition as skde
from sklearn import feature_selection as fs
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
from azureml.core.run import Run
import joblib
import json
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

# +
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn import ensemble

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
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iterations = max_iterations
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


# +
global ws # The workspace

def azure_login():
    """
    Logs into the Azure ML workspace
    
    :return: The workspace
    """
    print("Authenticating to Azure, logging into AzureML workspace")
    service_authenticator = AzureMLAuthenticator(config_path=os.path.normpath(f"{os.getcwd()}/../Config"))
    ws = service_authenticator.get_workspace("aml_research")
    return ws

def load_dataset(dataset_name, sample_size=2):
    """
    Loads the dataset from an Azure ML dataset
    """
    if ws is not None:
        print("Successfuly connected to worksapce")
        print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')
    else:
        print("Workspace not available")
    print(f"Downloading dataset {dataset_name}...")
    main_test_set = dataset_name
    dataset = None
    if main_test_set in ws.datasets.keys(): 
        dataset = ws.datasets[main_test_set] 
    if dataset is not None:
        print("Success")
    df = dataset.to_pandas_dataframe()
    if sample_size!=0:
        visualize_nb_data(df.describe())
    if sample_size!=0:
        print("View sample data below")
        visualize_nb_data(df.sample(sample_size))    
    return df

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

def prepara_dataset(df, sample_size=2):
    """
    Prepares the dataset fortraining
    
    :return An dictionary containing training, validation and test sets and their labels
    """
    training_data_pd = df
    training_data_pd = training_data_pd.drop('row_id', axis=1)
    if sample_size!=0:
        visualize_nb_data(training_data_pd.sample(sample_size))    
    # Receive training data target labels
    print("Removing label from dataset, converting categorical to binary data")
    training_rate_spread = np.array(training_data_pd['rate_spread'])
    prepared_training_set = convert_categorical_columns(training_data_pd)
    features = np.array(prepared_training_set)
    labels = np.array(training_rate_spread).reshape(-1,1)
    print("Prepared training set")
    if sample_size!=0:
        visualize_nb_data(prepared_training_set.sample(sample_size))    
    scaler = preprocessing.StandardScaler().fit(features)
    features = scaler.transform(features)
    label_scaler = preprocessing.StandardScaler().fit(labels)
    labels = label_scaler.transform(labels)
    X_train, X_valid, y_train, y_valid = ms.train_test_split(features, labels, test_size=0.1, random_state=42)
    X_train, X_test, y_train, y_test = ms.train_test_split(X_train, y_train, test_size=0.1, random_state=42)    
    return {'train': [X_train, y_train], 'valid': [X_valid, y_valid], 'test': [X_test, y_test]}


# -

available_model_architectures = ['linear', 'mlpregressor', 'gradientboosting']
current_model = None


def train_models(architectures, parameters, X_train, y_train, X_test, y_test, plot=False):
    """
    Trains all models provided in architectures and returns each models' performance.
    
    :param architectures: A list of architectures to try. linear, mlpregressor or gradientboosting are valid values.
    """
    models = []
    for model_architecture in architectures:
        if model_architecture=='mlpregressor':
            model_parameters = {
                'max_iterations':parameters.iterations, 
                'hidden_layer_sizes': [4+int(round(parameters.complexity*28)),4+int(round(parameters.complexity*4))],
                'learning_rate':0.001*parameters.lrf
            }
            print("Using MLPRegressor model")
            current_model = MLPRegressorModel(**model_parameters)
        elif model_architecture=='gradientboosting':
            print("Using GradientBoosting model")
            model_parameters = {
                    'estimators':10+int(round(parameters.complexity*90)),
                    'learning_rate':0.2**parameters.lrf,
                    'max_depth':3+int(round(parameters.complexity*6))
            }            
            model_parameters['max_depth'] = min(model_parameters['max_depth'], 8)
            model_parameters['estimators'] = min(model_parameters['estimators'], 200)
            current_model = GradientBoostingModel(**model_parameters)
        else:
            model_parameters = {}
            print("Using linear regression model")
            current_model = LinearRegressionModel(**model_parameters)
        model_parameters['type'] = model_architecture
        print("Using parameters:")
        print(model_parameters)
        current_model.fit(X_train, y_train)        
        y_score = current_model.predict(X_test)
        metrics = get_regression_metrics(y_test, y_score, X_train.shape[1])
        print_metrics(y_test, y_score, X_train.shape[1])
        if check_isnotebook and plot:
            show_pred_vs_test(y_score, y_test, model_architecture)
            resid_qq(y_score, y_test, model_architecture)
        models.append({'model': current_model, 'type': model_architecture, 'metrics': metrics, 'parameters': model_parameters})
    return models


# +
import argparse

def main():
    if check_isnotebook(): # calm down argparse when testing in a notebook
        sys.argv = ['','--models=linear,mlpregressor','--complexity=0.2', '--iterations=50']
    run = Run.get_context()
    # Add arguments to script
    parser = argparse.ArgumentParser(description='Trains a machine learning model to predict the mortgage spread for different genders and ethical groups of people in different regions of the US to verify the fairness of the morgage rate computation process.')
    parser.add_argument('--lrf', type=float, default="1.0", help="The learning rate factor. 1.0 = Recommended learning rate for each model type.")
    parser.add_argument('--dataset', type=str, default="EngineeredMortgageSpread", help="The name of the registered dataset to use. EngineeredMortgageSpread by default")
    parser.add_argument('--models', type=str, default="all", help="The comma separated list of models to try. linear, mlpregressor, gradientboosting or all are valid values.")
    parser.add_argument('--iterations', type=int, default=200, help="The number of neural network training iterations")
    parser.add_argument('--complexity', type=float, default=1.0, help="The complexit of the models used. 1.0 = reasonable complex. Values should be between 0.0 and 4.0.")
    args = parser.parse_args()
    # login to asure
    global ws
    ws = azure_login()
    # load and prepare datasets
    df = load_dataset(args.dataset, sample_size=0)
    training_data = prepara_dataset(df, sample_size=0)
    # select models to train
    architectures = available_model_architectures.copy() if args.models=='all' else args.models.split(',')
    for model in architectures:
        if not model in available_model_architectures:
            print(f"Unknown model type {model}")
    # execute model training
    print(f"Training the following model types: {architectures}")
    result_models = train_models(architectures, args, *training_data['train'], *training_data['test'], plot=False)
    # find best model
    r2_scores = [cur_model['metrics']['r2_score'] for cur_model in result_models]    
    best_model_index = np.argmax(r2_scores)
    best_model_metrics = result_models[best_model_index]['metrics']
    print(f'\nTraining finished.\n\nThe best performing model is the model using the following parameters:')
    print(result_models[best_model_index]['parameters'])
    print(f'Best model performance:')
    print(best_model_metrics)
    run.log("Best model:", result_models[best_model_index]['parameters'])
    # Log results 
    print("Storing model to disk")
    if not 'outputs' in os.listdir('.'):
        os.mkdir('outputs')
    # save model and config to disk
    joblib.dump(result_models[best_model_index]['model'].model,'outputs/model.pkl') # store the model as pickl
    with open('outputs/model_training_config.json', "w") as config_file:
        config_file.write(json.dumps(result_models[best_model_index]['parameters'], indent=4, sort_keys=True))
    for key, value in best_model_metrics.items(): # log performance to Azure
        run.log(key, value)
    
if __name__ == '__main__':
    main()
# -



