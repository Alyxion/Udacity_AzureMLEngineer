#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# ### Upload data from Microsoft's Capstone Project DAT102x
#  
# This notebook uploads US mortgage spread data as three different datasets:
# * The original data - as provided by Microsoft
# * The engineered data - contained lots of statistics attached to each row such as the average rate spread in each state and county
# * The engineered data with some removed features to make it more difficult for the model so it don't fits too much to statistics

# %%
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
# Check core SDK version number
print("SDK version:", azureml.core.VERSION)
# add common directory as module search path
common_path = os.getcwd()+"/../common"
if not common_path in sys.path:
    sys.path.append(common_path)
# %load_ext autoreload
# %autoreload 2
from ml_principal_authenticate import AzureMLAuthenticator


# %%
print("Logging into Azure ML Workspace...")
service_authenticator = AzureMLAuthenticator(config_path=os.path.normpath(f"{os.getcwd()}/../Config"))
ws = service_authenticator.get_workspace("aml_research")
if ws is not None:
    print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')
else:
    print("Workspace not available")

# %%
print("Re-combining training data and labels...")
import pandas as pd
# In the original dataset values and labels were split into separate files
# we combine them now again into a single Pandas dataframe
original_data = pd.read_csv('data/train_values.csv')
original_data_labels = pd.read_csv('data/train_labels.csv')
original_data['rate_spread'] = original_data_labels['rate_spread']
print(f"{len(original_data)} total rows")
print(original_data.sample(10))


# %%
# Upload dataset to Azure
uncleaned_dataset_name = "UncleanedMortgageSpread"
print(f"Uploading uncleaned dataset to {uncleaned_dataset_name}...")
datastore = ws.get_default_datastore()
registered_set = TabularDatasetFactory.register_pandas_dataframe(original_data, datastore, uncleaned_dataset_name)
print("Done")

# %%
print("Loading cleaned data sets...")
import zipfile
import io
dataset_zip = zipfile.ZipFile("data/cleanedEngineeredData.zip", "r")
engineered_data = pd.read_csv(io.BytesIO(dataset_zip.read("train_cleaned.csv")))


# %%
print(f"{len(engineered_data)} total rows")
print(engineered_data.columns)
print(engineered_data.sample(10))


# %%
cleaned_dataset_name = "EngineeredMortgageSpread"
print(f"Uploading cleaned dataset to {cleaned_dataset_name}")
datastore = ws.get_default_datastore()
registered_set = TabularDatasetFactory.register_pandas_dataframe(engineered_data, datastore, cleaned_dataset_name)
print("Done")


# %%
# Remove some statistics
non_lender_spread = engineered_data.copy()
non_lender_spread = non_lender_spread.drop(columns=['lender_spread', 'lender_spread_lt','lender_spread_lp', 'lender_spread_pt'])
print(f"{len(non_lender_spread)} total rows")
print(non_lender_spread.columns)
non_lender_spread.head()


# %%
reduced_engineered_set = "EngineeredMortgageSpreadNoLenderStats"
print(f"Upload cleaned dataset with reduced amount of static features to {reduced_engineered_set}")
datastore = ws.get_default_datastore()
registered_set = TabularDatasetFactory.register_pandas_dataframe(non_lender_spread, datastore, reduced_engineered_set)

