## Preparing the environment

- Create an Azure Machine Learning workspace named **"aml_research"**. A step by step guide can be found here: 
  - https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?tabs=azure-portal#create-a-workspace
- Setup a **Service Principal** and make notes of the **clientId, clientSecret, tenantId, subScriptionId** and the **resource group**'s name:
  - https://docs.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication
- Execute the first two cells of the **ProvisionDataSets** notebook. It will complain about a missing **Config/ml_principal.json** and automatically create it for you.
  - Fill in these values accordingly in the file path it shows to you in the notebook
    `"clientId": "f898dc57-2c20-485e-96f5-XXXXXXXXXX",`
    `"clientSecret": "n~Ip4XXXXXX12gCwJXXXXXX_sxuED338xl",`
    `"subscriptionId": "3cd9cbbe-XXXX-4315-a11d-47eed87a8547",`
    `"tenantId": "12b8031c-fe22-XXXX-a54f-43dc40076af1",`
    `"resourceGroup": "aml_research"`
- Create an Azure Compute Instance of type Standard_DS3_V2
- Open Jupyter Notebook
  - Open a terminal in the default AzureML 3.6 environment
  - Install seaborn via "pip install seaborn"



## Preparing the scripts and data

- Fetch this repo from GitHub via `git clone https://github.com/Alyxion/Udacity_AzureMLEngineer`
- Enter the directory **Project_03_Capstone**
- Execute the Python script **ProvisionDataSets.py** via **python ProvisionDataSets.py**
  - This script will create 3 different versions of the same dataset named
    - UncleanedMortgageSpread - The original dataset once provided for the Microsoft Professional Program for Data Science.
    - EngineeredMortgageSpread - The original dataset enhanced with lots of row-wise statistics such as the average rate spread in each county, state etc.
    - EngineeredMortgageSpreadNoLenderSpread - With slightly reduced statistics.
  - All sets can be found in the directory data in form of CSV files or here:
    https://github.com/Alyxion/Udacity_AzureMLEngineer/tree/main/Project_03_Capstone/data
- Execute the Python script **ExecuteAutoML.py** via `python ExecuteAutoML.py` in the AzureML 3.6 environment
  - The notebook will run for around 90 minutes
  - The notebook assumes you have a quota of 20 low priority standard vCPUs remaining
- Alternatively you can execute the script via `python ExecuteAutoML.py --unclean=1` to verify the performance AutoML would achieve without any human help / data engineering.