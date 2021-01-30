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

