import json
import os
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.workspace import Workspace

class AzureMLAuthenticator:
    """
    Helps accessing an Azure ML workspace using a Service Principal
    
    For more details see https://docs.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication
    """
    
    CONFIG_FILENAME = "ml_principal.json"
    CONFIG_TEMPLATE_FILENAME = "_ml_principal.json"
    
    def __init__(self, config_path):
        """
        Initializer
        
        :param config_path: The path where the ml_principal.json is located
        """
        self.config_filename = config_filename = f"{config_path}/{self.CONFIG_FILENAME}"
        self.workspace = None
        self.valid_config = False
        
        if os.path.isfile(config_filename):
            with open(config_filename, "r") as config_file:
                self.config_data = json.loads(config_file.read())
        else:
            print(f"Service principal configuration file not found at {config_filename} - creating file from template.\nPlease store the service principal's credentials associcated to your Azure ML Workspace.")
            template_config_filename = f"{config_path}/{self.CONFIG_TEMPLATE_FILENAME}"
            with open(config_filename, "w") as config_file:
                with open(template_config_filename, "r") as template_config_file:
                    data = template_config_file.read()
                    config_file.write(data)
                    self.config_data = json.loads(data)
            self.config_data = None
        self.verify_config()
            
    def verify_config(self):
        """
        Verifies the configuration
        """
        if self.config_data is None:
            return False
        any_error = False
        for key, value in self.config_data.items():
            if 'YOUR_' in value or len(value)==0:
                print(f"Please configure the field {key} in your configuration data")
                any_error = True
        if any_error:
            print(f"Configuration file {self.config_filename} not correctly set up yet. Please update the incorrect fields.")
            return False
        self.valid_config = True
        return True
    

    def get_workspace(self, workspace_name):
        """
        Authenticates and returns the AzureML workspace handle
        
        :param workspace_name: The workspace's name
        :return The workspace handle. None on failure
        """
        if not self.valid_config:
            print("Please configure your Service Principal before trying to access the workspace")
            print("\nVisit this URL for further details about setting up a Service Principal in your own subscription:")
            print("https://docs.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication")
            return None
        # Authenticate via principal's credentials
        config_data = self.config_data
        svc_pr = ServicePrincipalAuthentication(
           tenant_id=config_data['tenantId'],
           service_principal_id=config_data['clientId'],
           service_principal_password=config_data['clientSecret'])        
        workspaces = [key for key in Workspace.list(config_data['subscriptionId']).keys()]
        if workspace_name not in workspaces:
            print(f"Workspace {workspace_name} not found in workspace list. Please create the workspace or adjust the scripts accordingly.")
            print("\nFollowing workspaces could be found:")
            for workspace in workspaces:
                print(f"- {workspace}")
            return None
        if self.workspace==None:
            self.workspace = Workspace.get(name=workspace_name, auth=svc_pr, resource_group=config_data['resourceGroup'], subscription_id=config_data['subscriptionId'])        
        return self.workspace


