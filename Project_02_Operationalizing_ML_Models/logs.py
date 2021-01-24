endpoint_name = "bank-marketing-prediction-srv"

from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.webservice import Webservice
import json

# +
# Requires the config to be downloaded first to the current working directory
with open("../Config/ml_principal.json") as config_file:
    config_data = json.loads(config_file.read())

# Authenticate via principal's credentials
svc_pr = ServicePrincipalAuthentication(
   tenant_id=config_data['tenantId'],
   service_principal_id=config_data['clientId'],
   service_principal_password=config_data['clientSecret'])

ws = Workspace.get(name="aml_research", auth=svc_pr, resource_group=config_data['resourceGroup'], subscription_id=config_data['subscriptionId'])
# -

# Set with the deployment name
name = endpoint_name

# load existing web service
service = Webservice(name=name, workspace=ws)
logs = service.get_logs()

for line in logs.split('\n'):
    print(line)


