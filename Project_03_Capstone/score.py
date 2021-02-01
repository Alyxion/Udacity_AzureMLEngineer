# +
# based upon https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-existing-model

import os
import pickle
import json
import time
import pickle
import joblib
import pandas as pd


# Called when the deployed service starts
def init():
    global model

    # Get the path where the deployed model can be found.
    print(os.listdir(os.getenv('AZUREML_MODEL_DIR')))
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model_data')
    model = joblib.load(model_path)

# Handle requests to the service
def run(data):
    try:
        feature_str = json.loads(data)        
        feature_data = pd.read_json(feature_str['data'])
        prediction = model.predict(feature_data)
        return prediction.tolist()
    except Exception as e:
        error = str(e)
        return error

if __name__=="__main__":
    if os.path.isfile('temp/model_data'): # test if the script is executed locally
        os.environ['AZUREML_MODEL_DIR'] = 'temp'
        init()
        with open('temp/test_sample.json', 'r') as test_file:
            test_data = test_file.read()
        test_data = json.dumps({'data':test_data})
        print(run(test_data))
# -


