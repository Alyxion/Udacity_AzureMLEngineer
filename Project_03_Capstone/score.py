# +
# based upon https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-existing-model

import os
import pickle
import json
import time
import pickle
import joblib
import pandas as pd

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

# Called when the deployed service starts
def init():
    global model
    global is_custom_model

    # Get the path where the deployed model can be found.
    model_list = os.listdir(os.getenv('AZUREML_MODEL_DIR'))
    print(model_list)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), model_list[0])
    model = joblib.load(model_path)
    is_custom_model = isinstance(model, dict) and 'trainingParameters' in model # check if it's a custom model or an AutoML one

# Handle requests to the service
def run(data):
    try:
        feature_str = json.loads(data)        
        feature_data = pd.read_json(feature_str['data'])
        
        if not is_custom_model: # AutoML model
            prediction = model.predict(feature_data)
        else: # custom model
            test_df = feature_data.drop(columns=['row_id'])
            converted = convert_categorical_columns(test_df)
            transformed_set = model['scaler'].transform(converted)
            prediction = model['model'].predict(transformed_set)
            prediction = model['labelScaler'].inverse_transform(prediction)
            
        return prediction.tolist()
    except Exception as e:
        error = str(e)
        return error

if __name__=="__main__":
    any_model = False
    if os.path.isfile('hd_model/model.pkl'):
        os.environ['AZUREML_MODEL_DIR'] = 'hd_model'
        any_model = True
    if os.path.isfile('aml_model/model_data'): # test if the script is executed locally
        os.environ['AZUREML_MODEL_DIR'] = 'aml_model'
        any_model = True
    if any_model:
        init()
        with open('temp/test_sample.json', 'r') as test_file:
            test_data = test_file.read()
        test_data = json.dumps({'data':test_data})
        print(run(test_data))
# -


