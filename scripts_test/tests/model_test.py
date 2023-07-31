import sys
sys.path.insert(0, 'mlops-project/mlops_customer_churn_prediction/scripts_test/')

import test_flask

import pandas as pd

data = {
        'CreditScore': 619,
        'Gender': 'Male',
        'Age': 42,
        'Tenure': 2,
        'Balance': 0.00,
        'NumOfProducts': 1,
        'HasCrCard': 1,
        'IsActiveMember': 1,
        'EstimatedSalary': 101348.88
    }


df = pd.DataFrame([data])

def test_df_from_json():
    
    actual_result = test_flask.df_from_json(data) 

    expected_result = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                       'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    
    assert actual_result == expected_result

def test_prep_data():
    
    actual_result = test_flask.prep_data(df)

    expected_result = 'male'

    assert actual_result == expected_result


def test_apply_prep_data():

    actual_result = test_flask.apply_prep_data(df)

    expected_result = 1

    assert actual_result == expected_result
    
