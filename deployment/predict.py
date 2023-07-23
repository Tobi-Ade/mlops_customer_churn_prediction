#import libraries
import pickle
import json
import pandas as pd
from flask import Flask, jsonify, request

# with open("../models/preprocessor.b", "rb") as f_in:
#         preprocessor = pickle.load(f_in)

with open("../models/xgb.bin", "rb") as model_in:
        model, preprocessor = pickle.load(model_in)

def df_from_json(data):
    """
    create dataframe from csv file
    params:csv file 
    returns: pandas dataframe 
    """
    df = pd.DataFrame([data])
    return df 


def prep_data(df):
    """
    performs preprocessing on dataframe 
    returns: preprocessed dataframe 
    rtype: pandas dataframe
    """
    gender = (str(df['Gender'].values).strip('[]').strip("''").lower())
    df['Gender'] = [1 if gender.lower()=="male" else 0]

    return df

def get_prediction(data, preprocessor, model):
        
    prepped_data = preprocessor(data)
    # reshaped_prepped_data = prepped_data.reshape(1, -1)
    prediction = model.predict(prepped_data)

    return prediction
    

app = Flask("churn-prediction")

@app.route("/predict", methods=["POST"])
def run_model():
     
     customer_details = request.get_json()

     data = df_from_json(customer_details)
    #  print(data)
     prepped_data = prep_data(data)
    #  print(f"prepped_data\n {prepped_data}")
     prediction = get_prediction(prepped_data, preprocessor, model)[0]
     churn_decision = (prediction >= 0.5)
    #  print(f"prediction\n {churn_decision}")

     result = {
            "verdict": bool(churn_decision),
     }

     return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)



