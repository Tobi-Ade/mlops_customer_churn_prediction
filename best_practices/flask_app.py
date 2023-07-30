#import libraries
import pickle
import pandas as pd
from flask import Flask, jsonify, request

with open("xgb.bin", "rb") as model_in:
        model, preprocessor = pickle.load(model_in)

def df_from_json(data):
    """
    create dataframe from data
    params: json object
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
    """
    takes data, dmatrix preprocessor, and xgboost model,
    generate prediction for the data
    returns: predicition
    rtype: float
    """
        
    prepped_data = preprocessor(data)
    prediction = model.predict(prepped_data)

    return prediction[0]
    

app = Flask("churn-prediction")

@app.route("/predict", methods=["POST"])
def run_model():
     
     customer_details = request.get_json()

     data = df_from_json(customer_details)
    #  print(data)
     prepped_data = prep_data(data)
    #  print(f"prepped_data\n {prepped_data}")
     prediction = get_prediction(prepped_data, preprocessor, model)
     churn_decision = (prediction >= 0.5)
    #  print(f"prediction\n {churn_decision}")

     result = {
            "verdict": bool(churn_decision),
     }

     return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

