#import libraries
import pickle
import pandas as pd

with open("../models/xgb.bin", "rb") as model_in:
        model, preprocessor = pickle.load(model_in)

def df_from_json(data):
    """
    create dataframe from data
    params: json object
    returns: pandas dataframe 
    """
    df = pd.DataFrame([data])
    return df.columns.to_list()


def prep_data(df):
    """
    performs preprocessing on dataframe 
    returns: preprocessed dataframe 
    rtype: pandas dataframe
    """
    gender = df['Gender'].str.lower().str.strip()[0]

    return gender
    

    
def apply_prep_data(df):
    
    gender = prep_data(df)
    df.Gender = [1 if gender == 'male' else 0]
    
    return df['Gender'].values[0]

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
