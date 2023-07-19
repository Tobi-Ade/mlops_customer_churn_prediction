#import libraries
import pickle
import mlflow
import pandas as pd
import xgboost as xgb
from prefect import flow, task
from xgboost import DMatrix as dmatrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


@task(retries=2, retry_delay_seconds=0.2)
def read_data(file_path):
    """
    create dataframe from csv file
    params:csv file 
    returns: pandas dataframe 
    """
    df = pd.read_csv(file_path)
    return df 


@task(retries=2, retry_delay_seconds=0.2)
def get_data_splits(df):
    """
    splits dataframe into sets for modelling
    params: pandas dataframe
    returns: split data arrays
    rtype: numpy arrays
    """
    #specifying data splits
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    # df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

    y_full_train = df_full_train['Exited'].values
    # y_train = df_train['Exited'].values
    # y_val = df_val['Exited'].values
    y_test = df_test['Exited'].values

    del df_full_train['Exited']
    # del df_train['Exited']
    # del df_val['Exited']
    del df_test['Exited']

    #converting data splits into arrays 
    X_full_train = df_full_train.to_numpy()
    # X_train = df_train.to_numpy()
    # X_val = df_val.to_numpy()
    X_test = df_test.to_numpy()

    return X_full_train, y_full_train, X_test, y_test

@task(retries=2, retry_delay_seconds=0.2, log_prints=True)
def train_xgb_model(X_train, X_test, y_train, y_test):
    """
    this function trains an xgboost model on input arrays
    params: train and test features and targets
    returns: roc_auc_score 
    rtype: float
    """
    with mlflow.start_run():

        xgb_params = {
            'eta': 0.3,
            'max_depth': 10,
            'min_child_weight': 1,

            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'nthread': 8,
            'seed': 1
        }
        
        mlflow.log_param("data", "../data/bank-customers/Churn Modeling.csv")
        mlflow.log_params(xgb_params)


        d_full_train = dmatrix(X_train, label=y_train)
        dtest = dmatrix(X_test, label=y_test)
        
        xgb_clf = xgb.train(xgb_params, d_full_train)
        y_pred_xgb = xgb_clf.predict(dtest)

        xgb_score = roc_auc_score(y_test, y_pred_xgb)

        mlflow.log_metric("roc_auc_score", xgb_score)

        print(xgb_score)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dmatrix, f_out)
        
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(xgb_clf, artifact_path="model_artifact")
        mlflow.end_run()
        return xgb_score

@flow(name="churn_pred")
def main_flow(csv_path):

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("mlops-project")
    
    df = read_data(csv_path)
    df.Gender.replace(["Female", "Male"], [0, 1], inplace=True)
    df.drop(columns=["RowNumber", "CustomerId", "Surname", "Geography"], inplace=True)

    X_train, y_train, X_test, y_test = get_data_splits(df)
    train_xgb_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main_flow(csv_path="data/bank-customers/Churn Modeling.csv")



