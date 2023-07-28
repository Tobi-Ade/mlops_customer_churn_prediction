import datetime
import time
import pickle
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg

from prefect import task, flow
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	timestamp timestamp,
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float
)
"""
with open("../models/xgb.bin", "rb") as f_in:
	model, dmatrix = pickle.load(f_in)   
	
reference_data = pd.read_csv("./data/reference_data.csv")
reference_data.drop(columns=['Unnamed: 0'], inplace=True)
print(f"ref_columns: {reference_data.columns}")


raw_data = pd.read_csv("data/bank-customers/Churn Modeling.csv")
raw_data.drop(columns=["RowNumber", "CustomerId", "Surname", "Geography"], inplace=True)
raw_data.Gender.replace(["Female", "Male"], [0, 1], inplace=True)

target = raw_data['Exited'].values
del raw_data['Exited']

num_features = list(raw_data.columns)
num_features.remove('Gender')
cat_features = ['Gender']

# begin = datetime.datetime(2023, 08, 01, 0, 0)


column_mapping = ColumnMapping(
        numerical_features = num_features,
        categorical_features = cat_features,
        target=None,
        prediction = 'prediction',
        
)

report = Report(metrics=[
     ColumnDriftMetric(column_name='prediction'),
     DatasetDriftMetric(),
     DatasetMissingValuesMetric()
 ]
)

@task(name='prepare-database', log_prints=True)
def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=1234", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
		if len(res.fetchall()) == 0:
			conn.execute("create database test;")
		with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=1234") as conn:
			conn.execute(create_table_statement)


@task(name='calcualte-metrics', log_prints=True)
def calculate_metrics_postgresql(curr, i):
	
    current_data = raw_data.copy()
    # current_data_array = current_data.to_numpy()
    dtrain = dmatrix(current_data, target, feature_names=current_data.columns)
    prediction = model.predict(dtrain)
    current_data['prediction'] = prediction
    
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
    result = report.as_dict()

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']

    curr.execute(
		"insert into dummy_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s)",
		(datetime.datetime.now(pytz.timezone('Europe/London')), prediction_drift, num_drifted_columns, share_missing_values)
	)


@flow(name='main-flow')
def main():
	prep_db()
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
	with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=1234", autocommit=True) as conn:
		for i in range(0, 100):
			with conn.cursor() as curr:
				calculate_metrics_postgresql(curr, i)

			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("data sent")

if __name__ == '__main__':
	main()