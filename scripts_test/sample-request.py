import requests

url = 'http://127.0.0.1:9696/predict'

customer_details = {
    
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

response = requests.post(url, json=customer_details).json()

if response == True:
    print("customer will quit")
else:
    print("customer will not quit")

