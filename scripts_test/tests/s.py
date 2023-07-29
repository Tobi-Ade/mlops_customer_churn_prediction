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

gender = df['Gender'] = df['Gender'].str.lower().str.strip()[0]
print(gender)

# print(gender == 'male')
# print(str(df['Gender'].values).strip('[]').strip("''").lower())
df['Gender'] = df['Gender'].str.lower().str.strip()

print(df['Gender'].values[0])

