import requests


url = 'http://localhost:9696/predict'

customer_id = 'xyz-123'

customer = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}
response = requests.post(url, json=customer).json()
print(response)

if response['convert'] == True:
    print('Customer %s will subscribe' % customer_id)
else:
    print('Customer %s will not subscribe' % customer_id)
