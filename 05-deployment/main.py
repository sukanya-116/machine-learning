import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = "pipeline_v1.bin"

with open(model_file, 'rb') as f_in: 
    dv, model = pickle.load(f_in)

customer1 = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

X = dv.transform([customer1])
y_pred = model.predict_proba(X)[:, 1]

print('input:', customer1)
print('output:', y_pred)

app = Flask('lead-scoring')

@app.route('/predict', methods=['POST'])
def predict_lead_scoring():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    convert = y_pred >= 0.5
    result = {
        'convert_probability': float(y_pred),
        'convert': bool(convert)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
