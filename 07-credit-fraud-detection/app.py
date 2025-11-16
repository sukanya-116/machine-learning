from fastapi import FastAPI
import xgboost as xgb
import pandas as pd
import joblib
import uvicorn

# Load model & scaler
model = xgb.Booster()
model.load_model("credit_fraud_detect_model.json")
scaler = joblib.load("scaler.pkl")

# Same feature order used during training
features = [
    "time", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
    "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
    "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
    "v28", "amount"
]

app = FastAPI()

@app.post("/predict")
def predict(data: dict):

    # Convert dict → dataframe
    df = pd.DataFrame([data])

    # Reorder columns exactly as trained
    df = df[features]

    # Scale input
    df_scaled = scaler.transform(df)

    # XGBoost DMatrix
    dmatrix = xgb.DMatrix(df_scaled, feature_names=features)

    pred = model.predict(dmatrix)[0]

    # If probability >= 0.5, classify as fraud
    fraud_transaction = pred >= 0.5

    result = {
        'probability': float(pred),
        'risk': bool(fraud_transaction)
    }

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
