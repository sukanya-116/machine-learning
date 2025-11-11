import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np

# --- 1. Model Loading ---
model_file = "pipeline_v1.bin"
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)
    print(f"Model and DictVectorizer loaded successfully from {model_file}")


# --- 2. Define Input Schema using Pydantic ---
# This ensures the request body adheres to the required data types and fields.
class LeadInput(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# --- 3. Initialize FastAPI App ---
app = FastAPI(
    title="Lead Scoring Service",
    description="Predicts the probability of a lead converting based on user attributes.",
    version="1.0.0"
)

# --- 4. Define Prediction Endpoint ---
@app.post("/predict")
async def predict_lead_scoring(customer: LeadInput):
    """
    Predicts the conversion probability and binary outcome for a single customer.
    """
    if dv is None or model is None:
        return {"error": "Model not loaded. Check server logs."}

    # Convert the Pydantic model instance to a dictionary for DictVectorizer
    customer_dict = customer.model_dump()

    # Transformation and Prediction logic 
    X = dv.transform([customer_dict])
    # model.predict_proba returns an array, we take the probability of the positive class (index 1)
    y_pred = model.predict_proba(X)[:, 1][0] # Access the single prediction probability

    # Calculate binary prediction
    convert = y_pred >= 0.5

    # Format the result
    result = {
        'convert_probability': float(y_pred),
        'convert': bool(convert)
    }

    return result

# --- 5. Run the Application with Uvicorn ---
if __name__ == "__main__":
    # In production, you would run 'uvicorn predict-fastapi:app --host 0.0.0.0 --port 9696'
    uvicorn.run(app, host="0.0.0.0", port=9696)