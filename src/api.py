from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import os
from pydantic import BaseModel

app = FastAPI(title="Fraud Detection API")

class TransactionData(BaseModel):
    v_features: list 
    amount: float

@app.get("/")
def home():
    return {"message": "API is Live"}

@app.post("/predict")
def predict(data: TransactionData):
    model_path = "models/fraud_model.pkl"
    scaler_path = "models/scaler.pkl"
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=503, detail="Model not found")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    try:
        # 1. Scale the amount
        scaled_amt = scaler.transform([[data.amount]])[0][0]
        
        # 2. Create a raw list of 29 numbers (28 V-features + 1 scaled amount)
        full_input = data.v_features + [scaled_amt]
        
        # 3. Convert to a 2D Numpy Array (The model loves this format)
        final_input = np.array([full_input])
        
        # 4. Predict using the raw array (ignoring column names)
        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0][1]

        return {
            "is_fraud": bool(prediction),
            "fraud_probability": round(float(probability), 4),
            "action": "BLOCK" if prediction else "ALLOW"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")



