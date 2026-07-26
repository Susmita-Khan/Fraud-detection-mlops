import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# 1. Initialize FastAPI
app = FastAPI(title="Fraud Detection API")

# 2. Define the path to models
MODEL_PATH = "models/fraud_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# 3. Load the model and scaler
model = None
scaler = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ Model and Scaler loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model files: {e}")
else:
    print("⚠️ Error: Model files not found. Ensure 'models/' folder is present.")

# 4. Input structure (30 features total: 1 Time + 28 V-features + 1 Amount)
class Transaction(BaseModel):
    time: float
    v_features: List[float]
    amount: float

@app.get("/")
def home():
    return {"message": "Fraud Detection API is active"}

@app.post("/predict")
def predict(data: Transaction):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")

    try:
        # Validate that we have exactly 28 V-features
        if len(data.v_features) != 28:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected 28 V-features, but got {len(data.v_features)}"
            )

        # Reconstruct the 30-column input: [Time, V1...V28, Amount]
        input_list = [data.time] + data.v_features + [data.amount]
        features_array = np.array(input_list).reshape(1, -1)

        # Scale and Predict
        scaled_features = scaler.transform(features_array)
        prediction = model.predict(scaled_features)
        prediction_int = int(prediction[0])
        
        # Get probability if available (XGBoost supports this)
        prob = "N/A"
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(scaled_features).tolist()[0]

        return {
            "prediction": prediction_int,
            "status": "Fraud" if prediction_int == 1 else "Legit",
            "probability": prob
        }

    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))