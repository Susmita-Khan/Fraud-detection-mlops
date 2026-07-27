from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# 1. Initialize FastAPI
app = FastAPI(title="Fraud Detection API")

# 2. Define the input data structure (matches your Swagger UI test)
class TransactionInput(BaseModel):
    time: float
    v_features: list[float]  # For the V1-V28 features
    amount: float

# 3. Load the Model and Scaler 
# We use a try-except block to make sure the app doesn't crash if paths are wrong
try:
    model_path = os.path.join("models", "fraud_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Model and Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model/scaler: {e}")

# --- NEW ROOT ROUTE (Fixes the 405 Health Check Error) ---
@app.get("/")
def home():
    return {
        "message": "Fraud Detection System is Online",
        "documentation": "/docs",
        "status": "Healthy",
        "version": "1.0.0"
    }

# 4. The Prediction Endpoint
@app.post("/predict")
def predict(data: TransactionInput):
    # Combine features into a single array for the model
    # Order: [Time, V1...V28, Amount]
    features = [data.time] + data.v_features + [data.amount]
    
    # Reshape for a single prediction
    features_array = np.array(features).reshape(1, -1)
    
    # Scale the data
    scaled_features = scaler.transform(features_array)
    
    # Make Prediction
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0].tolist()
    
    status = "Fraud" if prediction == 1 else "Legitimate"
    
    return {
        "prediction": int(prediction),
        "status": status,
        "probability": probability
    }

if __name__ == "__main__":
    import uvicorn
    # Use environment variable for port (Render compatibility)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)