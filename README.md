# 💳 Fraud Detection MLOps Project

This project is a production-ready Machine Learning API that detects fraudulent credit card transactions using XGBoost and FastAPI.

## 🚀 Live Demo
The API is deployed on Render and is accessible here:
**[Live API Link](https://fraud-detection-mlops-ypz3.onrender.com/docs)**
*(Note: Use the `/docs` endpoint to access the interactive Swagger UI)*

## 🛠️ Tech Stack
- **Backend:** FastAPI (Python)
- **Machine Learning:** Scikit-learn, XGBoost
- **Containerization:** Docker
- **Deployment:** Render
- **CI/CD:** GitHub Integration

## 📋 How to Use the API
1. Open the [Live Link](https://fraud-detection-mlops-ypz3.onrender.com/docs).
2. Click on the **POST /predict** endpoint.
3. Click **"Try it out"**.
4. Enter transaction data (Time, V1-V28 features, and Amount).
5. Click **"Execute"** to get the fraud prediction.

## 📁 Project Structure
- `models/`: Contains the trained `.pkl` model and scaler.
- `src/`: Source code for data processing and API.
- `main.py`: Main entry point for the FastAPI application.
- `Dockerfile`: Configuration for containerizing the app.
