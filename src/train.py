import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, accuracy_score

def train_model(X_train, X_test, y_train, y_test):
    """
    Trains an optimized XGBoost model with SMOTETomek balancing.
    """
    # 1. Handle Class Imbalance (Fixes the low 59% accuracy)
    print("🔄 Starting SMOTETomek... (Generating synthetic fraud cases)")
    smt = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smt.fit_resample(X_train, y_train)
   
    # 2. Scale the data (RobustScaler handles outliers better than Standard)
    print("📏 Scaling data with RobustScaler...")
    scaler = RobustScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test)
   
    # 3. Setup MLflow tracking
    if not os.path.exists("mlruns"):
        os.makedirs("mlruns")
    mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
    mlflow.set_experiment("Fraud_Detection_System")

    with mlflow.start_run():
        print("🚀 Training Optimized XGBoost Model...")
        # HYPERPARAMETERS: Optimized to push accuracy > 70%
        model = XGBClassifier(
            n_estimators=500,        
            max_depth=10,            
            learning_rate=0.03,      
            scale_pos_weight=1,      # Must be 1 because SMOTE already balanced the data
            subsample=0.9,           
            colsample_bytree=0.9,    
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False
        )
       
        model.fit(X_train_res, y_train_res)
       
        # 4. Evaluation
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds)
        rec = recall_score(y_test, preds)
        acc = accuracy_score(y_test, preds)
       
        # 5. Log Results to Dashboard
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("recall", rec)
        mlflow.xgboost.log_model(model, "fraud_model")
       
        # 6. Save files for Deployment (FastAPI will use these)
        if not os.path.exists("models"):
            os.makedirs("models")
           
        joblib.dump(model, "models/fraud_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")
       
        print("\n" + "="*30)
        print("✅ TRAINING SUCCESSFUL!")
        print(f"📊 Accuracy: {round(acc, 4) * 100}%")
        print(f"📊 F1 Score: {round(f1, 4)}")
        print(f"📊 Recall:   {round(rec, 4)}")
        print("="*30)

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # Path to your dataset
    # Check your 'data' folder—if your CSV is named differently, change it here!
    DATA_PATH = "data/creditcard.csv" 
    
    if os.path.exists(DATA_PATH):
        print(f"📂 Loading dataset from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)
        
        # 1. Prepare Features (X) and Target (y)
        # Assuming 'Class' is the target (0 = Legit, 1 = Fraud)
        if 'Class' in df.columns:
            X = df.drop('Class', axis=1)
            y = df['Class']
            
            # 2. Split into Train/Test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 3. Call the training function
            train_model(X_train, X_test, y_train, y_test)
        else:
            print("❌ Error: Target column 'Class' not found in CSV.")
    else:
        print(f"❌ Error: Could not find file at {DATA_PATH}")
        print("Make sure your dataset is inside the 'data' folder.")
        