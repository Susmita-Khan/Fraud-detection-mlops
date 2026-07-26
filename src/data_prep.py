import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def prepare_data(data_path="data/creditcard.csv"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing dataset at {data_path}. Please upload it.")

    df = pd.read_csv(data_path, encoding='ISO-8859-1')
    
    # Real-world step: Standardizing the 'Amount' column
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    
    # Dropping original columns that aren't needed (Time and raw Amount)
    X = df.drop(['Time', 'Amount', 'Class'], axis=1)
    y = df['Class']
    
    # Stratified split ensures fraud cases are in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save the scaler so the API can use it to process new transactions
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_train, X_test, y_train, y_test

