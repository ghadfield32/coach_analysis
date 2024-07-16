import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Replace NA values with 0
    df = df.fillna(0)
    
    # Select relevant features
    features = ['PTS', 'VORP', 'GS', 'AST', 'TRB', '3P', 'STL', 'Age', 'BPM']
    target = 'Salary'
    
    X = df[features]
    y = df[target]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
