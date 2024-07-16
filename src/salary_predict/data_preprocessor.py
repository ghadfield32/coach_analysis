
# data_preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

def handle_missing_values(df):
    df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    return df

def feature_engineering(df, use_inflated_data=False):
    df['PPG'] = df['PTS'] / df['GP']
    df['APG'] = df['AST'] / df['GP']
    df['RPG'] = df['TRB'] / df['GP']
    df['SPG'] = df['STL'] / df['GP']
    df['BPG'] = df['BLK'] / df['GP']
    df['TOPG'] = df['TOV'] / df['GP']
    df['WinPct'] = df['Wins'] / (df['Wins'] + df['Losses'])
    df['Availability'] = df['GP'] / 82
    
    # Calculate SalaryPct using the correct Salary Cap column
    df['SalaryPct'] = df['Salary'] / df['Salary Cap']
    
    df['SalaryGrowth'] = df.groupby('Player')['SalaryPct'].pct_change().fillna(0)

    return df

def calculate_vorp_salary_ratio(df):
    df['Salary_M'] = df['Salary'] / 1e6
    if 'VORP' in df.columns:
        df['VORP_Salary_Ratio'] = df['VORP'] / df['Salary_M']
    else:
        print("Warning: 'VORP' column not found. VORP/Salary ratio cannot be calculated.")
    return df

def cluster_career_trajectories(df):
    features = ['Age', 'Years of Service', 'PTS', 'TRB', 'AST', 'PER', 'WS', 'VORP']
    X = df[features]
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Adding cluster definitions
    cluster_definitions = {
        0: "Young Bench Players",
        1: "Rising Role Players",
        2: "Star Players",
        3: "Superstars",
        4: "Veteran Players"
    }
    
    df['Cluster_Definition'] = df['Cluster'].map(cluster_definitions)
    return df

