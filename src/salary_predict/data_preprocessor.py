
# data_preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

def handle_missing_values(df):
    df = df.copy()
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    print(f"Number of numeric columns: {len(numeric_columns)}")
    print(f"Numeric columns: {numeric_columns}")
    
    # Remove columns with all NaN values
    df = df.dropna(axis=1, how='all')
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    print(f"Number of numeric columns after dropping all-NaN columns: {len(numeric_columns)}")
    
    imputer = SimpleImputer(strategy='mean')
    imputed_data = imputer.fit_transform(df[numeric_columns])
    print(f"Shape of imputed data: {imputed_data.shape}")
    print(f"Shape of original numeric data: {df[numeric_columns].shape}")
    
    df[numeric_columns] = imputed_data
    return df

def feature_engineering(df, use_inflated_data=False):
    df = df.copy()
    # Calculate per-game stats if not already present
    if 'PPG' not in df.columns:
        df['PPG'] = df['PTS'] / df['GP']
    if 'APG' not in df.columns:
        df['APG'] = df['AST'] / df['GP']
    if 'RPG' not in df.columns:
        df['RPG'] = df['TRB'] / df['GP']
    if 'SPG' not in df.columns:
        df['SPG'] = df['STL'] / df['GP']
    if 'BPG' not in df.columns:
        df['BPG'] = df['BLK'] / df['GP']
    if 'TOPG' not in df.columns:
        df['TOPG'] = df['TOV'] / df['GP']
    
    # Calculate win percentage if not already present
    if 'WinPct' not in df.columns:
        df['WinPct'] = df['Wins'] / (df['Wins'] + df['Losses'])
    
    # Calculate availability if not already present
    if 'Availability' not in df.columns:
        df['Availability'] = df['GP'] / 82
    
    # Calculate SalaryPct using the correct Salary Cap column
    salary_cap_column = 'Salary_Cap_Inflated' if use_inflated_data else 'Salary Cap'
    if salary_cap_column not in df.columns:
        raise KeyError(f"The '{salary_cap_column}' column is missing in the dataset.")
    df['SalaryPct'] = df['Salary'] / df[salary_cap_column]
    
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

if __name__ == "__main__":
    # from data_loader import load_data
    
    print("Loading data...")
    df = load_data(inflated=False)
    
    print("\nHandling missing values...")
    df = handle_missing_values(df)
    print("NaN values after handling:")
    print(df.isna().sum())
    
    print("\nPerforming feature engineering...")
    df = feature_engineering(df)
    print("New columns after feature engineering:", df.columns)
    
    print("\nCalculating VORP salary ratio...")
    df = calculate_vorp_salary_ratio(df)
    print("VORP salary ratio stats:")
    print(df['VORP_Salary_Ratio'].describe())
    
    print("\nClustering career trajectories...")
    df = cluster_career_trajectories(df)
    print("Cluster distribution:")
    print(df['Cluster_Definition'].value_counts())
