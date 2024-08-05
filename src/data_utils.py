
import pandas as pd
import numpy as np
from process_utils import inflate_value

def clean_dataframe(df):
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Remove columns with all NaN values
    df = df.dropna(axis=1, how='all')
    
    # Remove rows with all NaN values
    df = df.dropna(axis=0, how='all')
    
    # Ensure only one 'Season' column exists
    season_columns = [col for col in df.columns if 'Season' in col]
    if len(season_columns) > 1:
        df = df.rename(columns={season_columns[0]: 'Season'})
        for col in season_columns[1:]:
            df = df.drop(columns=[col])
    
    # Remove '3PAr' and 'FTr' columns
    columns_to_remove = ['3PAr', 'FTr']
    df = df.drop(columns=columns_to_remove, errors='ignore')
    
    # Round numeric columns to 2 decimal places
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(2)
    
    return df

def merge_salary_cap_data(player_data, salary_cap_data):
    player_data['Season_Year'] = player_data['Season'].str[:4].astype(int)
    salary_cap_data['Season_Year'] = salary_cap_data['Season'].str[:4].astype(int)
    
    # Add inflation-adjusted salary cap
    salary_cap_data['Salary_Cap_Inflated'] = salary_cap_data.apply(
        lambda row: inflate_value(row['Salary Cap'], row['Season']),
        axis=1
    )
    
    # Merge salary cap data
    merged_data = pd.merge(player_data, salary_cap_data, on='Season_Year', how='left', suffixes=('', '_cap'))
    
    # Update salary cap columns
    cap_columns = ['Mid-Level Exception', 'Salary Cap', 'Luxury Tax', '1st Apron', '2nd Apron', 'BAE',
                   'Standard /Non-Taxpayer', 'Taxpayer', 'Team Room /Under Cap', 'Salary_Cap_Inflated']
    for col in cap_columns:
        if f'{col}_cap' in merged_data.columns:
            merged_data[col] = merged_data[col].fillna(merged_data[f'{col}_cap'])
            merged_data.drop(columns=[f'{col}_cap'], inplace=True)
    
    # Clean up temporary columns
    merged_data.drop(columns=['Season_Year'], inplace=True)
    
    # Clean the dataframe
    merged_data = clean_dataframe(merged_data)
    
    return merged_data

def validate_data(df):
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("Warning: Missing values found in the following columns:")
        print(missing_values[missing_values > 0])
    
    # Check for duplicate rows
    duplicates = df.duplicated()
    if duplicates.sum() > 0:
        print(f"Warning: {duplicates.sum()} duplicate rows found")
    
    # Check data types
    expected_types = {
        'Season': 'object',
        'Player': 'object',
        'Age': 'float64',
        'GP': 'float64',
        'MP': 'float64',
        'Salary': 'float64',
        'Team_Salary': 'float64',
        'Salary Cap': 'float64',
        'Salary_Cap_Inflated': 'float64'
    }
    for col, expected_type in expected_types.items():
        if col in df.columns:
            actual_type = df[col].dtype
            if str(actual_type) != expected_type:
                print(f"Warning: Column '{col}' has type {actual_type}, expected {expected_type}")
    
    # Check for negative values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (df[col] < 0).any():
            print(f"Warning: Negative values found in column '{col}'")
    
    return df
