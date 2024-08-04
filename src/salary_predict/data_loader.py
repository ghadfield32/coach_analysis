
import pandas as pd
import os

def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))

def load_data(inflated=False):
    root_dir = get_project_root()
    file_name = 'final_salary_data_with_yos_and_inflated_cap_2000_on.csv'
    file_path = os.path.join(root_dir, 'data', 'processed', file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please check the file path.")
    
    df = pd.read_csv(file_path)
    if 'Salary' not in df.columns:
        raise KeyError("The 'Salary' column is missing in the dataset.")
    
    # Convert 'Season' to the correct format if necessary
    if df['Season'].dtype == 'object':
        df['Season'] = df['Season'].str[:4].astype(int)
    
    # Ensure both 'Salary Cap' and 'Salary_Cap_Inflated' columns are present
    if 'Salary Cap' not in df.columns:
        raise KeyError("The 'Salary Cap' column is missing in the dataset.")
    if 'Salary_Cap_Inflated' not in df.columns:
        df['Salary_Cap_Inflated'] = df['Salary Cap']  # Use non-inflated as fallback
    
    # Use the appropriate salary cap column based on the 'inflated' parameter
    if inflated:
        df['Salary Cap'] = df['Salary_Cap_Inflated']
    else:
        df['Salary_Cap_Inflated'] = df['Salary Cap']
    
    return df

def load_predictions(inflated=False, team=None):
    # Load the actual data
    df_actual = load_data(inflated)
    
    # Load predictions
    root_dir = get_project_root()
    predictions_file = 'salary_predictions_inflated.csv' if inflated else 'salary_predictions.csv'
    predictions_path = os.path.join(root_dir, 'data', 'predictions', predictions_file)
    
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"The predictions file {predictions_path} does not exist.")
    
    df_predictions = pd.read_csv(predictions_path)
    
    # Check for required columns
    required_columns_predictions = ['Player', 'Predicted_Season']
    required_columns_actual = ['Player', 'Season']
    
    missing_columns_predictions = [col for col in required_columns_predictions if col not in df_predictions.columns]
    missing_columns_actual = [col for col in required_columns_actual if col not in df_actual.columns]
    
    if missing_columns_predictions:
        raise KeyError(f"The following required columns are missing in the predictions dataframe: {', '.join(missing_columns_predictions)}")
    if missing_columns_actual:
        raise KeyError(f"The following required columns are missing in the actual data dataframe: {', '.join(missing_columns_actual)}")
    
    # Rename 'Predicted_Season' to 'Season' in predictions dataframe for merging
    df_predictions = df_predictions.rename(columns={'Predicted_Season': 'Season'})
    
    # Merge predictions with actual data
    df_merged = pd.merge(df_predictions, df_actual, on=['Player', 'Season'], suffixes=('_pred', ''), how='left')
    
    # Rename columns to match expected names
    df_merged = df_merged.rename(columns={
        'Season': 'Predicted_Season',  # Change back to 'Predicted_Season'
        'Salary': 'Previous_Season_Salary',
        'Predicted_Salary_Pct': 'SalaryPct',
        'Age_pred': 'Age'  # Use predicted age
    })
    
    # Select relevant columns
    relevant_columns = ['Player', 'Predicted_Season', 'Team', 'Age', 'Position', 'Previous_Season_Salary', 
                        'Predicted_Salary', 'Salary_Change', 'SalaryPct', 'GP', 'MP', 'PTS', 'TRB', 'AST', 
                        'FG%', '3P%', 'FT%', 'PER', 'WS', 'VORP']
    
    # Check if all relevant columns are present
    missing_columns = [col for col in relevant_columns if col not in df_merged.columns]
    if missing_columns:
        print(f"Warning: The following columns are missing and will be excluded: {', '.join(missing_columns)}")
        relevant_columns = [col for col in relevant_columns if col in df_merged.columns]
    
    df_merged = df_merged[relevant_columns]
    
    # Filter by team if specified
    if team:
        if 'Team' not in df_merged.columns:
            raise KeyError("The 'Team' column is missing in the merged dataframe.")
        df_merged = df_merged[df_merged['Team'] == team]
    
    return df_merged

def merge_predictions_with_original(predictions, original_data):
    merged = predictions.merge(original_data[['Player', 'Position']], on='Player', how='left')
    merged['Position'] = merged['Position'].fillna('Unknown')
    merged.rename(columns={'Previous_Season_Salary': 'Salary'}, inplace=True)
    return merged

if __name__ == "__main__":
    get_project_root()
    # Example usage
    print("Loading data...")
    df = load_data(inflated=False)
    print("\nDataframe shape:", df.shape)
    print("\nColumns:", df.columns)
    print("\nFirst few rows:")
    print(df.head())
    print("\nNaN values:")
    print(df.isna().sum())
    
    print("\nLoading predictions...")
    predictions = load_predictions(inflated=False)
    print("\nPredictions shape:", predictions.shape)
    print("\nPredictions columns:", predictions.columns)
    print("\nFirst few rows of predictions:")
    print(predictions.head())
    print("\nNaN values in predictions:")
    print(predictions.isna().sum())
