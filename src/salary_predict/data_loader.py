
import pandas as pd
import os

def get_project_root():
    """
    This function returns the path to the project root directory, 
    which is assumed to be the directory containing this file or its ancestors.
    """
    try:
        # Use the __file__ attribute to determine the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback to current working directory if __file__ is not available
        current_dir = os.getcwd()

    # Define the expected name of the root directory
    root_dir_name = 'coach_analysis'
    
    # Traverse upwards in the directory hierarchy to find the root directory
    while True:
        if os.path.basename(current_dir) == root_dir_name:
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError(f"Root directory '{root_dir_name}' not found.")
        current_dir = parent_dir
        
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
    
    print("Debug: df_predictions shape:", df_predictions.shape)
    print("Debug: df_predictions columns:", df_predictions.columns)
    
    # Merge predictions with actual data to get team information
    df_merged = pd.merge(df_predictions, df_actual[['Player', 'Team', 'Season']], 
                         left_on=['Player', 'Predicted_Season'], 
                         right_on=['Player', 'Season'], 
                         how='left')
    
    print("Debug: df_merged shape after merge:", df_merged.shape)
    print("Debug: df_merged columns after merge:", df_merged.columns)
    
    # Drop the redundant 'Season' column and rename 'Predicted_Season'
    df_merged = df_merged.drop(columns=['Season'])
    df_merged = df_merged.rename(columns={'Predicted_Season': 'Season'})
    
    # Ensure all required columns are present
    required_columns = ['Player', 'Season', 'Team', 'Age', 'Predicted_Salary', 'Previous_Season_Salary', 'Salary_Change']
    missing_columns = [col for col in required_columns if col not in df_merged.columns]
    if missing_columns:
        print(f"Debug: Missing columns: {missing_columns}")
        raise KeyError(f"The following required columns are missing in the merged dataframe: {', '.join(missing_columns)}")
    
    print("Debug: Unique teams in df_merged:", df_merged['Team'].unique())
    print("Debug: Number of unique teams in df_merged:", df_merged['Team'].nunique())
    
    # Filter by team if specified
    if team:
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
