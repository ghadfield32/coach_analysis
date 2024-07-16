
import pandas as pd
import os

def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))

def load_data(inflated=False):
    root_dir = get_project_root()
    file_name = 'final_salary_data_with_yos_and_inflated_cap.csv' if inflated else 'final_salary_data_with_yos_and_cap.csv'
    file_path = os.path.join(root_dir, 'data', 'processed', file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please check the file path.")
    
    df = pd.read_csv(file_path)
    if 'Salary' not in df.columns:
        raise KeyError("The 'Salary' column is missing in the dataset.")
    df['Season'] = df['Season'].str[:4].astype(int)
    
    # Use the correct salary cap column
    if inflated:
        df['Salary Cap'] = df['Salary_Cap_Inflated']
    
    return df

def load_predictions(inflated=False):
    root_dir = get_project_root()
    file_name = 'salary_predictions_inflated.csv' if inflated else 'salary_predictions.csv'
    predictions = pd.read_csv(os.path.join(root_dir, 'data', 'predictions', file_name))
    original_data = load_data(inflated)
    merged_data = merge_predictions_with_original(predictions, original_data)
    merged_data = merged_data.drop_duplicates(subset=['Player', 'Predicted_Season'], keep='first')
    return merged_data

def merge_predictions_with_original(predictions, original_data):
    merged = predictions.merge(original_data[['Player', 'Position']], on='Player', how='left')
    merged['Position'] = merged['Position'].fillna('Unknown')
    merged.rename(columns={'Previous_Season_Salary': 'Salary'}, inplace=True)
    return merged

