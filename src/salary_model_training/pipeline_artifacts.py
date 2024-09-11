
import joblib
import pandas as pd
import os
import logging

# Relative imports
from .data_loader_preprocessor import preprocessed_datasets, format_season, build_pipeline, get_feature_names, create_season_folder


logger = logging.getLogger(__name__)

# Constants
INJURY_RISK_MAP = {
    'Low Risk': 1,
    'Moderate Risk': 2,
    'High Risk': 3
}

REVERSE_INJURY_RISK_MAP = {
    1: 'Low Risk',
    2: 'Moderate Risk',
    3: 'High Risk'
}

# Define feature groups
NUMERIC_FEATURES = ['Age', 'Years of Service', 'PER', 'TS%', 'ORB%', 'DRB%', 'TRB%', 'AST%', 
                    'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 'WS/48', 'VORP', 
                    'PPG', 'APG', 'SPG', 'TPG', 'BPG', 'Availability', 
                    'Efficiency', 'Days_Injured_Percentage', 'ValueOverReplacement', 'ExperienceSquared']

ONE_HOT_ENCODE_CATEGORICAL_FEATURES = ['Position', 'Team']
LEAVE_ALONE_FEATURES = ['Season', 'Injury_Risk']

# Restore to DataFrame using the pipeline to untransform the data
def restore_to_dataframe(processed_data, pipeline, original_data):
    """Restores the transformed data to a DataFrame, using pipeline to inverse transform the data."""
    
    # Get column names from the pipeline transformers
    num_col_names = NUMERIC_FEATURES
    cat_col_names = pipeline.named_transformers_['cat']['onehot'].get_feature_names_out(ONE_HOT_ENCODE_CATEGORICAL_FEATURES)
    
    # Combine all column names: numeric, one-hot encoded, and passthrough
    all_col_names = list(num_col_names) + list(cat_col_names) + LEAVE_ALONE_FEATURES
    
    # Create DataFrame with processed data
    restored_df = pd.DataFrame(processed_data, columns=all_col_names)
    logger.info(f"Restored DataFrame shape: {restored_df.shape}")
    logger.info(f"Restored DataFrame columns: {restored_df.columns.tolist()}")
    
    # Inverse transform numeric features
    scaler = pipeline.named_transformers_['num']['scaler']
    numeric_data_scaled = restored_df[num_col_names]
    
    # Perform inverse transformation on numeric columns
    restored_df[num_col_names] = scaler.inverse_transform(numeric_data_scaled)
    logger.debug(f"First few rows after inverse transforming numerical columns:\n{restored_df[num_col_names].head()}")
    
    # Decode one-hot encoded 'Team' and 'Position' back to categorical values
    logger.debug("Decoding one-hot encoded columns for Team and Position...")
    for feature in ONE_HOT_ENCODE_CATEGORICAL_FEATURES:
        onehot_encoded_columns = [col for col in restored_df.columns if col.startswith(f"{feature}_")]
        
        # Get the index of the max value in one-hot encoded columns to map back to original category
        restored_df[feature] = restored_df[onehot_encoded_columns].idxmax(axis=1).str.replace(f"{feature}_", "")
        
        # Drop one-hot encoded columns after decoding
        restored_df.drop(onehot_encoded_columns, axis=1, inplace=True)
        logger.debug(f"Decoded {feature} and dropped one-hot encoded columns.")
    
    # Restore Injury_Risk back to original categories
    logger.debug("Decoding Injury_Risk...")
    restored_df['Injury_Risk'] = restored_df['Injury_Risk'].map(REVERSE_INJURY_RISK_MAP)
    logger.debug(f"First few Injury_Risk values after decoding:\n{restored_df['Injury_Risk'].head()}")
    
    print("Pre join checks, original data columns =", original_data.columns, "restored df columns =", restored_df.columns)
    # Left join the Salary_Cap_Inflated based on Season
    salary_cap_df = original_data[['Season', 'Salary_Cap_Inflated']].drop_duplicates()
    restored_df = pd.merge(restored_df, salary_cap_df, on='Season', how='left')
    logger.info(f"Left joined Salary_Cap_Inflated onto restored DataFrame based on Season.")

    # Left join the SalaryPct back into restored_df based on Player and Season
    if 'SalaryPct' in original_data.columns:
        salary_pct_df = original_data[['Player', 'Season', 'SalaryPct']].drop_duplicates()
        restored_df = pd.merge(restored_df, salary_pct_df, on=['Player', 'Season'], how='left')
        logger.info(f"Left joined SalaryPct back onto restored DataFrame based on Player and Season.")
    else:
        logger.warning("SalaryPct not found in the original data. Ensure it is computed earlier in the pipeline.")
    
    return restored_df

# Save pipeline and other artifacts
def save_pipeline_and_artifacts(pipeline, feature_names, season, base_path='../data/models'):
    """Saves the pipeline and feature names for a given season."""
    season_folder = create_season_folder(base_path, season)
    
    # Save pipeline
    pipeline_path = os.path.join(season_folder, 'preprocessing_pipeline.pkl')
    joblib.dump(pipeline, pipeline_path)
    logger.info(f"Pipeline saved at {pipeline_path}")
    
    # Save feature names
    feature_names_path = os.path.join(season_folder, 'feature_names.pkl')
    print("We Saved These Feature_names to the path =", feature_names, "to this path:", feature_names_path)
    joblib.dump(feature_names, feature_names_path)
    logger.info(f"Feature names saved at {feature_names_path}")

def load_pipeline(season, base_path='../data/models'):
    pipeline_path = os.path.join(base_path, f'season_{season}', 'preprocessing_pipeline.pkl')
    return joblib.load(pipeline_path)

def load_feature_names(season, base_path='../data/models'):
    feature_names_path = os.path.join(base_path, f'season_{season}', 'feature_names.pkl')
    return joblib.load(feature_names_path)



# Main for testing
if __name__ == "__main__":
    try:
        file_path = '../data/processed/nba_player_data_final_inflated.csv'
        original_data = pd.read_csv(file_path)
        season = 2023
        
        cleaned_data, engineered_data, injury_encoded_data = preprocessed_datasets(file_path)
        print("clean columns = ", cleaned_data.columns)
        print("engineered_data columns = ", engineered_data.columns)
        print("Pre-pipeline columns = ", injury_encoded_data.columns)
        
        # Load pipeline
        pipeline = load_pipeline(season)
        logger.info("Pipeline loaded successfully.")
        
        # Get processed data (placeholder for actual pipeline process)
        processed_data = pipeline.transform(injury_encoded_data)
        
        # Restore data
        restored_df = restore_to_dataframe(processed_data, pipeline, original_data)
        logger.info(f"Restored DataFrame shape: {restored_df.shape}")
        
        # Test saving the pipeline
        feature_names = list(injury_encoded_data.columns)
        save_pipeline_and_artifacts(pipeline, feature_names, season)
        logger.info(f"Pipeline saved successfully for season {season}.")
        
    except Exception as e:
        logger.critical(f"Error in pipeline artifact operations: {e}")
