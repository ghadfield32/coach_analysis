
import os
import joblib
import pandas as pd
import numpy as np
import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
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
                    'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 'WS/48', 
                    'PPG', 'APG', 'SPG', 'TPG', 'BPG', 'Availability', 
                    'Efficiency', 'Days_Injured_Percentage', 'ValueOverReplacement', 'ExperienceSquared']

ONE_HOT_ENCODE_CATEGORICAL_FEATURES = ['Position', 'Team']
LEAVE_ALONE_FEATURES = ['Season', 'Injury_Risk', 'SalaryPct'] #SalaryPct included so it's included in engineer data filter
PIPELINE_LEAVE_ALONE_FEATURES = ['Season', 'Injury_Risk'] #SalaryPct taken out because this goes through the pipeline. So it's included in engineer features, split into train/test and x/y datasets, then input through the pipeline where it shouldn't be used
columns_to_add_back_later = ['Season', 'Salary_Cap_Inflated', 'Player', 'SalaryPct']  

# Format Season Column
def format_season(data):
    """Converts the 'Season' column from 'YYYY-YY' to 'YYYY' format."""
    try:
        data['Season'] = data['Season'].apply(lambda x: int(x.split('-')[0]))
        logger.info(f"Seasons in data: {data['Season'].unique()}")
        logger.info(f"Shape after season formatting: {data.shape}")
        logger.info(f"Null values after season formatting:\n{data.isnull().sum()}")
        return data
    except Exception as e:
        logger.error(f"Failed to format season data: {e}")
        raise
    
def filter_seasons(data, predict_season):
    """Split the data into prior seasons (train) and the selected season (test)."""
    prior_seasons_data = data[data['Season'] < predict_season]
    target_season_data = data[data['Season'] == predict_season]
    
    logger.debug(f"Data filtered. Prior seasons shape: {prior_seasons_data.shape}, Target season shape: {target_season_data.shape}")
    logger.debug(f"Feature columns used for training: {prior_seasons_data.columns.tolist()}")

    return prior_seasons_data, target_season_data

# Get Feature Names from Pipeline
def get_feature_names(pipeline):
    """Extract feature names after applying transformations in the pipeline."""
    # Numeric feature names
    num_col_names = NUMERIC_FEATURES
    
    # Categorical feature names (after one-hot encoding)
    cat_col_names = pipeline.named_transformers_['cat']['onehot'].get_feature_names_out(ONE_HOT_ENCODE_CATEGORICAL_FEATURES)
    
    # Combine all column names: numeric, one-hot encoded, and passthrough (without 'SalaryPct')
    all_col_names = list(num_col_names) + list(cat_col_names) + PIPELINE_LEAVE_ALONE_FEATURES
    
    return all_col_names


# Label Encoding Injury Risk
def label_encode_injury_risk(data):
    """Encode Injury_Risk using predefined mapping."""
    logger.debug("Label encoding Injury_Risk...")
    logger.debug(f"First few Injury_Risk values before encoding:\n{data['Injury_Risk'].head()}")
    
    # Encode Injury_Risk
    data['Injury_Risk'] = data['Injury_Risk'].map(INJURY_RISK_MAP)
    logger.debug(f"First few Injury_Risk values after encoding:\n{data['Injury_Risk'].head()}")
    
    return data

def inverse_transform_injury_risk(data):
    """Inverse transform Injury_Risk using predefined reverse mapping."""
    logger.debug("Inverse transforming Injury_Risk...")
    logger.debug(f"First few Injury_Risk values before inverse transformation:\n{data['Injury_Risk'].head()}")

    # Inverse transform Injury_Risk
    data['Injury_Risk'] = data['Injury_Risk'].map(REVERSE_INJURY_RISK_MAP)
    logger.debug(f"First few Injury_Risk values after inverse transformation:\n{data['Injury_Risk'].head()}")
    
    return data



# Step 1: load and clean the data
def clean_data(file_path):
    """Load and clean data."""
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded. Initial shape: {data.shape}")

        # Handle missing percentages and drop unnecessary columns
        data['3P%'] = np.where(data['3PA'] != 0, data['3P'] / data['3PA'], np.nan)
        data['FT%'] = np.where(data['FTA'] != 0, data['FT'] / data['FTA'], np.nan)
        data['2P%'] = np.where(data['2PA'] != 0, data['2P'] / data['2PA'], np.nan)
        data.drop(['3P%', 'FT%', '2P%'], axis=1, inplace=True)

        columns_to_remove = ['Salary Cap', 'Luxury Tax', '1st Apron', 'BAE', 'Standard /Non-Taxpayer', 
                             'Taxpayer', 'Team Room /Under Cap', 'Wins', 'Losses', '2nd Apron', 'Injury_Periods']
        data.drop(columns_to_remove, axis=1, inplace=True)

        # Filter out rows with nulls in advanced stats
        advanced_stats_columns = ['PER', 'TS%', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 
                                  'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP']
        data = data.dropna(subset=advanced_stats_columns)
        
        logger.info(f"Final shape after processing: {data.shape}")
        return data

    except Exception as e:
        logger.error(f"Data cleaning failed: {e}")
        raise

# Feature Engineering
def engineer_features(data):
    """Feature engineering step where new features are derived from existing ones."""
    per_game_cols = ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'TOV']
    for col in per_game_cols:
        data[f'{col[0]}PG'] = data[col] / data['GP']
    
    data['Availability'] = data['GP'] / 82
    data['SalaryPct'] = data['Salary'] / data['Salary_Cap_Inflated']
    data['Efficiency'] = (data['PTS'] + data['TRB'] + data['AST'] + data['STL'] + data['BLK']) / (data['FGA'] + data['FTA'] + data['TOV'] + 1)
    data['ValueOverReplacement'] = data['VORP'] / data['GP'] 
    data['ExperienceSquared'] = data['Years of Service'] ** 2
    data['Days_Injured_Percentage'] = data['Total_Days_Injured'] / data['GP']

    engineered_data = data.copy()

    columns_to_keep_for_pipeline = NUMERIC_FEATURES + ONE_HOT_ENCODE_CATEGORICAL_FEATURES + LEAVE_ALONE_FEATURES
    pipeline_data = data[columns_to_keep_for_pipeline]
    columns_to_re_add = data[columns_to_add_back_later]
    
    return engineered_data, pipeline_data, columns_to_re_add

# After preprocessing, extract SalaryPct as the target (y)
def preprocessed_datasets(file_path):
    original_data = pd.read_csv(file_path)
    
    # Load and preprocess data
    cleaned_data = clean_data(file_path)
    cleaned_data = format_season(cleaned_data)
    
    # Get the pipeline data and columns to re-add
    engineered_data, pipeline_data, columns_to_re_add = engineer_features(cleaned_data)
    
    # Label encode the pipeline data
    pipeline_data = label_encode_injury_risk(pipeline_data)
    
    return cleaned_data, engineered_data, pipeline_data, columns_to_re_add

# Split the dataset into train and test sets based on the season
def filter_seasons(data, predict_season):
    """Split the data into prior seasons (train) and the selected season (test)."""
    prior_seasons_data = data[data['Season'] < predict_season]
    target_season_data = data[data['Season'] == predict_season]
    
    return prior_seasons_data, target_season_data

# Build the Pipeline
def build_pipeline():
    """Creates a data processing pipeline that applies encoding and scaling transformations."""
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, ONE_HOT_ENCODE_CATEGORICAL_FEATURES),
            ('passthrough', 'passthrough', PIPELINE_LEAVE_ALONE_FEATURES)  # Season and Injury_Risk passthrough
        ],
        remainder='drop'
    )
    
    return preprocessor


# Main execution
if __name__ == "__main__":
    try:
        file_path = '../data/processed/nba_player_data_final_inflated.csv'
        season = 2022
        original_data = pd.read_csv(file_path)

        # Step 1: Preprocess the dataset
        cleaned_data, engineered_data, pipeline_data, columns_to_re_add = preprocessed_datasets(file_path)

        # Step 2: Split data into train and test sets based on season
        train_data, test_data = filter_seasons(pipeline_data, season)
        print("days injured unique values = ", train_data['Days_Injured_Percentage'].unique())
        print("days injured unique values = ", test_data['Days_Injured_Percentage'].unique())
        # Step 3: Separate features (X) and target (y)
        X_train = train_data.drop('SalaryPct', axis=1)
        y_train = train_data['SalaryPct']
        X_test = test_data.drop('SalaryPct', axis=1)
        y_test = test_data['SalaryPct']

        # Step 4: Build and apply the pipeline
        pipeline = build_pipeline()
        # Before and after pipeline debug
        logger.debug(f"Before pipeline transformation: {X_train.columns.tolist()}")
        X_train_transformed = pipeline.fit_transform(X_train)
        logger.debug(f"After pipeline transformation: {X_train_transformed.shape}")
        logger.debug(f"Transformed feature names: {pipeline.get_feature_names_out()}")
        print("Sample of transformed data:", X_train_transformed[:5])


        # Save the fitted pipeline
        joblib.dump(pipeline, f'../data/models/season_{season}/preprocessing_pipeline.pkl')
        
        columns_to_re_add_train_data, columns_to_re_add_test_data = filter_seasons(columns_to_re_add, season)
        columns_to_re_add_train_data = columns_to_re_add_train_data.drop('Season', axis=1)
        columns_to_re_add = columns_to_re_add_test_data.drop('Season', axis=1)
        print("columns_to_re_add =", columns_to_re_add)
        # Save columns to re-add later
        joblib.dump(columns_to_re_add, f'../data/models/season_{season}/columns_to_re_add.pkl')

        # all_col_names = get_feature_names(pipeline)
        # print("all column names = ", all_col_names)
        # joblib.dump(all_col_names, f'../data/models/season_{season}/feature_names.pkl')


    except Exception as e:
        logger.critical(f"Critical error in data processing pipeline: {e}")
        raise
