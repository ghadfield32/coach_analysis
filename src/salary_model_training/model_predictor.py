

import pandas as pd
import joblib
import numpy as np  # Add this import for numpy
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
import logging

# Add relative imports for functions and constants
from .data_loader_preprocessor import preprocessed_datasets, filter_seasons, inverse_transform_injury_risk
from .model_trainer import load_and_preprocess_data, train_and_save_models, evaluate_models  # Add this line


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
LEAVE_ALONE_FEATURES = ['Season', 'Injury_Risk', 'SalaryPct'] 
PIPELINE_LEAVE_ALONE_FEATURES = ['Season', 'Injury_Risk'] 
columns_to_add_back_later = ['Season', 'Salary_Cap_Inflated', 'Player', 'SalaryPct']  

CATEGORICAL_FEATURES = ['Position', 'Team']
PASSTHROUGH_FEATURES = ['Season', 'Injury_Risk']

def load_models_and_pipeline(model_save_path, predict_season):
    season_model_path = model_save_path
    logger.debug(f"Loading models and pipeline for season {predict_season} from {season_model_path}")

    rf_model = joblib.load(f"{season_model_path}/best_rf_model.pkl")
    xgb_model = joblib.load(f"{season_model_path}/best_xgb_model.pkl")
    pipeline = joblib.load(f"{season_model_path}/preprocessing_pipeline.pkl")
    columns_to_re_add = joblib.load(f"{season_model_path}/columns_to_re_add.pkl")
    feature_names = joblib.load(f"{season_model_path}/feature_names.pkl")
    
    return rf_model, xgb_model, pipeline, columns_to_re_add, feature_names

def load_and_preprocess_test_data(file_path, predict_season, model_save_path):
    logger.debug("Loading and preprocessing test data...")
    cleaned_data, engineered_data, pipeline_data, _ = preprocessed_datasets(file_path)

    # Filter data for the prediction season
    _, test_data = filter_seasons(pipeline_data, predict_season)

    X_test = test_data.drop('SalaryPct', axis=1)
    y_test = test_data['SalaryPct']

    # Debugging the columns before transformation
    logger.debug(f"Test data shape before transformation: {X_test.shape}")
    logger.debug(f"Test data columns before transformation: {X_test.columns.tolist()}")

    # Check unique values for Injury_Risk and Total_Days_Injured before transformation
    logger.debug(f"Unique values of 'Injury_Risk' before transformation: {X_test['Injury_Risk'].unique()}")
    logger.debug(f"Unique values of 'Days_Injured_Percentage' before transformation: {test_data['Days_Injured_Percentage'].unique()}")

    rf_model, xgb_model, pipeline, columns_to_re_add, feature_names = load_models_and_pipeline(model_save_path, predict_season)

    # Separate and log the numerical columns
    X_test_numeric = X_test[NUMERIC_FEATURES]
    logger.debug(f"Numerical data before transformation (shape): {X_test_numeric.shape}")
    logger.debug(f"Numerical data before transformation (columns): {X_test_numeric.columns.tolist()}")
    logger.debug(f"Sample of numerical data: {X_test_numeric.head()}")

    # Transform the data
    X_test_transformed = pipeline.transform(X_test)

    # Debugging the transformed numerical data
    numeric_transformer = pipeline.named_transformers_['num']['scaler']
    transformed_numeric = numeric_transformer.transform(X_test_numeric)
    logger.debug(f"Transformed numerical data shape: {transformed_numeric.shape}")
    logger.debug(f"Sample of transformed numerical data: {transformed_numeric[:5]}")

    # Check passthrough Injury_Risk after transformation (since it's not transformed)
    logger.debug(f"Unique values of 'Injury_Risk' after transformation (passthrough): {X_test['Injury_Risk'].unique()}")

    # Check Total_Days_Injured after transformation (it should be included in the numeric transformations)
    logger.debug(f"Transformed 'Total_Days_Injured' values (numeric feature): {transformed_numeric[:, NUMERIC_FEATURES.index('Days_Injured_Percentage')][:5]}")

    # Debug the final transformed data shape
    logger.debug(f"Shape of transformed data: {X_test_transformed.shape}")
    
    return X_test, X_test_transformed, y_test, columns_to_re_add, feature_names, pipeline

def inverse_transform_and_add_context(rf_predictions, xgb_predictions, X_test, X_test_transformed, columns_to_re_add, feature_names, pipeline):
    logger.debug(f"Shape of rf_predictions: {rf_predictions.shape}")

    # Convert predictions to DataFrame
    rf_predictions_df = pd.DataFrame(rf_predictions, columns=['Predicted_SalaryPct'], index=X_test.index)
    xgb_predictions_df = pd.DataFrame(xgb_predictions, columns=['Predicted_SalaryPct'], index=X_test.index)

    # Inverse transform numerical features
    numeric_transformer = pipeline.named_transformers_['num']['scaler']
    X_test_numeric = X_test[NUMERIC_FEATURES]
    X_test_numeric_inverse = pd.DataFrame(
        numeric_transformer.inverse_transform(X_test_transformed[:, :len(NUMERIC_FEATURES)]),
        columns=NUMERIC_FEATURES,
        index=X_test.index
    )

    # Inverse transform categorical features
    categorical_transformer = pipeline.named_transformers_['cat']['onehot']
    transformed_cat_indices = slice(len(NUMERIC_FEATURES), -len(PASSTHROUGH_FEATURES))  # Indices of categorical features
    X_test_categorical_inverse = pd.DataFrame(
        categorical_transformer.inverse_transform(X_test_transformed[:, transformed_cat_indices]),
        columns=ONE_HOT_ENCODE_CATEGORICAL_FEATURES,
        index=X_test.index
    )

    # Handle passthrough features (Season, Injury_Risk) directly
    X_test_passthrough = X_test[PASSTHROUGH_FEATURES]

    # Concatenate inverse-transformed numeric, categorical, and passthrough columns
    X_test_inverse_transformed = pd.concat([X_test_numeric_inverse, X_test_categorical_inverse, X_test_passthrough], axis=1)

    # Re-add the context columns (e.g., Salary_Cap_Inflated, Total_Days_Injured)
    context_columns_df = pd.DataFrame(columns_to_re_add, index=X_test.index)

    # Inverse transform the Injury_Risk column back to original categories
    X_test_inverse_transformed = inverse_transform_injury_risk(X_test_inverse_transformed)

    # Final prediction DataFrames
    final_rf_df = pd.concat([X_test_inverse_transformed, context_columns_df, rf_predictions_df], axis=1)
    final_xgb_df = pd.concat([X_test_inverse_transformed, context_columns_df, xgb_predictions_df], axis=1)

    # Add Predicted_Salary column (in millions) by multiplying Predicted_SalaryPct with Salary_Cap_Inflated
    final_rf_df['Predicted_Salary'] = (final_rf_df['Predicted_SalaryPct'] * final_rf_df['Salary_Cap_Inflated'] / 1_000_000).round(2)
    final_xgb_df['Predicted_Salary'] = (final_xgb_df['Predicted_SalaryPct'] * final_xgb_df['Salary_Cap_Inflated'] / 1_000_000).round(2)

    # Add Predicted_Salary column (in millions) by multiplying Predicted_SalaryPct with Salary_Cap_Inflated
    final_rf_df['Salary'] = (final_rf_df['SalaryPct'] * final_rf_df['Salary_Cap_Inflated'] / 1_000_000).round(2)
    final_xgb_df['Salary'] = (final_xgb_df['SalaryPct'] * final_xgb_df['Salary_Cap_Inflated'] / 1_000_000).round(2)

    
    return final_rf_df, final_xgb_df


def save_predictions(final_rf_df, final_xgb_df, model_save_path):
    rf_save_path = f"{model_save_path}/rf_predictions.csv"
    xgb_save_path = f"{model_save_path}/xgb_predictions.csv"

    final_rf_df.to_csv(rf_save_path, index=False)
    final_xgb_df.to_csv(xgb_save_path, index=False)

    logger.info(f"Predictions saved to {rf_save_path} and {xgb_save_path}")


# Main function to run the prediction pipeline
def make_predictions(file_path, predict_season, model_save_path):
    logger.debug("Starting prediction pipeline...")

    # Step 1: Load and preprocess test data
    X_test, X_test_transformed, y_test, columns_to_re_add, feature_names, pipeline = load_and_preprocess_test_data(file_path, predict_season, model_save_path)

    logger.debug(f"Shape of X_test_transformed before predictions: {X_test_transformed.shape}")

    # Step 2: Load models
    rf_model, xgb_model, _, _, _ = load_models_and_pipeline(model_save_path, predict_season)

    # Step 3: Make predictions
    rf_predictions = rf_model.predict(X_test_transformed)
    xgb_predictions = xgb_model.predict(X_test_transformed)

    logger.debug(f"RF Predictions: {rf_predictions[:5]}")
    logger.debug(f"XGB Predictions: {xgb_predictions[:5]}")

    # Step 4: Inverse transform and add context
    final_rf_df, final_xgb_df = inverse_transform_and_add_context(rf_predictions, xgb_predictions, X_test, X_test_transformed, columns_to_re_add, feature_names, pipeline)

    # Drop one of the duplicate 'Season' columns
    if 'Season' in final_rf_df.columns:
        final_rf_df = final_rf_df.loc[:, ~final_rf_df.columns.duplicated()]
    
    if 'Season' in final_xgb_df.columns:
        final_xgb_df = final_xgb_df.loc[:, ~final_xgb_df.columns.duplicated()]

    # Step 5: Save predictions
    save_predictions(final_rf_df, final_xgb_df, model_save_path)

    return final_rf_df, final_xgb_df

if __name__ == "__main__":
    file_path = '../data/processed/nba_player_data_final_inflated.csv'
    predict_season = 2022
    model_save_path = f'../data/models/season_{predict_season}'

    rf_final_df, xgb_final_df = make_predictions(file_path, predict_season, model_save_path)
    print(rf_final_df)
    print(rf_final_df.columns)
