import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
import os
import logging
from .data_loader_preprocessor import preprocessed_datasets, build_pipeline, filter_seasons, get_feature_names

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def perform_grid_search(model, param_grid, X_train, y_train):
    """Performs grid search for hyperparameter tuning."""
    logger.debug(f"Starting grid search for {model.__class__.__name__}. Parameters: {param_grid}")
    logger.debug(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=1)
    grid_search.fit(X_train, y_train)
    logger.info(f"Best parameters for {model.__class__.__name__}: {grid_search.best_params_}")
    logger.info(f"Best score for {model.__class__.__name__}: {-grid_search.best_score_}")
    return grid_search.best_estimator_

def train_and_save_models(X_train, y_train, model_save_path):
    """Train models and save them along with preprocessing pipeline."""
    logger.debug(f"Starting model training. Model save path: {model_save_path}")
    logger.debug(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    rf_model = RandomForestRegressor(random_state=42)
    xgb_model = xgb.XGBRegressor(random_state=42, enable_categorical=False)

    rf_param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10], 'min_samples_split': [2, 5]}
    xgb_param_grid = {'n_estimators': [50, 100], 'max_depth': [3], 'learning_rate': [0.01]}

    logger.debug(f"Performing grid search for RandomForestRegressor.")
    best_rf_model = perform_grid_search(rf_model, rf_param_grid, X_train, y_train)
    
    logger.debug(f"Performing grid search for XGBoostRegressor.")
    best_xgb_model = perform_grid_search(xgb_model, xgb_param_grid, X_train, y_train)

    # Save the models
    joblib.dump(best_rf_model, os.path.join(model_save_path, 'best_rf_model.pkl'))
    joblib.dump(best_xgb_model, os.path.join(model_save_path, 'best_xgb_model.pkl'))
    logger.info(f"Models saved in {model_save_path}.")

def evaluate_models(X_test, y_test, model_save_path):
    """Evaluate models on the test set and save predictions."""
    rf_model = joblib.load(f"{model_save_path}/best_rf_model.pkl")
    xgb_model = joblib.load(f"{model_save_path}/best_xgb_model.pkl")

    logger.debug(f"Loaded models for evaluation.")
    logger.debug(f"Evaluating on test data. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    rf_predictions = rf_model.predict(X_test)
    xgb_predictions = xgb_model.predict(X_test)

    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_r2 = r2_score(y_test, rf_predictions)
    rf_mse = mean_squared_error(y_test, rf_predictions)

    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
    xgb_mae = mean_absolute_error(y_test, xgb_predictions)
    xgb_r2 = r2_score(y_test, xgb_predictions)
    xgb_mse = mean_squared_error(y_test, xgb_predictions)

    logger.info(f"\nRandom Forest RMSE: {rf_rmse}")
    logger.info(f"Random Forest MAE: {rf_mae}")
    logger.info(f"Random Forest R²: {rf_r2}")
    logger.info(f"Random Forest MSE: {rf_mse}")

    logger.info(f"\nXGBoost RMSE: {xgb_rmse}")
    logger.info(f"XGBoost MAE: {xgb_mae}")
    logger.info(f"XGBoost R²: {xgb_r2}")
    logger.info(f"XGBoost MSE: {xgb_mse}")

    eval_results = {
        'rf_predictions': rf_predictions,
        'xgb_predictions': xgb_predictions,
        'rf_rmse': rf_rmse,
        'rf_mae': rf_mae,
        'rf_r2': rf_r2,
        'rf_mse': rf_mse,
        'xgb_rmse': xgb_rmse,
        'xgb_mae': xgb_mae,
        'xgb_r2': xgb_r2,
        'xgb_mse': xgb_mse
    }

    eval_save_path = f"{model_save_path}/evaluation_results.pkl"
    joblib.dump(eval_results, eval_save_path)
    logger.info(f"Evaluation results saved at {eval_save_path}")
    
    return eval_results

def load_and_preprocess_data(file_path, predict_season, model_save_path):
    """Load data, filter by seasons, and apply preprocessing pipeline."""
    logger.debug(f"Loading data and preprocessing for season {predict_season}")
    
    # Step 1: Preprocess the dataset
    cleaned_data, engineered_data, pipeline_data, columns_to_re_add = preprocessed_datasets(file_path)
    
    # Step 2: Split data into train and test sets based on season
    train_data, test_data = filter_seasons(pipeline_data, predict_season)

    # Step 3: Separate features (X) and target (y)
    X_train = train_data.drop('SalaryPct', axis=1)
    y_train = train_data['SalaryPct']
    X_test = test_data.drop('SalaryPct', axis=1)
    y_test = test_data['SalaryPct']

    # Step 4: Build and apply the pipeline
    pipeline = build_pipeline()

    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    # Save the fitted pipeline
    joblib.dump(pipeline, os.path.join(model_save_path, 'preprocessing_pipeline.pkl'))

    # Save the columns to re-add later
    joblib.dump(columns_to_re_add, os.path.join(model_save_path, 'columns_to_re_add.pkl'))

    # Save Features
    all_col_names = get_feature_names(pipeline)
    print("all column names = ", all_col_names)
    joblib.dump(all_col_names, os.path.join(model_save_path, 'feature_names.pkl'))
    
    return X_train_transformed, X_test_transformed, y_train, y_test

# Model Training Pipeline
if __name__ == "__main__":
    file_path = '../data/processed/nba_player_data_final_inflated.csv'
    predict_season = 2022
    model_save_path = f'../data/models/season_{predict_season}'

    
    logger.debug(f"Starting the pipeline for season {predict_season} with file: {file_path}")
    
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, predict_season, model_save_path)
    
    # Train and save models
    train_and_save_models(X_train, y_train, model_save_path)
    
    # Evaluate models on the test set
    evaluated_models = evaluate_models(X_test, y_test, model_save_path)
    print("metrics = ", evaluated_models)
