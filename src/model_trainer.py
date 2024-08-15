import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np

def inspect_data_types(X):
    print("Data types of features:")
    print(X.dtypes)
    object_columns = X.select_dtypes(include=['object']).columns
    if not object_columns.empty:
        print("Columns with object data types:", object_columns.tolist())
    else:
        print("No columns with object data types.")

def perform_grid_search(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for {model.__class__.__name__}: {grid_search.best_params_}")
    print(f"Best score for {model.__class__.__name__}: {-grid_search.best_score_}")
    return grid_search.best_estimator_

def train_and_save_models(X_train, y_train, model_save_path, scaler, feature_names, encoders, player_encoder, numeric_cols):
    # Inspect data types before training
    inspect_data_types(X_train)

    # Initialize models with default parameters
    rf_model = RandomForestRegressor(random_state=42)
    xgb_model = xgb.XGBRegressor(random_state=42, enable_categorical=True)

    # Define parameter grids for grid search
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    # Perform grid search
    best_rf_model = perform_grid_search(rf_model, rf_param_grid, X_train, y_train)
    best_xgb_model = perform_grid_search(xgb_model, xgb_param_grid, X_train, y_train)

    # Train models with best parameters
    best_rf_model.fit(X_train, y_train)
    best_xgb_model.fit(X_train, y_train)

    # Scale the features used for training
    X_train_scaled = scaler.fit_transform(X_train)

    # Save models, scaler, feature names, encoders, and other artifacts
    joblib.dump(best_rf_model, f"{model_save_path}/best_rf_model.pkl")
    joblib.dump(best_xgb_model, f"{model_save_path}/best_xgb_model.pkl")
    joblib.dump(scaler, f"{model_save_path}/scaler.pkl")
    joblib.dump(feature_names, f"{model_save_path}/feature_names.pkl")
    joblib.dump(encoders, f"{model_save_path}/encoders.pkl")
    joblib.dump(injury_risk_mapping, f"{model_save_path}/injury_risk_mapping.pkl")
    joblib.dump(numeric_cols, f"{model_save_path}/numeric_cols.pkl")

    joblib.dump(player_encoder, f"{model_save_path}/player_encoder.pkl")
    print("Models, scaler, feature names, encoders, and other artifacts trained and saved successfully.")

def evaluate_models(X_test, y_test, model_save_path):
    # Load models, scaler, and feature names
    rf_model = joblib.load(f"{model_save_path}/best_rf_model.pkl")
    xgb_model = joblib.load(f"{model_save_path}/best_xgb_model.pkl")

    # Make predictions
    rf_predictions = rf_model.predict(X_test)
    xgb_predictions = xgb_model.predict(X_test)

    # Evaluate models using multiple metrics
    metrics = {'Random Forest': rf_predictions, 'XGBoost': xgb_predictions}

    for model_name, predictions in metrics.items():
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"\n{model_name} Evaluation:")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R-squared: {r2}")
        
def filter_seasons(data, predict_season):
    """
    Filters the dataset into prior seasons and the target season for prediction.

    Args:
        data (pd.DataFrame): The dataset containing season data.
        predict_season (int): The season that you want to predict.

    Returns:
        tuple: A tuple containing two DataFrames:
            - prior_seasons_data: Data for seasons before the predict_season.
            - target_season_data: Data for the predict_season.
    """
    # Separate data into prior seasons and the target season
    prior_seasons_data = data[data['Season'] < predict_season]
    target_season_data = data[data['Season'] == predict_season]
    
    print(f"Data filtered. Prior seasons shape: {prior_seasons_data.shape}, Target season shape: {target_season_data.shape}")
    
    return target_season_data, prior_seasons_data

# Data preprocessing
def load_and_preprocess_data(file_path, predict_season):
    data = load_data(file_path)
    data = format_season(data)
    _, prior_seasons_data = filter_seasons(data, predict_season)
    prior_seasons_data = clean_data(prior_seasons_data)
    prior_seasons_data = engineer_features(prior_seasons_data)
    return prior_seasons_data

# Feature selection
def select_features(data, target_column, additional_features=[]):
    top_features = ['PPG', 'APG', 'RPG', 'SPG', 'TOPG', 'Years of Service', 'PER', 'VORP', 'WSPG', 'OWSPG']
    
    # Add 'Injury_Risk', 'Position', and 'Team' to ensure they're included for encoding
    top_features += ['Injury_Risk', 'Position', 'Team']
    
    # Add any additional features
    top_features += additional_features
    
    # Ensure all selected features are in the dataset
    available_features = [col for col in top_features if col in data.columns]
    
    print("Available features for modeling:", available_features)  # Debug statement

    X = data[available_features]
    y = data[target_column]
    return X, y

# Main execution
if __name__ == "__main__":
    file_path = 'data/processed/nba_player_data_final_inflated.csv'
    predict_season = 2023
    target_column = 'SalaryPct'

    # Load and preprocess data
    preprocessed_data = load_and_preprocess_data(file_path, predict_season)
    print("Columns after preprocessing:", preprocessed_data.columns)

    # Select features
    X, y = select_features(preprocessed_data, target_column)
    print("Columns after feature selection:", X.columns)

    # Encode data
    encoded_data, injury_risk_mapping, encoders, scaler, numeric_cols, player_encoder = encode_data(X)
    print("Columns after encoding:", encoded_data.columns)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(encoded_data, y, test_size=0.2, random_state=42)

    # Train and evaluate models
    model_save_path = 'data/models'
    train_and_save_models(X_train, y_train, model_save_path, scaler, encoded_data.columns, encoders, injury_risk_mapping, numeric_cols)
    evaluate_models(X_test, y_test, model_save_path)
