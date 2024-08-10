
import os
import numpy as np
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint, uniform
import xgboost as xgb

def train_models(X_train, y_train, debug=False):
    if debug:
        print("Debug: Features used for training:")
        print(X_train.columns.tolist())
        print(f"Debug: Number of features: {X_train.shape[1]}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if debug:
        print("Debug: Scaling completed. Mean of scaled X_train:", np.mean(X_train_scaled, axis=0))
        print("Debug: Scaling completed. Std of scaled X_train:", np.std(X_train_scaled, axis=0))

    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42)
    }

    param_distributions = {
        'Random Forest': {
            'n_estimators': randint(100, 500),
            'max_depth': randint(5, 20),
            'min_samples_split': randint(2, 11),
            'min_samples_leaf': randint(1, 11)
        },
        'Gradient Boosting': {
            'n_estimators': randint(100, 500),
            'learning_rate': uniform(0.01, 0.2),
            'max_depth': randint(3, 10),
            'min_samples_split': randint(2, 11),
            'min_samples_leaf': randint(1, 11)
        },
        'XGBoost': {
            'n_estimators': randint(100, 500),
            'learning_rate': uniform(0.01, 0.2),
            'max_depth': randint(3, 10),
            'min_child_weight': randint(1, 10),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5)
        }
    }

    best_models = {}

    for model_name, model in models.items():
        if debug:
            print(f"\nDebug: Starting training for {model_name}")
            print("Debug: Parameter distribution for random search:")
            print(param_distributions[model_name])

        random_search = RandomizedSearchCV(
            model, 
            param_distributions=param_distributions[model_name],
            n_iter=100, 
            cv=5, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1, 
            random_state=42,
            error_score='raise'
        )
        random_search.fit(X_train_scaled, y_train)

        best_model = random_search.best_estimator_

        if debug:
            print(f"Debug: Best parameters for {model_name}: {random_search.best_params_}")

        best_models[model_name] = {'model': best_model}

    return best_models, scaler

def predict_for_season(data, season, features, label_encoders, salary_cap_column, model, scaler, debug=False):
    if debug:
        print(f"Debug: Predicting for season {season}")
        print(f"Debug: Available seasons in data: {data['Season'].unique()}")
        print(f"Debug: Features expected: {features}")
        print(f"Debug: Features in data: {data.columns.tolist()}")

    # Check if the season exists in the data
    if season not in data['Season'].unique():
        raise ValueError(f"Season {season} not found in the dataset. Available seasons are: {data['Season'].unique()}")

    # Filter data into training and testing sets based on the specified season
    test_data = data[data['Season'] == season]

    if debug:
        print(f"Debug: Test data shape: {test_data.shape}")

    if test_data.empty:
        raise ValueError(f"No data available for the season {season}. Please check the dataset and ensure it includes data for the specified season.")

    # Prepare test data
    X_test = test_data[features]
    
    if debug:
        print(f"Debug: X_test shape: {X_test.shape}")
        print(f"Debug: X_test columns: {X_test.columns.tolist()}")
        print(f"Debug: Scaler n_features_in_: {scaler.n_features_in_}")

    # Check for feature mismatch
    if X_test.shape[1] != scaler.n_features_in_:
        print("Warning: Feature mismatch detected.")
        print(f"Expected {scaler.n_features_in_} features, but got {X_test.shape[1]}.")
        print("Missing features:", set(scaler.feature_names_in_) - set(X_test.columns))
        print("Extra features:", set(X_test.columns) - set(scaler.feature_names_in_))
        
        # Adjust X_test to match scaler's expected features
        missing_features = set(scaler.feature_names_in_) - set(X_test.columns)
        for feature in missing_features:
            X_test[feature] = 0  # or some other appropriate default value
        
        X_test = X_test[scaler.feature_names_in_]

    # Scale the test data
    X_test_scaled = scaler.transform(X_test)

    # Make predictions using the provided model
    predictions = model.predict(X_test_scaled)

    # Add predictions to the test data
    test_data['Predicted_SalaryPct'] = predictions
    
    # Convert Predicted_SalaryPct back to Predicted_Salary
    test_data['Predicted_Salary'] = test_data['Predicted_SalaryPct'] * test_data[salary_cap_column]

    if debug:
        print("Debug: Salary predictions completed")
        print("Debug: First few rows of predicted salaries:")
        print(test_data[['Player', 'Salary', 'Predicted_Salary']].head())

    # Decode categorical columns
    for column, le in label_encoders.items():
        if column in test_data.columns:
            test_data[column] = le.inverse_transform(test_data[column].astype(int))

    if debug:
        print("Debug: Data with decoded categorical features:")
        print(test_data.head())

    return test_data

def load_pretrained_models(output_dir):
    models = {}
    try:
        for model_name in ['random_forest', 'gradient_boosting', 'xgboost']:
            model_path = os.path.join(output_dir, f'{model_name}_model.joblib')
            if os.path.exists(model_path):
                models[model_name] = {'model': joblib.load(model_path)}
        scaler = joblib.load(os.path.join(output_dir, 'scaler.joblib'))
        features = joblib.load(os.path.join(output_dir, 'features.joblib'))
        return models, scaler, features
    except Exception as e:
        print(f"Error loading pre-trained models: {e}")
        return None, None, None

def analyze_salary_predictions(data):
    data['Salary_Difference'] = data['Predicted_Salary'] - data['Salary']
    
    overpaid = data.sort_values(by='Salary_Difference', ascending=True).head(20)
    underpaid = data.sort_values(by='Salary_Difference', ascending=False).head(20)
    
    return overpaid, underpaid


def save_models(best_models, scaler, trained_features, output_dir):
    """
    Save the trained models, scaler, and features to the specified output directory.
    
    Args:
    best_models (dict): Dictionary containing the trained models
    scaler (StandardScaler): The fitted StandardScaler object
    trained_features (list): List of features used for training
    output_dir (str): Directory to save the models and related files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for model_name, model_info in best_models.items():
        model_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_model.joblib")
        joblib.dump(model_info['model'], model_path)
    
    scaler_path = os.path.join(output_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    
    features_path = os.path.join(output_dir, 'features.joblib')
    joblib.dump(trained_features, features_path)
    
    print(f"Models, scaler, and features saved to {output_dir}")
    
# Example usage
if __name__ == "__main__":
    file_path = '../data/processed/nba_player_data_final_inflated.csv'
    output_dir = '../data/models'  # Add this line to specify where models are saved
    print("Loading and preprocessing data...")
    data, salary_cap_column = load_and_preprocess_data(file_path, use_inflated_cap=True, debug=False)
    if data is not None:
        data_prepared, features, encoders = prepare_data_for_training(data, salary_cap_column, debug=True)
        print("Debug: Features after preparation:")
        print(features)
        print(f"Debug: Number of features: {len(features)}")
        
        # Load pre-trained models
        loaded_models, loaded_scaler, loaded_features = load_pretrained_models(output_dir)

        if loaded_models is None or loaded_scaler is None or loaded_features is None:
            print("Pre-trained models not found. Training new models...")
            best_models, scaler = train_models(data_prepared[features], data_prepared['SalaryPct'], debug=True)
            # Save the models here if needed
        else:
            best_models, scaler = loaded_models, loaded_scaler

        # Specify the season you want to predict
        season_to_predict = 2022
        print(f"\nPredicting salaries for season {season_to_predict}...")
        try:
            # Use the first model in best_models for prediction
            if isinstance(best_models, dict):
                model_name, model_info = next(iter(best_models.items()))
                if isinstance(model_info, dict) and 'model' in model_info:
                    model = model_info['model']
                else:
                    model = model_info  # Assume the value is the model itself
            else:
                model = best_models  # Assume best_models is the model itself

            predictions_for_season = predict_for_season(
                data_prepared, 
                season_to_predict, 
                features,  # This should now include 'Season'
                encoders, 
                salary_cap_column, 
                model,
                scaler,
                debug=True
            )

            # Analyze the predictions
            overpaid_players, underpaid_players = analyze_salary_predictions(predictions_for_season)
            print("Top 20 Overpaid Players:\n", overpaid_players[['Player', 'Salary', 'Predicted_Salary', 'Salary_Difference']])
            print("Top 20 Underpaid Players:\n", underpaid_players[['Player', 'Salary', 'Predicted_Salary', 'Salary_Difference']])
        except ValueError as e:
            print(e)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
