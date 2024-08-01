
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import commonallplayers, commonplayerinfo, playercareerstats, leaguestandings
import time
from requests.exceptions import RequestException
from json.decoder import JSONDecodeError
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
import joblib
from sklearn.inspection import permutation_importance
from fetch_utils import fetch_with_retry
from scrape_utils import scrape_player_salary_data, scrape_team_salary_data, scrape_salary_cap_history
from player_utils import fetch_all_players, process_player_data

# Import other necessary functions (fetch_with_retry, scrape_functions, etc.)

def load_and_preprocess_data(file_path, use_inflated_cap=True):
    data = pd.read_csv(file_path)
    
    if use_inflated_cap:
        data.drop(columns=['2022 Dollars', 'Salary Cap'], inplace=True)
        salary_cap_column = 'Salary_Cap_Inflated'
    else:
        data.drop(columns=['2022 Dollars', 'Salary_Cap_Inflated'], inplace=True)
        salary_cap_column = 'Salary Cap'

    data['Season'] = data['Season'].str[:4].astype(int)

    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy='mean')
    data[numerical_cols] = imputer.fit_transform(data[numerical_cols])

    # Feature engineering
    data['PPG'] = data['PTS'] / data['GP']
    data['APG'] = data['AST'] / data['GP']
    data['RPG'] = data['TRB'] / data['GP']
    data['SPG'] = data['STL'] / data['GP']
    data['BPG'] = data['BLK'] / data['GP']
    data['TOPG'] = data['TOV'] / data['GP']
    data['WinPct'] = data['Wins'] / (data['Wins'] + data['Losses'])
    data['SalaryGrowth'] = data['Salary'].pct_change().fillna(0)
    data['Availability'] = data['GP'] / 82
    data['SalaryPct'] = data['Salary'] / data[salary_cap_column]

    return data, salary_cap_column

def prepare_data_for_training(data, salary_cap_column):
    categorical_cols = ['Player', 'Season', 'Position', 'Team']
    numerical_cols = data.columns.difference(categorical_cols + ['Salary', 'SalaryPct'])

    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_cats = pd.DataFrame(encoder.fit_transform(data[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))

    data = pd.concat([data[numerical_cols], encoded_cats, data[['Player', 'Season', 'Salary', 'SalaryPct']]], axis=1)

    initial_features = ['Age', 'Years of Service', 'GP', 'PPG', 'APG', 'RPG', 'SPG', 'BPG', 'TOPG', 'FG%', '3P%', 'FT%', 'PER', 'WS', 'VORP', 'Availability'] + list(encoded_cats.columns)

    data_subset = data[initial_features + ['SalaryPct', 'Salary']].copy()
    data_cleaned = data_subset.dropna()

    return data_cleaned, initial_features

def train_models(data_cleaned, initial_features, target_variable='SalaryPct', n_features_to_select=10):
    X = data_cleaned[initial_features]
    y = data_cleaned[target_variable]

    rfe = RFE(estimator=RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=n_features_to_select)
    rfe = rfe.fit(X, y)
    selected_features = [feature for feature, selected in zip(initial_features, rfe.support_) if selected]

    print("Selected features by RFE:", selected_features)

    X = data_cleaned[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Random_Forest': RandomForestRegressor(random_state=42),
        'Gradient_Boosting': GradientBoostingRegressor(random_state=42),
        'Ridge_Regression': Ridge(),
        'ElasticNet': ElasticNet(max_iter=10000),
        'SVR': SVR(),
        'Decision_Tree': DecisionTreeRegressor(random_state=42)
    }

    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [8, 10, 12],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        },
        'Ridge Regression': {'alpha': [0.1, 1.0, 10.0, 100.0]},
        'ElasticNet': {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]},
        'SVR': {'C': [0.1, 1, 10], 'epsilon': [0.1, 0.2, 0.5]},
        'Decision Tree': {'max_depth': [6, 8, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
    }

    best_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(X_train_scaled, y_train)
        best_models[name] = grid_search.best_estimator_
        
        cv_scores = cross_val_score(best_models[name], X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
        print(f"{name} - Best params: {grid_search.best_params_}")
        print(f"{name} - Cross-validation MSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        y_pred = best_models[name].predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name} - Test MSE: {mse:.4f}, R²: {r2:.4f}")
        
        if name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
            importances = best_models[name].feature_importances_
            feature_importance = pd.DataFrame({'feature': selected_features, 'importance': importances})
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            print(f"\n{name} - Top 5 important features:")
            print(feature_importance.head())
        else:
            perm_importance = permutation_importance(best_models[name], X_test_scaled, y_test, n_repeats=10, random_state=42)
            feature_importance = pd.DataFrame({'feature': selected_features, 'importance': perm_importance.importances_mean})
            feature_importance = feature_importance.sort_values('importance', ascending=False)
            print(f"\n{name} - Top 5 important features (Permutation Importance):")
            print(feature_importance.head())
        
        model_filename = f"../data/models/{name}_salary_prediction_model_{'inflated' if target_variable == 'SalaryPct' else 'regular'}.joblib"
        joblib.dump(best_models[name], model_filename)
        print(f"{name} model saved to '{model_filename}'")

    return best_models, scaler, selected_features

def main():
    start_year = 2024  # Current year
    end_year = 2022    # Adjust as needed
    processed_file_path = '../data/processed/nba_player_data.csv'
    salary_cap_file_path = '../data/processed/salary_cap_history.csv'

    player_filter = input("Enter player name or 'all' for all players: ").strip().lower()
    min_avg_minutes = None
    if player_filter == 'all':
        min_avg_minutes = float(input("Enter the minimum average minutes per game (default 25 mins): ") or 25)

    existing_data = load_existing_data(processed_file_path)

    try:
        print(f"Updating data for years {start_year} to {end_year}")
        updated_data = update_data(existing_data, start_year, end_year, player_filter, min_avg_minutes)
        
        if not updated_data.equals(existing_data):
            print("New data retrieved. Merging with existing data...")
            
            print("Fetching salary cap data...")
            salary_cap_data = scrape_salary_cap_history()
            
            if salary_cap_data is not None:
                print("Salary cap data successfully retrieved.")
                
                salary_cap_data.to_csv(salary_cap_file_path, index=False)
                print(f"Salary cap data saved to {salary_cap_file_path}")
                
                print("Merging salary cap data with player data...")
                
                salary_cap_columns = [col for col in updated_data.columns if 'Salary Cap' in col]
                if salary_cap_columns:
                    print(f"Removing existing Salary Cap columns: {salary_cap_columns}")
                    updated_data = updated_data.drop(columns=salary_cap_columns)
                    
                updated_data = pd.merge(updated_data, salary_cap_data[['Season', 'Salary Cap']], on='Season', how='left')
                
                print("Final data shape:", updated_data.shape)
                print("Final data columns:", updated_data.columns)
            else:
                print("Warning: Failed to retrieve salary cap data. Skipping merge.")

            updated_data.to_csv(processed_file_path, index=False)
            print(f"Updated data saved to {processed_file_path}")
        else:
            print("No new data to save. The dataset is already up-to-date.")

        # Ask if user wants to train models
        train_models_option = input("Do you want to train the models? (yes/no): ").strip().lower()
        if train_models_option == 'yes':
            use_inflated_cap = input("Use inflated salary cap? (yes/no): ").strip().lower() == 'yes'
            target_variable = input("Choose target variable (SalaryPct/Salary): ").strip()
            
            data, salary_cap_column = load_and_preprocess_data(processed_file_path, use_inflated_cap)
            data_cleaned, initial_features = prepare_data_for_training(data, salary_cap_column)
            best_models, scaler, selected_features = train_models(data_cleaned, initial_features, target_variable)
            
            print("Model training completed.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
