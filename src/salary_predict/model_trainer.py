import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score

def load_data(inflated=False, debug=False):
    root_dir = get_project_root()
    file_name = 'nba_player_data_final_inflated.csv'
    file_path = os.path.join(root_dir, 'data', 'processed', file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please check the file path.")
    
    data = pd.read_csv(file_path)
    
    if inflated:
        data.drop(columns=['Salary Cap'], inplace=True, errors='ignore')
        salary_cap_column = 'Salary_Cap_Inflated'
    else:
        data.drop(columns=['Salary_Cap_Inflated'], inplace=True, errors='ignore')
        salary_cap_column = 'Salary Cap'

    # Convert 'Season' to the correct format
    data['Season'] = data['Season'].str[:4].astype(int)
    
    # Handle missing values for numerical columns
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
    data['Availability'] = data['GP'] / 82
    data['SalaryPct'] = data['Salary'] / data[salary_cap_column]

    if debug:
        print("Debug: Data shape after preprocessing:", data.shape)
        print("Debug: Columns after preprocessing:", data.columns)
        print("Debug: First few rows of preprocessed data:")
        print(data.head())

    return data, salary_cap_column

# Usage:
data, salary_cap_column = load_data(inflated=False, debug=True)

def prepare_data_for_training(data, salary_cap_column, debug=False):
    # Using Label Encoding for categorical columns
    label_encoders = {}
    for column in ['Player', 'Season', 'Position', 'Team']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    
    initial_features = ['Age', 'Years of Service', 'GP', 'PPG', 'APG', 'RPG', 'SPG', 'BPG', 'TOPG', 'FG%', '3P%', 'FT%', 'PER', 'WS', 'VORP', 'Availability', 'Player', 'Season', 'Position', 'Team']

    data_subset = data[initial_features + ['SalaryPct', 'Salary']].copy()
    data_cleaned = data_subset.dropna()

    if debug:
        print("Debug: Data shape after cleaning:", data_cleaned.shape)
        print("Debug: Selected features:", initial_features)

    return data_cleaned, initial_features, label_encoders

# Usage:
data_cleaned, initial_features, label_encoders = prepare_data_for_training(data, salary_cap_column, debug=True)

def save_models(models, scaler, selected_features, inflated=False):
    root_dir = get_project_root()
    suffix = '_inflated' if inflated else ''
    model_name_mapping = {
        'Random_Forest': 'Random_Forest',
        'Gradient_Boosting': 'Gradient_Boosting',
        'Ridge_Regression': 'Ridge_Regression',
        'ElasticNet': 'ElasticNet',
        'SVR': 'SVR',
        'Decision_Tree': 'Decision_Tree'
    }
    for name, model in models.items():
        try:
            formatted_name = model_name_mapping.get(name, name)
            joblib.dump(model, os.path.join(root_dir, 'data', 'models', f'{formatted_name}_salary_prediction_model{suffix}.joblib'))
        except Exception as e:
            print(f"Error saving model {name}: {str(e)}")
    
    
    try:
        joblib.dump(scaler, os.path.join(root_dir, 'data', 'models', f'scaler{suffix}.joblib'))
        joblib.dump(selected_features, os.path.join(root_dir, 'data', 'models', f'selected_features{suffix}.joblib'))
    except Exception as e:
        print(f"Error saving scaler or selected features: {str(e)}")
    
def retrain_and_save_models(use_inflated_data, debug=False):
    data, salary_cap_column = load_data(use_inflated_data, debug)
    data_cleaned, initial_features, label_encoders = prepare_data_for_training(data, salary_cap_column, debug)

    X = data_cleaned[initial_features]
    y = data_cleaned['SalaryPct']

    # Feature selection using mutual information
    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, index=initial_features)
    top_features = mi_scores.nlargest(15).index.tolist()  # Select top 15 features

    if debug:
        print("Debug: Top 15 features based on mutual information:")
        print(mi_scores.nlargest(15))

    X = X[top_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models and their parameter grids
    models = {
        'Random_Forest': (RandomForestRegressor(random_state=42), {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }),
        'Gradient_Boosting': (GradientBoostingRegressor(random_state=42), {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5]
        }),
        'Ridge_Regression': (Ridge(), {
            'alpha': [0.1, 1.0, 10.0]
        })
    }

    best_models = {}
    evaluations = {}
    feature_importances = {}

    for name, (model, param_grid) in models.items():
        if debug:
            print(f"Debug: Training {name}...")
        
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        
        best_model = grid_search.best_estimator_
        best_models[name] = best_model
        
        y_pred = best_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Perform cross-validation for the best model
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_mse = -cv_scores.mean()
        
        evaluations[name] = {
            "Test MSE": mse,
            "Test R²": r2,
            "CV MSE": cv_mse,
            "Best Params": grid_search.best_params_
        }
        
        if debug:
            print(f"Debug: {name} - Test MSE: {mse:.4f}, R²: {r2:.4f}")
            print(f"Debug: {name} - CV MSE: {cv_mse:.4f}")
            print(f"Debug: {name} - Best Parameters: {grid_search.best_params_}")

        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
        else:
            importances = permutation_importance(best_model, X_test_scaled, y_test, n_repeats=10, random_state=42).importances_mean

        feature_importances[name] = dict(zip(top_features, importances))
        
        if debug:
            print(f"Debug: {name} - Top 5 important features:")
            print(sorted(feature_importances[name].items(), key=lambda x: x[1], reverse=True)[:5])

    # Identify the best overall model
    best_model_name = min(evaluations, key=lambda x: evaluations[x]['Test MSE'])
    best_model = best_models[best_model_name]

    if debug:
        print(f"Debug: Best overall model: {best_model_name}")

    # Save only the best model and related information
    save_models({best_model_name: best_model}, scaler, top_features, use_inflated_data)

    return best_model_name, best_model, evaluations, top_features, scaler, data[salary_cap_column].max(), feature_importances

# Usage:
best_model_name, best_model, evaluations, selected_features, scaler, max_salary_cap, feature_importances = retrain_and_save_models(use_inflated_data=False, debug=True)


