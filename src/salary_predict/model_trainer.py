
import pandas as pd
import numpy as np
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
from data_loader import get_project_root, load_data
import os

def retrain_models(X, y, model_params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    models = {}
    for name, (model, params) in model_params.items():
        grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(X_train_scaled, y_train)
        models[name] = grid_search.best_estimator_
    
    return models, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_models(models, scaler, selected_features, inflated=False):
    root_dir = get_project_root()
    suffix = '_inflated' if inflated else ''
    model_name_mapping = {
        'Random Forest': 'Random_Forest',
        'Gradient Boosting': 'Gradient_Boosting',
        'Ridge Regression': 'Ridge_Regression',
        'ElasticNet': 'ElasticNet',
        'SVR': 'SVR',
        'Decision Tree': 'Decision_Tree'
    }
    for name, model in models.items():
        formatted_name = model_name_mapping[name]
        joblib.dump(model, os.path.join(root_dir, 'data', 'models', f'{formatted_name}_salary_prediction_model{suffix}.joblib'))
    joblib.dump(scaler, os.path.join(root_dir, 'data', 'models', f'scaler{suffix}.joblib'))
    joblib.dump(selected_features, os.path.join(root_dir, 'data', 'models', f'selected_features{suffix}.joblib'))

def evaluate_models(models, X_test, y_test):
    evaluations = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        evaluations[name] = {"MSE": mse, "R²": r2}
    return evaluations

def retrain_and_save_models(use_inflated_data):
    # Load the appropriate data
    data = load_data(use_inflated_data)
    
    # Drop unnecessary columns
    if use_inflated_data:
        data.drop(columns=['2022 Dollars', 'Salary Cap'], inplace=True)
        salary_cap_column = 'Salary_Cap_Inflated'
    else:
        data.drop(columns=['2022 Dollars', 'Salary_Cap_Inflated'], inplace=True)
        salary_cap_column = 'Salary Cap'

    # Convert 'Season' to an integer
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
    data['SalaryGrowth'] = data['Salary'].pct_change().fillna(0)
    data['Availability'] = data['GP'] / 82
    data['SalaryPct'] = data['Salary'] / data[salary_cap_column]

    # Identify categorical and numerical columns
    categorical_cols = ['Player', 'Season', 'Position', 'Team']
    numerical_cols = data.columns.difference(categorical_cols + ['Salary', 'SalaryPct', salary_cap_column])

    # One-hot encode categorical variables
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_cats = pd.DataFrame(encoder.fit_transform(data[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))

    # Combine the numerical and encoded categorical data
    data = pd.concat([data[numerical_cols], encoded_cats, data[['Player', 'Season', 'Salary', 'SalaryPct', salary_cap_column]]], axis=1)

    # Select initial features
    initial_features = ['Age', 'Years of Service', 'GP', 'PPG', 'APG', 'RPG', 'SPG', 'BPG', 'TOPG', 'FG%', '3P%', 'FT%', 'PER', 'WS', 'VORP', 'Availability'] + list(encoded_cats.columns)

    # Create a new DataFrame with only the features we're interested in and the target variable
    data_subset = data[initial_features + ['SalaryPct']].copy()

    # Drop rows with any missing values
    data_cleaned = data_subset.dropna()

    # Separate features and target variable
    X = data_cleaned[initial_features]
    y = data_cleaned['SalaryPct']

    # Perform feature selection
    rfe = RFE(estimator=RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=10)
    rfe = rfe.fit(X, y)
    selected_features = [feature for feature, selected in zip(initial_features, rfe.support_) if selected]

    print("Selected features by RFE:", selected_features)

    X = data_cleaned[selected_features]
    y = data_cleaned['SalaryPct']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models with updated parameters
    models = {
        'Random_Forest': RandomForestRegressor(random_state=42),
        'Gradient_Boosting': GradientBoostingRegressor(random_state=42),
        'Ridge_Regression': Ridge(),
        'ElasticNet': ElasticNet(max_iter=10000),
        'SVR': SVR(),
        'Decision_Tree': DecisionTreeRegressor(random_state=42)
    }

    # Define parameter grids
    param_grids = {
        'Random_Forest': {
            'n_estimators': [50, 100, 200],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [8, 10, 12],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [1, 2, 4]
        },
        'Gradient_Boosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        },
        'Ridge_Regression': {'alpha': [0.1, 1.0, 10.0, 100.0]},
        'ElasticNet': {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]},
        'SVR': {'C': [0.1, 1, 10], 'epsilon': [0.1, 0.2, 0.5]},
        'Decision_Tree': {'max_depth': [6, 8, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
    }

    # Train and evaluate models
    best_models = {}
    evaluations = {}
    for name, model in models.items():
        print(f"Training {name}...")
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(X_train_scaled, y_train)
        best_models[name] = grid_search.best_estimator_
        
        # Cross-validation
        cv_scores = cross_val_score(best_models[name], X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
        print(f"{name} - Best params: {grid_search.best_params_}")
        print(f"{name} - Cross-validation MSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Test set performance
        y_pred = best_models[name].predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name} - Test MSE: {mse:.4f}, R²: {r2:.4f}")
        
        evaluations[name] = {"MSE": mse, "R²": r2}
        
        # Feature importance
        if name in ['Random_Forest', 'Gradient_Boosting', 'Decision_Tree']:
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
        
        # Save the model
        root_dir = get_project_root()
        suffix = '_inflated' if use_inflated_data else ''
        model_filename = os.path.join(root_dir, 'data', 'models', f'{name}_salary_prediction_model{suffix}.joblib')
        joblib.dump(best_models[name], model_filename)
        print(f"{name} model saved to '{model_filename}'")

    # Identify the best overall model
    best_model_name = min(evaluations, key=lambda x: evaluations[x]['MSE'])
    best_model = best_models[best_model_name]

    print(f"Best overall model: {best_model_name}")

    # Save the scaler, selected features, and best model name
    root_dir = get_project_root()
    suffix = '_inflated' if use_inflated_data else ''
    
    scaler_filename = os.path.join(root_dir, 'data', 'models', f'scaler{suffix}.joblib')
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler saved to '{scaler_filename}'")
    
    selected_features_filename = os.path.join(root_dir, 'data', 'models', f'selected_features{suffix}.joblib')
    joblib.dump(selected_features, selected_features_filename)
    print(f"Selected features saved to '{selected_features_filename}'")
    
    with open(os.path.join(root_dir, 'data', 'models', f'best_model_name{suffix}.txt'), 'w') as f:
        f.write(best_model_name)

    return best_model_name, best_model, evaluations, selected_features, scaler, data[salary_cap_column].max()

