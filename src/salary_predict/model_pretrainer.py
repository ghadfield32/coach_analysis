
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
import joblib
import os
from data_loader import get_project_root, load_data
from data_preprocessor import handle_missing_values, feature_engineering

def pretrain_models(X, y):
    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Ridge Regression': Ridge(),
        'ElasticNet': ElasticNet(max_iter=10000),
        'SVR': SVR(),
        'Decision Tree': DecisionTreeRegressor(random_state=42)
    }

    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [8, 10, 12],
            'min_samples_split': [5, 10, 15]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5]
        },
        'Ridge Regression': {'alpha': [0.1, 1.0, 10.0]},
        'ElasticNet': {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]},
        'SVR': {'C': [0.1, 1, 10], 'epsilon': [0.1, 0.2, 0.5]},
        'Decision Tree': {'max_depth': [6, 8, 10], 'min_samples_split': [2, 5, 10]}
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pretrained_models = {}
    for name, model in models.items():
        print(f"Pretraining {name}...")
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(X_train_scaled, y_train)
        pretrained_models[name] = grid_search.best_estimator_

    return pretrained_models, X_test_scaled, y_test, scaler

def evaluate_pretrained_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE': mse, 'R2': r2}
    return results

def save_best_pretrained_model(models, results, scaler, selected_features, use_inflated_data):
    best_model_name = min(results, key=lambda x: results[x]['MSE'])
    best_model = models[best_model_name]
    
    root_dir = get_project_root()
    suffix = '_inflated' if use_inflated_data else ''
    
    joblib.dump(best_model, os.path.join(root_dir, 'data', 'models', f'{best_model_name}_best_pretrained_salary_prediction_model{suffix}.joblib'))
    joblib.dump(scaler, os.path.join(root_dir, 'data', 'models', f'pretrained_scaler{suffix}.joblib'))
    joblib.dump(selected_features, os.path.join(root_dir, 'data', 'models', f'pretrained_selected_features{suffix}.joblib'))

    print(f"Best pretrained model ({best_model_name}) saved successfully.")
    return best_model_name, best_model

def pretrain_and_save_models(use_inflated_data):
    df = load_data(use_inflated_data)
    df = feature_engineering(df)
    df = handle_missing_values(df)

    # Feature selection
    initial_features = ['Age', 'Years of Service', 'GP', 'PPG', 'APG', 'RPG', 'SPG', 'BPG', 'TOPG', 'FG%', '3P%', 'FT%', 'PER', 'WS', 'VORP', 'Availability']
    X = df[initial_features]
    y = df['SalaryPct']

    rfe = RFE(estimator=RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=10)
    rfe = rfe.fit(X, y)
    selected_features = [feature for feature, selected in zip(initial_features, rfe.support_) if selected]

    X = df[selected_features]
    y = df['SalaryPct']

    pretrained_models, X_test, y_test, scaler = pretrain_models(X, y)
    results = evaluate_pretrained_models(pretrained_models, X_test, y_test)
    best_model_name, best_model = save_best_pretrained_model(pretrained_models, results, scaler, selected_features, use_inflated_data)

    return best_model_name, best_model, results
