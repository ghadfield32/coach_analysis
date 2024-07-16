
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import joblib

def create_ensemble_model(selected_models, param_grid):
    models = {
        'XGB': XGBClassifier(random_state=42, eval_metric='logloss'),
        'LGBM': LGBMClassifier(random_state=42),
        'RF': RandomForestClassifier(random_state=42),
        'MLP': MLPClassifier(random_state=42, max_iter=1000)
    }

    ensemble_estimators = [(name.lower(), models[name]) for name in selected_models]
    
    ensemble = VotingClassifier(
        estimators=ensemble_estimators,
        voting='soft'
    )
    
    # Adapt param_grid to selected models
    grid_params = {key: param_grid[key] for key in param_grid if any(key.startswith(model.lower()) for model in selected_models)}
    
    grid_search = GridSearchCV(ensemble, grid_params, cv=3, scoring='roc_auc', error_score='raise')
    return grid_search

def train_model(X, y, selected_models, param_grid):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train_scaled)
    X_test_imputed = imputer.transform(X_test_scaled)
    
    ensemble = create_ensemble_model(selected_models, param_grid)
    
    try:
        ensemble.fit(X_train_imputed, y_train)
    except ValueError as e:
        print(f"Error during model training: {e}")
        return None, None, None
    
    scores = cross_val_score(ensemble, X_train_imputed, y_train, cv=3, scoring='roc_auc')
    print(f'Cross-Validation Scores: {scores}')
    print(f'Mean CV Score: {np.mean(scores)}')
    
    try:
        y_pred = ensemble.predict(X_test_imputed)
        y_pred_proba = ensemble.predict_proba(X_test_imputed)[:, 1]
    except ValueError as e:
        print(f"Error during prediction: {e}")
        return None, None, None
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    print_metrics(metrics)
    
    return ensemble, scaler, imputer

def calculate_metrics(y_true, y_pred, y_pred_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }

def print_metrics(metrics):
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

def save_model(model, scaler, imputer, filename):
    joblib.dump({'model': model, 'scaler': scaler, 'imputer': imputer}, filename)

def load_model(filename):
    model_data = joblib.load(filename)
    return model_data['model'], model_data['scaler'], model_data['imputer']
