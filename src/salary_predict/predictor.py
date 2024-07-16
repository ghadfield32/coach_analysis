
import joblib
from data_loader import get_project_root
from data_preprocessor import feature_engineering
from sklearn.impute import SimpleImputer
import os

def load_model_and_scaler(model_name, inflated=False):
    root_dir = get_project_root()
    suffix = '_inflated' if inflated else ''
    
    # Convert model name to a consistent format
    model_name = model_name.replace(' ', '_')
    
    if 'Best' in model_name:
        model_file_name = f'{model_name}_salary_prediction_model{suffix}.joblib'
    else:
        model_file_name = f'{model_name}_salary_prediction_model{suffix}.joblib'
    
    model_path = os.path.join(root_dir, 'data', 'models', model_file_name)
    
    if not os.path.exists(model_path):
        # Try alternative naming conventions
        alternative_names = [
            f'{model_name.lower()}_salary_prediction_model{suffix}.joblib',
            f'{model_name.upper()}_salary_prediction_model{suffix}.joblib',
            f'{model_name.capitalize()}_salary_prediction_model{suffix}.joblib'
        ]
        
        for alt_name in alternative_names:
            alt_path = os.path.join(root_dir, 'data', 'models', alt_name)
            if os.path.exists(alt_path):
                model_path = alt_path
                break
        else:
            raise FileNotFoundError(f"The model file for '{model_name}' does not exist. Tried the following paths:\n"
                                    f"- {model_path}\n" + "\n".join(f"- {os.path.join(root_dir, 'data', 'models', name)}" for name in alternative_names))

    model = joblib.load(model_path)
    scaler = joblib.load(os.path.join(root_dir, 'data', 'models', f'scaler{suffix}.joblib'))
    selected_features = joblib.load(os.path.join(root_dir, 'data', 'models', f'selected_features{suffix}.joblib'))
    return model, scaler, selected_features

def make_predictions(df, model, scaler, selected_features, season, use_inflated_data, max_salary_cap):
    df = df[df['Season'] == season].copy()
    df = feature_engineering(df)
    df['Age'] += 1
    df['Season'] += 1
    
    if not all(feature in df.columns for feature in selected_features):
        missing_features = [f for f in selected_features if f not in df.columns]
        raise ValueError(f"Missing features in dataframe: {missing_features}")
    
    X = df[selected_features]
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.transform(X_imputed)
    
    df.loc[:, 'Predicted_Salary_Pct'] = model.predict(X_scaled)
    
    salary_cap_column = 'Salary_Cap_Inflated' if use_inflated_data else 'Salary Cap'
    
    if salary_cap_column not in df.columns:
        raise ValueError(f"Salary cap column '{salary_cap_column}' not found in dataframe")
    
    df.loc[:, 'Predicted_Salary'] = df['Predicted_Salary_Pct'] * df[salary_cap_column]
    df.loc[:, 'Salary_Change'] = df['Predicted_Salary'] - df['Salary']
    
    return df
