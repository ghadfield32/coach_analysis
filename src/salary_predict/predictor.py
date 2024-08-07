
import joblib
# from data_loader import get_project_root
# from data_preprocessor import feature_engineering
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

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def make_predictions(df, model, scaler, selected_features, season, use_inflated_data, max_salary_cap):
    df = df[df['Season'] == season].copy()
    df = feature_engineering(df, use_inflated_data)
    df['Age'] += 1
    df['Season'] += 1
   
    if not all(feature in df.columns for feature in selected_features):
        missing_features = [f for f in selected_features if f not in df.columns]
        raise ValueError(f"Missing features in dataframe: {missing_features}")
   
    X = df[selected_features]
   
    # Separate numeric and non-numeric columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(exclude=['int64', 'float64']).columns
   
    # Handle numeric features
    numeric_imputer = SimpleImputer(strategy='mean')
    X_numeric_imputed = numeric_imputer.fit_transform(X[numeric_features])
   
    # Handle categorical features
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    X_categorical_imputed = categorical_imputer.fit_transform(X[categorical_features])
   
    # Encode categorical features
    label_encoders = {}
    X_categorical_encoded = np.zeros_like(X_categorical_imputed)
    for i, feature in enumerate(categorical_features):
        le = LabelEncoder()
        X_categorical_encoded[:, i] = le.fit_transform(X_categorical_imputed[:, i])
        label_encoders[feature] = le
   
    # Combine numeric and encoded categorical features
    X_combined = np.hstack((X_numeric_imputed, X_categorical_encoded))
   
    # Scale the features
    X_scaled = scaler.transform(X_combined)
   
    df.loc[:, 'Predicted_Salary_Pct'] = model.predict(X_scaled)
   
    salary_cap_column = 'Salary_Cap_Inflated' if use_inflated_data else 'Salary Cap'
   
    if salary_cap_column not in df.columns:
        raise ValueError(f"Salary cap column '{salary_cap_column}' not found in dataframe")
   
    df.loc[:, 'Predicted_Salary'] = df['Predicted_Salary_Pct'] * df[salary_cap_column]
    df.loc[:, 'Salary_Change'] = df['Predicted_Salary'] - df['Salary']
   
    return df


if __name__ == "__main__":
    # from data_loader import load_data
   
    print("Loading model...")
    model, scaler, selected_features = load_model_and_scaler('Random_Forest', inflated=False)
   
    print("\nLoading data...")
    df, salary_cap_column = load_data(inflated=False)  # Unpack both return values
   
    print("\nMaking predictions...")
    season = df['Season'].max()
    predictions = make_predictions(df, model, scaler, selected_features, season, use_inflated_data=False, max_salary_cap=df[salary_cap_column].max())
   
    print("\nPredictions shape:", predictions.shape)
    print("\nFirst few rows of predictions:")
    print(predictions[['Player', 'Salary', 'Predicted_Salary', 'Salary_Change']].head())
   
    print("\nNaN values in predictions:")
    print(predictions.isna().sum())
