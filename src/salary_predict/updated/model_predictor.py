
import joblib
import pandas as pd

def load_models_and_utils(model_save_path):
    rf_model = joblib.load(f"{model_save_path}/best_rf_model.pkl")
    xgb_model = joblib.load(f"{model_save_path}/best_xgb_model.pkl")
    scaler = joblib.load(f"{model_save_path}/scaler.pkl")
    feature_names = joblib.load(f"{model_save_path}/feature_names.pkl")
    encoders = joblib.load(f"{model_save_path}/encoders.pkl")
    injury_risk_mapping = joblib.load(f"{model_save_path}/injury_risk_mapping.pkl")
    numeric_cols = joblib.load(f"{model_save_path}/numeric_cols.pkl")
    player_encoder = joblib.load(f"{model_save_path}/player_encoder.pkl")
    return rf_model, xgb_model, scaler, feature_names, encoders, injury_risk_mapping, numeric_cols, player_encoder

def predict(data, model_save_path):
    rf_model, xgb_model, scaler, feature_names, encoders, _, _, player_encoder = load_models_and_utils(model_save_path)
    
    print("Original data shape:", data.shape)
    print("Original data columns:", data.columns.tolist())

    # Preserve player names
    player_names = data['Player'] if 'Player' in data.columns else None
    
    # Drop the player column before encoding
    data = data.drop(columns=['Player'], errors='ignore')
    
    # Encode the data using the loaded encoders
    encoded_data, _, _, _, _, _ = encode_data(data, encoders, player_encoder)
    
    print("Encoded data shape:", encoded_data.shape)
    print("Encoded data columns:", encoded_data.columns.tolist())
    
    # Handle missing features: Add missing columns and set them to zero
    for col in feature_names:
        if col not in encoded_data.columns:
            encoded_data[col] = 0

    # Ensure encoded_data only has feature_names columns
    encoded_data = encoded_data[feature_names]
    
    print("Selected features shape:", encoded_data.shape)
    print("Selected features:", encoded_data.columns.tolist())
    print("Expected features:", feature_names)
    
    # Scale the encoded data
    encoded_data_scaled = scaler.transform(encoded_data)
    
    # Make predictions
    rf_predictions = rf_model.predict(encoded_data_scaled)
    xgb_predictions = xgb_model.predict(encoded_data_scaled)
    
    # Create a DataFrame for predictions
    predictions_df = pd.DataFrame({
        'RF_Predictions': rf_predictions,
        'XGB_Predictions': xgb_predictions,
        'Predicted_Salary': (rf_predictions + xgb_predictions) / 2
    })
    
    # Attach player names back to the predictions
    if player_names is not None:
        predictions_df['Player'] = player_names.values

    # Combine the predictions with the original data (excluding player names)
    result = pd.concat([data.reset_index(drop=True), predictions_df], axis=1)

    return result


if __name__ == "__main__":
    file_path = '../data/processed/nba_player_data_final_inflated.csv'
    predict_season = 2023
    data = load_data(file_path)
    data = format_season(data)
    current_season_data, _ = filter_seasons(data, predict_season)
    current_season_data = clean_data(current_season_data)
    current_season_data = engineer_features(current_season_data)
    model_save_path = '../data/models'
    predictions_df = predict(current_season_data, model_save_path)  # Save predictions as predictions_df
    print(predictions_df.head())
    
    # Save predictions_df for later use
    predictions_df.to_csv('../data/processed/predictions_df.csv', index=False)
