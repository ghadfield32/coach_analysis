import pandas as np

def predict_shot(model, scaler, imputer, shot_data, feature_columns):
    shot_data = shot_data.copy()
    
    # Prepare features (same as in prepare_data function)
    shot_data['SHOT_CLOCK'] = shot_data['MINUTES_REMAINING'] * 60 + shot_data['SECONDS_REMAINING']
    shot_data['DISTANCE_FROM_CENTER'] = np.sqrt(shot_data['LOC_X']**2 + shot_data['LOC_Y']**2)
    shot_data['ANGLE'] = np.arctan2(shot_data['LOC_Y'], shot_data['LOC_X'])
    shot_data['QUARTER_TIME'] = shot_data['MINUTES_REMAINING'] * 60 + shot_data['SECONDS_REMAINING']
    shot_data['GAME_TIME'] = (shot_data['PERIOD'] - 1) * 720 + (720 - shot_data['QUARTER_TIME'])
    
    # For simplicity, we'll use a constant value for PLAYER_ROLLING_AVG in this example
    shot_data['PLAYER_ROLLING_AVG'] = 0.5
    
    categorical_features = ['SHOT_ZONE_AREA', 'SHOT_ZONE_BASIC', 'SHOT_ZONE_RANGE', 'PLAYER_NAME']
    shot_data_encoded = pd.get_dummies(shot_data, columns=categorical_features)
    
    features = ['PERIOD', 'SHOT_CLOCK', 'SHOT_DISTANCE', 'LOC_X', 'LOC_Y', 
                'DISTANCE_FROM_CENTER', 'ANGLE', 'QUARTER_TIME', 'GAME_TIME',
                'PLAYER_ROLLING_AVG'] + [col for col in shot_data_encoded.columns 
                                         if col.startswith(tuple(categorical_features))]
    
    X_new = shot_data_encoded[features]
    
    # Align new data with training features
    X_new = X_new.reindex(columns=feature_columns, fill_value=0)
    
    X_new_scaled = scaler.transform(X_new)
    X_new_imputed = imputer.transform(X_new_scaled)
    
    shot_data['PREDICTION'] = model.predict(X_new_imputed)
    shot_data['PREDICTION_PROB'] = model.predict_proba(X_new_imputed)[:, 1]
    
    return shot_data
