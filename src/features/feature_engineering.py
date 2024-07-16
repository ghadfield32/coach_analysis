import numpy as np
import pandas as pd

def engineer_features(shots_data):
    """Engineer features for the shot data."""
    shots_data = shots_data.copy()
    
    # Create binary target variable
    shots_data['SHOT_MADE'] = shots_data['SHOT_MADE_FLAG']
    
    # Advanced Feature Engineering
    shots_data['SHOT_CLOCK'] = shots_data['MINUTES_REMAINING'] * 60 + shots_data['SECONDS_REMAINING']
    shots_data['DISTANCE_FROM_CENTER'] = np.sqrt(shots_data['LOC_X']**2 + shots_data['LOC_Y']**2)
    shots_data['ANGLE'] = np.arctan2(shots_data['LOC_Y'], shots_data['LOC_X'])
    shots_data['QUARTER_TIME'] = shots_data['MINUTES_REMAINING'] * 60 + shots_data['SECONDS_REMAINING']
    shots_data['GAME_TIME'] = (shots_data['PERIOD'] - 1) * 720 + (720 - shots_data['QUARTER_TIME'])
    
    # Create player-specific features (example: rolling average of last 5 shots)
    shots_data = shots_data.sort_values(['PLAYER_NAME', 'GAME_ID', 'GAME_TIME'])
    shots_data['PLAYER_ROLLING_AVG'] = shots_data.groupby('PLAYER_NAME')['SHOT_MADE'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    # One-hot encode categorical variables
    categorical_features = ['SHOT_ZONE_AREA', 'SHOT_ZONE_BASIC', 'SHOT_ZONE_RANGE', 'PLAYER_NAME']
    shots_data_encoded = pd.get_dummies(shots_data, columns=categorical_features)
    
    return shots_data_encoded

def prepare_features(shots_data_encoded):
    """Prepare features for model training."""
    features = ['PERIOD', 'SHOT_CLOCK', 'SHOT_DISTANCE', 'LOC_X', 'LOC_Y', 
                'DISTANCE_FROM_CENTER', 'ANGLE', 'QUARTER_TIME', 'GAME_TIME',
                'PLAYER_ROLLING_AVG'] + [col for col in shots_data_encoded.columns 
                                         if col.startswith(('SHOT_ZONE_', 'PLAYER_NAME_'))]
    
    X = shots_data_encoded[features]
    y = shots_data_encoded['SHOT_MADE']
    
    return X, y
