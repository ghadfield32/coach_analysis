
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from nba_shots import fetch_all_shots_data
from nba_helpers import categorize_shot

def preprocess_shots_data(shots):
    """Preprocesses shots data for model training."""
    shots['SHOT_ZONE_BASIC'], shots['SHOT_ZONE_RANGE'] = zip(*shots.apply(categorize_shot, axis=1))
    shots = shots.dropna(subset=['LOC_X', 'LOC_Y', 'SHOT_CLOCK', 'GAME_CLOCK'])
    
    feature_columns = ['LOC_X', 'LOC_Y', 'SHOT_DIST', 'PERIOD', 'MINUTES_REMAINING', 
                       'SECONDS_REMAINING', 'SHOT_CLOCK', 'GAME_CLOCK', 'PLAYER_POSITION']
    shots = shots[feature_columns + ['SHOT_MADE_FLAG']]
    
    return shots.dropna()

def calculate_efficiency(shots):
    """Calculates the efficiency of shots."""
    shots['Area'], shots['Distance'] = zip(*shots.apply(categorize_shot, axis=1))
    summary = shots.groupby(['Area', 'Distance']).agg(
        Attempts=('SHOT_MADE_FLAG', 'size'),
        Made=('SHOT_MADE_FLAG', 'sum')
    ).reset_index()
    summary['Efficiency'] = summary['Made'] / summary['Attempts']
    return summary

def calculate_alignment_score(offensive_efficiency, defensive_efficiency):
    common_areas = set(offensive_efficiency['Area']) & set(defensive_efficiency['Area'])
    alignment_scores = []

    for area in common_areas:
        off_eff = offensive_efficiency[offensive_efficiency['Area'] == area]['Efficiency'].values[0]
        def_eff = defensive_efficiency[defensive_efficiency['Area'] == area]['Efficiency'].values[0]
        alignment_score = off_eff * (1 - def_eff)
        alignment_scores.append(alignment_score)

    return np.mean(alignment_scores)

def fetch_and_train(team_name, season, opponent_team=None, game_date=None):
    """Fetches shots data and trains the model."""
    offensive_shots, defensive_shots = fetch_all_shots_data(team_name, season, opponent_team, game_date)
    shots = pd.concat([offensive_shots, defensive_shots])
    processed_shots = preprocess_shots_data(shots)
    return processed_shots

def train_models(X_train, y_train):
    """Trains multiple models and returns them."""
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "LinearRegression": LinearRegression()
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    return models

def evaluate_models(models, X_test, y_test):
    """Evaluates models and prints their performance."""
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'\n{name} Model Performance:')
        print(f'Mean Absolute Error: {mae}')
        print(f'R-squared: {r2}')

# Example Usage
teams = ['Boston Celtics', 'Dallas Mavericks', 'Golden State Warriors', 'Los Angeles Lakers']
seasons = ['2023-24']

alignment_scores = []
team_performances = []  # Replace with actual performance metrics

for season in seasons:
    for home_team in teams:
        for opponent_team in teams:
            if home_team == opponent_team:
                continue
            
            offensive_shots = fetch_shots_data(home_team, True, season, opponent_team)
            defensive_shots = fetch_defensive_shots_data(opponent_team, season, home_team)

            offensive_efficiency = calculate_efficiency(offensive_shots)
            defensive_efficiency = calculate_efficiency(defensive_shots)

            alignment_score = calculate_alignment_score(offensive_efficiency, defensive_efficiency)
            alignment_scores.append(alignment_score)

            team_performances.append(np.random.rand())  # Random performance metric for illustration

df = pd.DataFrame({
    'alignment_score': alignment_scores,
    'team_performance': team_performances
})

# Correlation Matrix
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Train regression models
X = df[['alignment_score']]
y = df['team_performance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = train_models(X_train, y_train)
evaluate_models(models, X_test, y_test)
