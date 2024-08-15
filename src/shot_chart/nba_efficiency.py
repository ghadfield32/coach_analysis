
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import os
from shot_chart.nba_helpers import categorize_shot
from shot_chart.nba_shots import fetch_shots_data, fetch_defensive_shots_data

def calculate_efficiency(shots):
    """Calculates the efficiency of shots."""
    shots['Area'], shots['Distance'] = zip(*shots.apply(categorize_shot, axis=1))
    summary = shots.groupby(['Area', 'Distance']).agg(
        Attempts=('SHOT_MADE_FLAG', 'size'),
        Made=('SHOT_MADE_FLAG', 'sum')
    ).reset_index()
    summary['Efficiency'] = summary['Made'] / summary['Attempts']
    return summary

def calculate_team_fit(home_efficiency, opponent_efficiency):
    """Calculates the team fit using MAE and MAPE."""
    common_areas = set(home_efficiency['Area']).intersection(set(opponent_efficiency['Area']))
    
    home_efficiency_common = home_efficiency[home_efficiency['Area'].isin(common_areas)]
    opponent_efficiency_common = opponent_efficiency[opponent_efficiency['Area'].isin(common_areas)]
    
    mae = mean_absolute_error(home_efficiency_common['Efficiency'], opponent_efficiency_common['Efficiency'])
    mape = mean_absolute_percentage_error(home_efficiency_common['Efficiency'], opponent_efficiency_common['Efficiency'])
    
    return mae, mape

def create_mae_table(home_team, season, all_teams):
    """Creates a table of MAE values for the given home team against all opponents."""
    mae_list = []
    home_shots = fetch_shots_data(home_team, True, season)
    home_efficiency = calculate_efficiency(home_shots)

    for opponent in all_teams:
        if opponent == home_team:
            continue
        
        # Ensure the season is passed to fetch_defensive_shots_data
        opponent_shots = fetch_defensive_shots_data(opponent, True, season)
        opponent_efficiency = calculate_efficiency(opponent_shots)
        
        mae, mape = calculate_team_fit(home_efficiency, opponent_efficiency)
        
        mae_list.append({
            'Home Team': home_team,
            'Opponent Team': opponent,
            'MAE': mae,
            'MAPE': mape,
            'Season': season
        })
    
    mae_df = pd.DataFrame(mae_list)
    mae_df = mae_df.sort_values(by='MAE')
    return mae_df

def save_mae_table(mae_df, file_path):
    """Saves the MAE table to a CSV file."""
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        mae_df = pd.concat([existing_df, mae_df]).drop_duplicates()
    mae_df.to_csv(file_path, index=False)

def load_mae_table(file_path):
    """Loads the MAE table from a CSV file if it exists."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def get_seasons_range(mae_df):
    """Returns the minimum and maximum seasons in the MAE DataFrame."""
    min_season = mae_df['Season'].min()
    max_season = mae_df['Season'].max()
    return min_season, max_season


