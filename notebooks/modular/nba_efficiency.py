
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from modular.nba_helpers import categorize_shot
from modular.nba_shots import fetch_shots_data, fetch_defensive_shots_data

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
    # Find common areas
    common_areas = set(home_efficiency['Area']).intersection(set(opponent_efficiency['Area']))
    
    home_efficiency_common = home_efficiency[home_efficiency['Area'].isin(common_areas)]
    opponent_efficiency_common = opponent_efficiency[opponent_efficiency['Area'].isin(common_areas)]
    
    # Calculate MAE and MAPE
    mae = mean_absolute_error(home_efficiency_common['Efficiency'], opponent_efficiency_common['Efficiency'])
    mape = mean_absolute_percentage_error(home_efficiency_common['Efficiency'], opponent_efficiency_common['Efficiency'])
    
    return mae, mape

def create_mae_table(home_team, season, all_teams):
    """Creates a table of MAE values for the given home team against all opponents."""
    mae_list = []
    for opponent in all_teams:
        if opponent == home_team:
            continue
        home_shots = fetch_shots_data(home_team, True, season)
        opponent_shots = fetch_defensive_shots_data(opponent, season)
        
        home_efficiency = calculate_efficiency(home_shots)
        opponent_efficiency = calculate_efficiency(opponent_shots)
        
        mae, mape = calculate_team_fit(home_efficiency, opponent_efficiency)
        
        mae_list.append({
            'Home Team': home_team,
            'Opponent Team': opponent,
            'MAE': mae,
            'MAPE': mape
        })
    
    mae_df = pd.DataFrame(mae_list)
    mae_df = mae_df.sort_values(by='MAE')
    return mae_df

def save_mae_table(mae_df):
    """Saves the MAE table to a CSV file."""
    file_path = 'data/mae_table.csv'
    mae_df.to_csv(file_path, index=False)

def load_mae_table():
    """Loads the MAE table from a CSV file if it exists."""
    file_path = 'data/mae_table.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None
