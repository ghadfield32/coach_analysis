
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import os
import pandas as pd
import matplotlib.pyplot as plt
from shot_chart.nba_helpers import categorize_shot
from shot_chart.nba_shots import fetch_shots_data, fetch_defensive_shots_data

def calculate_efficiency(shots, debug=False):
    """Calculates the efficiency of shots and ensures unique areas."""
    shots['Area'], shots['Distance'] = zip(*shots.apply(lambda row: categorize_shot(row, debug=debug), axis=1))
    
    # Group by Area and Distance to aggregate data
    summary = shots.groupby(['Area', 'Distance']).agg(
        Attempts=('SHOT_MADE_FLAG', 'size'),
        Made=('SHOT_MADE_FLAG', 'sum')
    ).reset_index()
    
    # Calculate Efficiency
    summary['Efficiency'] = summary['Made'] / summary['Attempts']
    
    return summary


def calculate_team_fit(home_efficiency, opponent_efficiency):
    """Calculates the team fit using MAE and MAPE, ensuring consistent data across areas."""
    # Aggregate data to ensure unique areas
    home_efficiency = home_efficiency.groupby('Area').agg({
        'Attempts': 'sum', 
        'Made': 'sum'
    }).reset_index()
    home_efficiency['Efficiency'] = home_efficiency['Made'] / home_efficiency['Attempts']

    opponent_efficiency = opponent_efficiency.groupby('Area').agg({
        'Attempts': 'sum', 
        'Made': 'sum'
    }).reset_index()
    opponent_efficiency['Efficiency'] = opponent_efficiency['Made'] / opponent_efficiency['Attempts']
    
    # Get all unique areas present in either player's data
    all_areas = set(home_efficiency['Area']).union(set(opponent_efficiency['Area']))
    
    # Create a complete DataFrame for home and opponent efficiency with all areas
    home_efficiency_complete = home_efficiency.set_index('Area').reindex(all_areas, fill_value=0).reset_index()
    opponent_efficiency_complete = opponent_efficiency.set_index('Area').reindex(all_areas, fill_value=0).reset_index()
    
    # Log areas with no data
    missing_home_areas = all_areas - set(home_efficiency['Area'])
    missing_opponent_areas = all_areas - set(opponent_efficiency['Area'])
    
    if missing_home_areas:
        print(f"Warning: The home player is missing data for areas: {', '.join(missing_home_areas)}. Filling with zero attempts and efficiency.")
    if missing_opponent_areas:
        print(f"Warning: The opponent player is missing data for areas: {', '.join(missing_opponent_areas)}. Filling with zero attempts and efficiency.")
    
    # Calculate MAE and MAPE
    mae = mean_absolute_error(home_efficiency_complete['Efficiency'], opponent_efficiency_complete['Efficiency'])
    mape = mean_absolute_percentage_error(home_efficiency_complete['Efficiency'], opponent_efficiency_complete['Efficiency'])
    
    return mae, mape



def create_mae_table(home_team, season, all_teams):
    from shot_chart.nba_shots import fetch_shots_data, fetch_defensive_shots_data
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

def calculate_compatibility_between_players(player_shots):
    """Calculate MAE between each pair of players and determine shooting area compatibility."""
    mae_list = []
    player_names = list(player_shots.keys())
    
    # Debugging: Ensure there are at least two players to compare
    if len(player_names) < 2:
        print("Error: Less than two players provided for comparison.")
        return pd.DataFrame()  # Return an empty DataFrame early to prevent errors

    for i, player1 in enumerate(player_names):
        for player2 in player_names[i+1:]:
            mae = calculate_team_fit(player_shots[player1]['efficiency'], player_shots[player2]['efficiency'])[0]
            print(f"MAE for {player1} vs {player2}: {mae}")
            
            # Calculate compatibility based on shooting percentages
            compatibility = []
            for area in player_shots[player1]['efficiency']['Area'].unique():
                eff1 = player_shots[player1]['efficiency'].loc[player_shots[player1]['efficiency']['Area'] == area, 'Efficiency'].values[0]
                eff2 = player_shots[player2]['efficiency'].loc[player_shots[player2]['efficiency']['Area'] == area, 'Efficiency'].values[0]
                
                if eff1 >= 0.5 and eff2 >= 0.5:
                    compatibility.append('same_area')
                elif (eff1 >= 0.5 and eff2 < 0.5) or (eff1 < 0.5 and eff2 >= 0.5):
                    compatibility.append('diff_area')
            
            # Determine overall compatibility based on majority
            if compatibility.count('same_area') > compatibility.count('diff_area'):
                shooting_area_compatibility = 'efficient_in_same_areas'
            else:
                shooting_area_compatibility = 'efficient_in_diff_areas'
                
            mae_list.append({
                'Player 1': player1,
                'Player 2': player2,
                'MAE': mae,
                'Shooting Area Compatibility': shooting_area_compatibility
            })
    
    # Convert to DataFrame and return
    mae_df = pd.DataFrame(mae_list)
    return mae_df




