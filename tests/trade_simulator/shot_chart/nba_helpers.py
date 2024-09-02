
import requests
import pandas as pd
import os
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.endpoints import commonallplayers
import numpy as np

def fetch_and_save_players_list():
    """Fetches the list of all players for specified seasons and saves it to a CSV file."""
    seasons = ["2023-24", "2022-23", "2021-22"]
    all_players = commonallplayers.CommonAllPlayers(is_only_current_season=0).get_data_frames()[0]

    players_data = []
    for season in seasons:
        for _, player in all_players.iterrows():
            players_data.append({
                'id': player['PERSON_ID'],
                'full_name': player['DISPLAY_FIRST_LAST'],
                'season': season
            })

    df = pd.DataFrame(players_data).drop_duplicates()
    df.to_csv('data/shot_chart_data/players_list.csv', index=False)

def load_players_list(season):
    """Loads the list of players for a specific season from a CSV file."""
    file_path = 'data/shot_chart_data/players_list.csv'
    if not os.path.exists(file_path):
        fetch_and_save_players_list()
    
    players_df = pd.read_csv(file_path)
    return players_df[players_df['season'] == season]

def get_team_abbreviation(team_name):
    """Gets the team abbreviation for a given team name."""
    team_dictionary = teams.get_teams()
    team_info = [team for team in team_dictionary if team['full_name'] == team_name]
    if not team_info:
        raise ValueError(f"No team found with name {team_name}")
    return team_info[0]['abbreviation']

def categorize_shot(row, debug=False):
    """Categorizes a shot based on its location with optional debugging.
    
    Args:
        row (pd.Series): A row of shot data containing 'LOC_X' and 'LOC_Y'.
        debug (bool): If True, logs detailed information about shots that don't fit into known categories.
    
    Returns:
        tuple: A tuple containing the area and distance category of the shot.
               Returns ('Unknown', 'Unknown') for shots that don't fit into known categories when debug=False.
    """
    x, y = row['LOC_X'], row['LOC_Y']
    distance_from_hoop = np.sqrt(x**2 + y**2)

    if distance_from_hoop > 300:  # Over 30 ft
        return 'Backcourt', 'Beyond 30 ft'
    elif distance_from_hoop > 240:  # 24-30 ft
        if x < -80:
            return 'Deep 3 Left', '24-30 ft'
        elif x > 80:
            return 'Deep 3 Right', '24-30 ft'
        else:
            return 'Deep 3 Center', '24-30 ft'
    elif y > 237.5:
        if x < -80:
            return 'Left Corner 3', '24+ ft'
        elif x > 80:
            return 'Right Corner 3', '24+ ft'
        else:
            return 'Left Wing 3' if x < 0 else 'Right Wing 3', '24+ ft'
    elif y > 142.5:
        if x < -80:
            return 'Left Wing 3', '24+ ft'
        elif x > 80:
            return 'Right Wing 3', '24+ ft'
        elif x < 0:
            return 'Left Top of Key 3', '20-24 ft'
        else:
            return 'Right Top of Key 3', '20-24 ft'
    elif y > 47.5:
        if x < -80:
            return 'Left Baseline Mid-range', '10-20 ft'
        elif x > 80:
            return 'Right Baseline Mid-range', '10-20 ft'
        elif x < -10:
            return 'Left Elbow Mid-range', '10-20 ft'
        elif x > 10:
            return 'Right Elbow Mid-range', '10-20 ft'
        else:
            return 'Center Mid-range', '10-20 ft'
    elif y >= 0:  # Near basket, including under the hoop
        if distance_from_hoop < 10:
            if x < -10:
                return 'Left of Near Basket', '0-10 ft'
            elif x > 10:
                return 'Right of Near Basket', '0-10 ft'
            else:
                return 'Center of Near Basket', '0-10 ft'
        elif distance_from_hoop < 20:  # Adjusted to correctly categorize shots at 10-20 ft range
            if x < -20:
                return 'Left of Near Basket', '10-20 ft'
            elif x > 20:
                return 'Right of Near Basket', '10-20 ft'
            else:
                return 'Center of Near Basket', '10-20 ft'
        elif distance_from_hoop < 30:  # Added condition for shots in the 20-30 ft range
            return 'Near Mid-range', '20-30 ft'
        else:
            if x < -80:
                return 'Left Wing Mid-range', '20-30 ft'
            elif x > 80:
                return 'Right Wing Mid-range', '20-30 ft'
            else:
                return 'Center Mid-range', '20-30 ft'
    
    if debug:
        print(f"Debug: Unknown shot location (x, y)=({x}, {y}), distance from hoop={distance_from_hoop}")
    
    return 'Unknown', 'Unknown'  # Ensure that a tuple is always returned



def get_all_court_areas():
    """Returns a list of all possible court areas defined in categorize_shot."""
    return [
        'Backcourt', 'Deep 3 Left', 'Deep 3 Center', 'Deep 3 Right',
        'Left Corner 3', 'Right Corner 3', 'Left Wing 3', 'Right Wing 3',
        'Left Top of Key 3', 'Right Top of Key 3', 'Center Mid-range',
        'Left Baseline Mid-range', 'Right Baseline Mid-range',
        'Left Elbow Mid-range', 'Right Elbow Mid-range',
        'Center of Near Basket', 'Left of Near Basket', 'Right of Near Basket',
        'Near Mid-range', 'Left Wing Mid-range', 'Right Wing Mid-range'
    ]

