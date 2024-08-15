
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

def categorize_shot(row):
    """Categorizes a shot based on its location."""
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
        elif x < 80:
            return 'Left Wing 3', '24+ ft'
        else:
            return 'Right Corner 3', '24+ ft'
    elif y > 142.5:
        if x < -80:
            return 'Left Wing 3', '24+ ft'
        elif x < 0:
            return 'Left Top of Key 3', '20-24 ft'
        else:
            return 'Right Top of Key 3', '20-24 ft'
    elif y > 47.5:
        if x < -80:
            return 'Left Baseline Mid-range', '10-20 ft'
        elif x < -10:
            return 'Left Elbow Mid-range', '10-20 ft'
        elif x < 10:
            return 'Center Mid-range', '10-20 ft'
        elif x < 80:
            return 'Right Elbow Mid-range', '10-20 ft'
        else:
            return 'Right Baseline Mid-range', '10-20 ft'
    elif y > 0:
        if x < -80:
            return 'Left of Near Basket', '0-10 ft'
        elif x < 80:
            return 'Center of Near Basket', '0-10 ft'
        else:
            return 'Right of Near Basket', '0-10 ft'
    else:
        return 'Unknown', 'Unknown'
