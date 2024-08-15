
import requests
import pandas as pd
import os
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    shotchartdetail, commonallplayers, commonteamroster, teamgamelog, playbyplayv2,
    scoreboardv2, boxscoresummaryv2, commonplayerinfo, teaminfocommon
)
import numpy as np
from datetime import datetime

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
    df.to_csv('data/players_list.csv', index=False)

def load_players_list(season):
    """Loads the list of players for a specific season from a CSV file."""
    file_path = 'data/players_list.csv'
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

def fetch_team_data(season):
    """Fetches team data for a given season."""
    team_data = []
    for team in teams.get_teams():
        team_data.append({
            'team_id': team['id'],
            'team_name': team['full_name'],
            'abbreviation': team['abbreviation']
        })
    return pd.DataFrame(team_data)

def fetch_coach_data(team_id, season):
    """Fetches coach data for a given team and season."""
    roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season).get_data_frames()[0]
    coaches = roster[roster['POSITION'] == 'Head Coach']
    return coaches[['COACH_ID', 'COACH_NAME']].rename(columns={'COACH_ID': 'coach_id', 'COACH_NAME': 'name'})

def fetch_detailed_game_data(game_id):
    """Fetches detailed game data for a given game ID."""
    box_score = boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id).get_data_frames()
    game_summary = box_score[0]
    return {
        'game_id': game_id,
        'date': game_summary['GAME_DATE_EST'].iloc[0],
        'home_team_id': game_summary['HOME_TEAM_ID'].iloc[0],
        'away_team_id': game_summary['VISITOR_TEAM_ID'].iloc[0],
        'home_team_score': game_summary['HOME_TEAM_SCORE'].iloc[0],
        'away_team_score': game_summary['VISITOR_TEAM_SCORE'].iloc[0],
        'season': game_summary['SEASON'].iloc[0]
    }

def fetch_detailed_player_data(player_id):
    """Fetches detailed player information."""
    player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
    return {
        'player_id': player_id,
        'full_name': player_info['DISPLAY_FIRST_LAST'].iloc[0],
        'position': player_info['POSITION'].iloc[0],
        'height': player_info['HEIGHT'].iloc[0],
        'weight': player_info['WEIGHT'].iloc[0],
        'birth_date': player_info['BIRTHDATE'].iloc[0],
        'country': player_info['COUNTRY'].iloc[0]
    }

def fetch_detailed_team_data(team_id):
    """Fetches detailed team information."""
    team_info = teaminfocommon.TeamInfoCommon(team_id=team_id).get_data_frames()[0]
    return {
        'team_id': team_id,
        'team_name': team_info['TEAM_NAME'].iloc[0],
        'abbreviation': team_info['TEAM_ABBREVIATION'].iloc[0],
        'city': team_info['TEAM_CITY'].iloc[0],
        'state': team_info['TEAM_STATE'].iloc[0],
        'year_founded': team_info['YEAR_FOUNDED'].iloc[0]
    }

def fetch_game_data(team_id, season):
    """Fetches game data for a given team and season."""
    game_log = teamgamelog.TeamGameLog(team_id=team_id, season=season).get_data_frames()[0]
    game_data = []
    for _, game in game_log.iterrows():
        game_data.append({
            'game_id': game['Game_ID'],
            'date': game['GAME_DATE'],
            'home_team': game['MATCHUP'].split()[-1],
            'away_team': game['MATCHUP'].split()[0],
            'team_id': team_id
        })
    return pd.DataFrame(game_data)

def fetch_shot_data(player_id, team_id, season):
    """Fetches shot data for a given player, team, and season."""
    shot_data = shotchartdetail.ShotChartDetail(
        team_id=team_id,
        player_id=player_id,
        season_nullable=season,
        context_measure_simple='FGA'
    ).get_data_frames()[0]
    
    shot_data['shot_id'] = shot_data.index  # Add a unique identifier for each shot
    return shot_data

def fetch_play_data(game_id):
    """Fetches play-by-play data for a given game."""
    play_by_play = playbyplayv2.PlayByPlayV2(game_id=game_id).get_data_frames()[0]
    play_data = []
    for _, play in play_by_play.iterrows():
        play_data.append({
            'play_id': play['EVENTNUM'],
            'game_id': game_id,
            'play_type': play['EVENTMSGTYPE']
        })
    return pd.DataFrame(play_data)

def prepare_data_for_neo4j(season):
    """Prepares all data for Neo4j import."""
    players_df = load_players_list(season)
    teams_df = fetch_team_data(season)
    
    all_game_data = []
    all_shot_data = []
    all_play_data = []
    all_coach_data = []
    all_player_details = []
    all_team_details = []
    
    for _, team in teams_df.iterrows():
        team_id = team['team_id']
        
        # Fetch detailed team data
        team_details = fetch_detailed_team_data(team_id)
        all_team_details.append(team_details)
        
        # Fetch coach data
        coach_data = fetch_coach_data(team_id, season)
        all_coach_data.append(coach_data)
        
        game_data = fetch_game_data(team_id, season)
        
        for _, game in game_data.iterrows():
            detailed_game = fetch_detailed_game_data(game['game_id'])
            all_game_data.append(detailed_game)
            
            play_data = fetch_play_data(game['game_id'])
            all_play_data.append(play_data)
        
        team_players = players_df[players_df['team_id'] == team_id]
        for _, player in team_players.iterrows():
            player_id = player['id']
            
            # Fetch detailed player data
            player_details = fetch_detailed_player_data(player_id)
            all_player_details.append(player_details)
            
            shot_data = fetch_shot_data(player_id, team_id, season)
            all_shot_data.append(shot_data)
    
    # Convert lists to DataFrames
    games_df = pd.DataFrame(all_game_data)
    shots_df = pd.concat(all_shot_data)
    plays_df = pd.concat(all_play_data)
    coaches_df = pd.concat(all_coach_data)
    players_df = pd.DataFrame(all_player_details)
    teams_df = pd.DataFrame(all_team_details)
    
    # Prepare CSV files for Neo4j import
    players_df.to_csv('data/neo4j_players.csv', index=False)
    teams_df.to_csv('data/neo4j_teams.csv', index=False)
    coaches_df.to_csv('data/neo4j_coaches.csv', index=False)
    games_df.to_csv('data/neo4j_games.csv', index=False)
    shots_df.to_csv('data/neo4j_shots.csv', index=False)
    plays_df.to_csv('data/neo4j_plays.csv', index=False)
    
    # Prepare relationship CSV files
    prepare_relationship_files(players_df, teams_df, shots_df, games_df, plays_df, coaches_df)

def prepare_relationship_files(players_df, teams_df, shots_df, games_df, plays_df, coaches_df):
    """Prepares CSV files for Neo4j relationships."""
    # PLAYS_FOR relationship
    plays_for = players_df[['player_id', 'team_id']]
    plays_for.to_csv('data/neo4j_plays_for.csv', index=False)
    
    # TOOK_SHOT relationship
    took_shot = shots_df[['PLAYER_ID', 'shot_id']].rename(columns={'PLAYER_ID': 'player_id'})
    took_shot.to_csv('data/neo4j_took_shot.csv', index=False)
    
    # IN_GAME relationship
    in_game = shots_df[['shot_id', 'GAME_ID']].rename(columns={'GAME_ID': 'game_id'})
    in_game.to_csv('data/neo4j_in_game.csv', index=False)
    
    # PARTICIPATED_IN relationship
    participated_in = games_df[['team_id', 'game_id']]
    participated_in.to_csv('data/neo4j_participated_in.csv', index=False)
    
    # COACHES relationship
    coaches_df[['coach_id', 'team_id']].to_csv('data/neo4j_coaches.csv', index=False)
    
    # USED_IN relationship
    used_in = plays_df[['play_id', 'game_id']]
    used_in.to_csv('data/neo4j_used_in.csv', index=False)
    
    # PART_OF relationship
    # This requires matching shots to plays, which might need more detailed data
    # For now, we'll create a placeholder relationship
    part_of = shots_df.merge(plays_df, left_on='GAME_ID', right_on='game_id')
    part_of[['shot_id', 'play_id']].to_csv('data/neo4j_part_of.csv', index=False)

def categorize_shot(row):
    """Categorizes a shot based on its location."""
    x, y = row['LOC_X'], row['LOC_Y']
    if y > 237.5:
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
    elif np.sqrt(x**2 + y**2) > 280:
        return 'Long 3', 'Beyond 3-pt'
    else:
        return 'Backcourt', 'Beyond Half Court'
