
import pandas as pd
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import players, teams
from modular.nba_helpers import get_team_abbreviation

def fetch_shots_data(name, is_team, season, opponent_team=None, game_date=None):
    """Fetches shots data for a team or player for a given season with optional filters."""
    if is_team:
        team_dictionary = teams.get_teams()
        team_info = [team for team in team_dictionary if team['full_name'] == name]
        
        if not team_info:
            raise ValueError(f"No team found with name {name}")
        
        team_id = team_info[0]['id']
        print(f"Fetching data for Team ID: {team_id}")
        
        shotchart = shotchartdetail.ShotChartDetail(
            team_id=team_id,
            player_id=0,
            context_measure_simple='FGA',
            season_nullable=season,
            season_type_all_star=['Regular Season', 'Playoffs']
        )
    else:
        player_dictionary = players.get_players()
        player_info = [player for player in player_dictionary if player['full_name'] == name]
        
        if not player_info:
            raise ValueError(f"No player found with name {name}")
        
        player_id = player_info[0]['id']
        print(f"Fetching data for Player ID: {player_id}")
        
        shotchart = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=player_id,
            context_measure_simple='FGA',
            season_nullable=season,
            season_type_all_star=['Regular Season', 'Playoffs']
        )
    
    data = shotchart.get_data_frames()[0]

    if opponent_team:
        opponent_abbreviation = get_team_abbreviation(opponent_team)
        data = data[(data['HTM'] == opponent_abbreviation) | (data['VTM'] == opponent_abbreviation)]
    
    if game_date:
        data = data[data['GAME_DATE'] == game_date.replace('-', '')]

    return data

def fetch_defensive_shots_data(team_name, season, opponent_team=None, game_date=None):
    """Fetches defensive shots data for a given team and season with optional filters."""
    team_abbr = get_team_abbreviation(team_name)
    shotchart = shotchartdetail.ShotChartDetail(
        team_id=0,
        player_id=0,
        context_measure_simple='FGA',
        season_nullable=season,
        season_type_all_star=['Regular Season', 'Playoffs']
    )
    
    data = shotchart.get_data_frames()[0]
    defensive_shots = data[(data['HTM'] == team_abbr) | (data['VTM'] == team_abbr)]
    defensive_shots = defensive_shots[defensive_shots['TEAM_NAME'] != team_name]

    if opponent_team:
        opponent_abbreviation = get_team_abbreviation(opponent_team)
        defensive_shots = defensive_shots[(defensive_shots['HTM'] == opponent_abbreviation) | (defensive_shots['VTM'] == opponent_abbreviation)]

    if game_date:
        defensive_shots = defensive_shots[defensive_shots['GAME_DATE'] == game_date.replace('-', '')]

    return defensive_shots
