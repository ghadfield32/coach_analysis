import pandas as pd
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import players, teams

def fetch_shots_data(team_name, is_team, season, opponent_team=None, game_date=None):
    """Fetches shots data for a team or player for a given season with optional filters."""
    if is_team:
        team_id = get_team_id(team_name)
        shotchart = shotchartdetail.ShotChartDetail(
            team_id=team_id,
            player_id=0,
            context_measure_simple='FGA',
            season_nullable=season,
            season_type_all_star=['Regular Season', 'Playoffs']
        )
    else:
        player_id = get_player_id(team_name)
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

def get_team_id(team_name):
    team_dict = teams.get_teams()
    team = next((team for team in team_dict if team['full_name'] == team_name), None)
    if team is None:
        raise ValueError(f"No team found with name {team_name}")
    return team['id']

def get_player_id(player_name):
    player_dict = players.get_players()
    player = next((player for player in player_dict if player['full_name'] == player_name), None)
    if player is None:
        raise ValueError(f"No player found with name {player_name}")
    return player['id']

def get_team_abbreviation(team_name):
    team_dict = teams.get_teams()
    team = next((team for team in team_dict if team['full_name'] == team_name), None)
    if team is None:
        raise ValueError(f"No team found with name {team_name}")
    return team['abbreviation']
