
import pandas as pd
from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import players, teams
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

from shot_chart.nba_helpers import categorize_shot
from shot_chart.nba_plotting import plot_shot_chart_hexbin
from shot_chart.nba_helpers import get_team_abbreviation


def fetch_shots_for_multiple_players(player_names, season, court_areas=None, opponent_name=None, debug=False):
    from shot_chart.nba_efficiency import calculate_efficiency
    """Fetch shots data for multiple players with an option to filter by court areas and an opponent team."""
    player_shots = {}
    for player_name in player_names:
        print(f"Fetching shots for {player_name}")
        shots = fetch_shots_data(player_name, is_team=False, season=season, opponent_team=opponent_name)
        
        # Apply the categorize_shot function to generate 'Area' and 'Distance' columns
        shots['Area'], shots['Distance'] = zip(*shots.apply(lambda row: categorize_shot(row, debug=debug), axis=1))
        
        # If court_areas is provided and not set to 'all', filter the shots data
        if court_areas and court_areas != 'all':
            shots = shots[shots['Area'].isin(court_areas)]
        
        efficiency = calculate_efficiency(shots)
        player_shots[player_name] = {
            'shots': shots,
            'efficiency': efficiency
        }
        
        # Plot shot chart for each player
        fig = plot_shot_chart_hexbin(shots, f'{player_name} Shot Chart', opponent=opponent_name if opponent_name else "the rest of the league")
        plt.show()
        
        # Print efficiency summary for each player
        print(f"Efficiency for {player_name}:")
        print(efficiency)
        
    return player_shots

def fetch_shots_data(name, is_team, season, opponent_team=None, opponent_player=None, game_date=None, debug=False):
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

    # Handle the case where opponent_team is "all"
    if opponent_team and opponent_team.lower() != "all":
        opponent_abbreviation = get_team_abbreviation(opponent_team)
        data = data[(data['HTM'] == opponent_abbreviation) | (data['VTM'] == opponent_abbreviation)]
    
    if opponent_player:
        opponent_dictionary = players.get_players()
        opponent_info = [player for player in opponent_dictionary if player['full_name'] == opponent_player]
        if opponent_info:
            opponent_player_id = opponent_info[0]['id']
            data = data[data['PLAYER_ID'] == opponent_player_id]

    if game_date:
        data = data[data['GAME_DATE'] == game_date.replace('-', '')]

    # Apply the categorize_shot function with the debug parameter
    data['Area'], data['Distance'] = zip(*data.apply(lambda row: categorize_shot(row, debug=debug), axis=1))

    return data


def fetch_defensive_shots_data(name, is_team, season, opponent_team=None, opponent_player=None, game_date=None, debug=False):
    """Fetches defensive shots data for a team or player for a given season with optional filters."""
    if is_team:
        team_abbr = get_team_abbreviation(name)
        shotchart = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=0,
            context_measure_simple='FGA',
            season_nullable=season,
            season_type_all_star=['Regular Season', 'Playoffs']
        )
        
        data = shotchart.get_data_frames()[0]
        defensive_shots = data[(data['HTM'] == team_abbr) | (data['VTM'] == team_abbr)]
        defensive_shots = defensive_shots[defensive_shots['TEAM_NAME'] != name]

    else:
        player_dictionary = players.get_players()
        player_info = [player for player in player_dictionary if player['full_name'] == name]
        
        if not player_info:
            raise ValueError(f"No player found with name {name}")
        
        player_id = player_info[0]['id']
        print(f"Fetching data for Player ID: {player_id}")
        
        shotchart = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=0,
            context_measure_simple='FGA',
            season_nullable=season,
            season_type_all_star=['Regular Season', 'Playoffs']
        )
        
        data = shotchart.get_data_frames()[0]
        defensive_shots = data[data['PLAYER_ID'] == player_id]

    if opponent_team:
        opponent_abbreviation = get_team_abbreviation(opponent_team)
        defensive_shots = defensive_shots[(defensive_shots['HTM'] == opponent_abbreviation) | (defensive_shots['VTM'] == opponent_abbreviation)]

    if opponent_player:
        opponent_dictionary = players.get_players()
        opponent_info = [player for player in opponent_dictionary if player['full_name'] == opponent_player]
        if opponent_info:
            opponent_player_id = opponent_info[0]['id']
            defensive_shots = defensive_shots[defensive_shots['PLAYER_ID'] == opponent_player_id]

    if game_date:
        defensive_shots = defensive_shots[defensive_shots['GAME_DATE'] == game_date.replace('-', '')]

    # Apply the categorize_shot function with the debug parameter
    defensive_shots['Area'], defensive_shots['Distance'] = zip(*defensive_shots.apply(lambda row: categorize_shot(row, debug=debug), axis=1))

    return defensive_shots
