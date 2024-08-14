
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelogs, leaguegamefinder
from tabulate import tabulate
import time
from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.static import teams

# Constants
RELEVANT_STATS = ['PTS', 'AST', 'TOV', 'STL', 'BLK', 'OREB', 'DREB', 'FGM', 'FG3M', 'FGA']

def load_team_data():
    nba_teams = teams.get_teams()
    team_df = pd.DataFrame(nba_teams)
    return team_df[['id', 'full_name', 'abbreviation']]


def fetch_player_info(player_id, debug=False):
    """Fetch player information based on player ID."""
    try:
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
        if debug:
            print(f"Fetched info for player ID {player_id}: {player_info['DISPLAY_FIRST_LAST'].values[0]}")
        return player_info
    except Exception as e:
        if debug:
            print(f"Error fetching info for player ID {player_id}: {e}")
        return None

def fetch_season_data_by_year(year, debug=False):
    """Fetch player game logs data for a given starting year of the NBA season."""
    season = f"{year}-{str(year+1)[-2:]}"
    if debug:
        print(f"Fetching player data for season {season}")
    try:
        player_logs = playergamelogs.PlayerGameLogs(season_nullable=season).get_data_frames()[0]
        player_logs['SEASON'] = season
        player_logs['GAME_DATE'] = pd.to_datetime(player_logs['GAME_DATE'])
        if debug:
            print(f"Player data for season {season} contains {player_logs.shape[0]} rows.")
        return player_logs
    except Exception as e:
        if debug:
            print(f"Error fetching player data for season {season}: {e}")
        return None

# Helper Functions
def get_champion(season, debug=False):
    """Fetch the champion team for a given NBA season."""
    try:
        games = leaguegamefinder.LeagueGameFinder(season_nullable=season, season_type_nullable='Playoffs').get_data_frames()[0]
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
        last_game = games.sort_values('GAME_DATE').iloc[-2:]
        winner = last_game[last_game['WL'] == 'W'].iloc[0]
        if debug:
            print(f"Champion for season {season}: {winner['TEAM_NAME']} ({winner['TEAM_ID']})")
        return winner['TEAM_NAME']
    except Exception as e:
        if debug:
            print(f"Error fetching champion for season {season}: {e}")
        return None

def get_champions(start_year, end_year, debug=False):
    """Fetch champions for each season from start_year to end_year."""
    champions = {}
    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year+1)[-2:]}"
        champ_name = get_champion(season, debug)
        if champ_name:
            champions[season] = {'ChampionTeamName': champ_name}
        elif debug:
            print(f"Champion data not available for season {season}")
        time.sleep(1)  # To avoid overwhelming the API
    if debug:
        print(f"Champions data: {champions}")
    return champions

def calculate_percentiles(stats_df, debug=False):
    """Calculate percentiles for stats after averages are computed."""
    # Group by season and calculate percentiles for each season separately
    for season in stats_df['SEASON'].unique():
        season_data = stats_df[stats_df['SEASON'] == season]
        for stat in RELEVANT_STATS + ['eFG%']:
            stat_per_game = f'{stat}_per_game'
            if stat_per_game in season_data.columns:
                stats_df.loc[season_data.index, f'{stat}_percentile'] = season_data[stat_per_game].rank(pct=True)
                if debug:
                    print(f"Calculated percentiles for {stat} in season {season}:")
                    print(stats_df.loc[season_data.index, [stat_per_game, f'{stat}_percentile']].head())
    return stats_df

def calculate_team_stats(player_data, period, debug=False):
    """Calculate team-level statistics, including averages."""
    if debug:
        print(f"Calculating {period} team-level statistics.")
        print("Initial player_data head:")
        print(player_data.head())

    # Calculate team-level stats by summing player stats for each team and season
    team_stats = (
        player_data.groupby(['SEASON', 'TEAM_NAME'])[RELEVANT_STATS]
        .sum()
        .reset_index()
    )

    # Calculate the number of games played by each team
    games_played = player_data.groupby(['SEASON', 'TEAM_NAME'])['GAME_ID'].nunique().reset_index(name='GAMES_PLAYED')

    # Merge games played with team stats
    team_stats = pd.merge(team_stats, games_played, on=['SEASON', 'TEAM_NAME'])

    # Calculate stats per game
    for stat in RELEVANT_STATS:
        team_stats[f'{stat}_per_game'] = team_stats[stat] / team_stats['GAMES_PLAYED']

    # Add period column
    team_stats['PERIOD'] = period

    if debug:
        print(f"{period} team-level statistics head:")
        print(team_stats.head())

    return team_stats

def process_champion_team_data(player_data, champions, debug=False):
    """Process the game logs to get data for the champion teams."""
    champion_team_stats = pd.DataFrame()

    for season, champ_info in champions.items():
        champ_name = champ_info['ChampionTeamName']

        # Filter player data for champion team
        champ_data = player_data[(player_data['SEASON'] == season) & (player_data['TEAM_NAME'] == champ_name)]

        if champ_data.empty:
            if debug:
                print(f"No data found for champion team {champ_name} in season {season}")
            continue

        # Calculate team statistics
        champ_stats = calculate_team_stats(champ_data, 'Champion', debug)
        champ_stats['ChampionTeamName'] = champ_name

        champion_team_stats = pd.concat([champion_team_stats, champ_stats], ignore_index=True)

    # Calculate eFG%
    champion_team_stats['eFG%_per_game'] = (
        (champion_team_stats['FGM_per_game'] + 0.5 * champion_team_stats['FG3M_per_game']) / champion_team_stats['FGA_per_game']
    )

    # Calculate percentiles for champion teams within their season
    champion_team_stats = calculate_percentiles(champion_team_stats, debug)

    return champion_team_stats

def calculate_post_trade_team_stats(player_data, traded_players, trade_date, season_data, debug=False):
    """Calculate post-trade team-level statistics, using entire season if necessary."""
    if debug:
        print("Calculating post-trade team-level statistics.")

    # Convert trade_date to datetime
    trade_date = pd.to_datetime(trade_date)

    # Determine the start of the season based on the SEASON column
    season_start_year = int(player_data['SEASON'].iloc[0].split('-')[0])
    season_start_date = pd.to_datetime(f"{season_start_year}-10-01")  # NBA season typically starts in October

    # Determine whether to use entire season data or data after trade date
    if trade_date < season_start_date:
        if debug:
            print(f"Warning: Trade date {trade_date} is earlier than the start of the season {season_start_date}. Using entire season data.")
        post_trade_data = season_data  # Use the entire season data
    else:
        post_trade_data = player_data[player_data['GAME_DATE'] >= trade_date].copy()

    if debug:
        print("Post-trade player data head:")
        print(post_trade_data.head())

    # Calculate post-trade stats
    post_trade_stats = calculate_team_stats(post_trade_data, 'Post-trade', debug)

    # Calculate traded players' post-trade averages
    traded_player_stats = {}
    for player_id, (new_team_name, player_name) in traded_players.items():
        player_post_trade_stats = post_trade_data[post_trade_data['PLAYER_ID'] == player_id][RELEVANT_STATS].mean()
        traded_player_stats[player_id] = player_post_trade_stats.to_dict()
        if debug:
            print(f"{player_name} averages post-trade (to {new_team_name}): {traded_player_stats[player_id]}")

    # Adjust post-trade stats based on traded players
    for player_id, (new_team_name, player_name) in traded_players.items():
        old_team_name = player_data[player_data['PLAYER_ID'] == player_id]['TEAM_NAME'].iloc[0]
        post_trade_games = post_trade_stats.loc[post_trade_stats['TEAM_NAME'] == new_team_name, 'GAMES_PLAYED'].values[0]
        
        if debug:
            print(f"\nAdjusting stats for trade: {player_name} from {old_team_name} to {new_team_name}")

        # Remove player's stats from old team
        for stat in RELEVANT_STATS:
            if debug:
                print(f"  Before adjustment - {old_team_name} {stat}: {post_trade_stats.loc[post_trade_stats['TEAM_NAME'] == old_team_name, stat].values[0]}")
            post_trade_stats.loc[post_trade_stats['TEAM_NAME'] == old_team_name, stat] -= traded_player_stats[player_id][stat] * post_trade_games
            if debug:
                print(f"  After adjustment - {old_team_name} {stat}: {post_trade_stats.loc[post_trade_stats['TEAM_NAME'] == old_team_name, stat].values[0]}")

        # Add player's stats to new team
        for stat in RELEVANT_STATS:
            if debug:
                print(f"  Before adjustment - {new_team_name} {stat}: {post_trade_stats.loc[post_trade_stats['TEAM_NAME'] == new_team_name, stat].values[0]}")
            post_trade_stats.loc[post_trade_stats['TEAM_NAME'] == new_team_name, stat] += traded_player_stats[player_id][stat] * post_trade_games
            if debug:
                print(f"  After adjustment - {new_team_name} {stat}: {post_trade_stats.loc[post_trade_stats['TEAM_NAME'] == new_team_name, stat].values[0]}")

    # Recalculate per-game stats
    for stat in RELEVANT_STATS:
        post_trade_stats[f'{stat}_per_game'] = post_trade_stats[stat] / post_trade_stats['GAMES_PLAYED']

    if debug:
        print("Post-trade team stats calculated successfully.")
        print("Post-trade team stats head:")
        print(post_trade_stats.head())

    return post_trade_stats

def calculate_average_champion_stats(champion_team_data, debug=False):
    """Calculate the average statistics for all champion teams."""
    if debug:
        print("Calculating average champion team statistics.")
    
    # Calculate average stats for all champion teams
    avg_stats = champion_team_data[RELEVANT_STATS + [f'{stat}_per_game' for stat in RELEVANT_STATS] + ['eFG%_per_game']].mean()

    # Create a DataFrame for the average stats
    avg_row = pd.DataFrame([avg_stats], columns=champion_team_data.columns)
    avg_row['SEASON'] = 'Average'
    avg_row['TEAM_NAME'] = 'Average Champion'
    avg_row['PERIOD'] = 'Champion'
    avg_row['ChampionTeamName'] = 'Average Champion'

    # Append the average row to the champion team data
    champion_team_data = pd.concat([champion_team_data, avg_row], ignore_index=True)

    # Recalculate percentiles for champion teams within their data
    champion_team_data = calculate_percentiles(champion_team_data, debug)
    
    if debug:
        print("\nChampion Team Stats with Average:")
        print(tabulate(champion_team_data, headers='keys', tablefmt='grid'))

    # Return the updated champion data with the new average
    return champion_team_data

def compare_team_performance(percentiles, average_champion_stats, traded_teams, debug=True):
    """Generate a comparison table for team performance before and after trades."""
    if debug:
        print("Comparing team performance:")
        print("Percentiles data head:")
        print(percentiles.head())
        print("Percentiles columns:")
        print(percentiles.columns)
        print("Average champion stats:")
        print(average_champion_stats)

    comparison_data = []
    
    for team in traded_teams:
        if debug:
            print(f"Processing team: {team}")
        
        pre_trade_stats = percentiles[(percentiles['TEAM_NAME'] == team) & (percentiles['PERIOD'] == 'Pre-trade')]
        post_trade_stats = percentiles[(percentiles['TEAM_NAME'] == team) & (percentiles['PERIOD'] == 'Post-trade')]
        
        if not pre_trade_stats.empty and not post_trade_stats.empty:
            team_comparison = {'Team': team}
            for stat in RELEVANT_STATS + ['eFG%']:
                if debug:
                    print(f"Processing stat: {stat}")
                    print(f"Pre-trade stats columns: {pre_trade_stats.columns}")
                    print(f"Post-trade stats columns: {post_trade_stats.columns}")
                
                per_game_col = f'{stat}_per_game'
                percentile_col = f'{stat}_percentile'
                
                # Pre-trade stats
                if per_game_col in pre_trade_stats.columns:
                    team_comparison[f'{stat} Pre-trade'] = pre_trade_stats[per_game_col].values[0]
                else:
                    print(f"Warning: {per_game_col} not found in pre_trade_stats")
                    team_comparison[f'{stat} Pre-trade'] = None
                
                if percentile_col in pre_trade_stats.columns:
                    team_comparison[f'{stat} Pre-trade Percentile'] = pre_trade_stats[percentile_col].values[0]
                else:
                    print(f"Warning: {percentile_col} not found in pre_trade_stats")
                    team_comparison[f'{stat} Pre-trade Percentile'] = None
                
                # Post-trade stats
                if per_game_col in post_trade_stats.columns:
                    team_comparison[f'{stat} Post-trade'] = post_trade_stats[per_game_col].values[0]
                else:
                    print(f"Warning: {per_game_col} not found in post_trade_stats")
                    team_comparison[f'{stat} Post-trade'] = None
                
                if percentile_col in post_trade_stats.columns:
                    team_comparison[f'{stat} Post-trade Percentile'] = post_trade_stats[percentile_col].values[0]
                else:
                    print(f"Warning: {percentile_col} not found in post_trade_stats")
                    team_comparison[f'{stat} Post-trade Percentile'] = None
                
                # Champion stats
                if per_game_col in average_champion_stats.columns:
                    team_comparison[f'{stat} Champion'] = average_champion_stats[per_game_col].values[0]
                else:
                    print(f"Warning: {per_game_col} not found in average_champion_stats")
                    team_comparison[f'{stat} Champion'] = None
            
            comparison_data.append(team_comparison)
        else:
            if debug:
                print(f"No data available for comparison for {team}.")
                print("Pre-trade stats head:")
                print(pre_trade_stats.head())
                print("Post-trade stats head:")
                print(post_trade_stats.head())

    comparison_df = pd.DataFrame(comparison_data)

    if debug:
        print("\nComparison Results:")
        print(comparison_df)

    return comparison_df

def validate_post_trade_stats(player_data, trade_date, traded_teams, post_trade_stats, debug=False):
    """Validate the post-trade statistics calculation."""
    trade_date = pd.to_datetime(trade_date)
    post_trade_data = player_data[player_data['GAME_DATE'] >= trade_date]

    validation_results = {}

    for team in traded_teams:
        team_data = post_trade_data[post_trade_data['TEAM_NAME'] == team]
        
        total_points = team_data['PTS'].sum()
        games_played = team_data['GAME_ID'].nunique()
        calculated_ppg = total_points / games_played if games_played > 0 else 0

        reported_ppg = post_trade_stats[post_trade_stats['TEAM_NAME'] == team]['PTS_per_game'].values[0]

        validation_results[team] = {
            'Calculated PPG': calculated_ppg,
            'Reported PPG': reported_ppg,
            'Difference': calculated_ppg - reported_ppg,
            'Games Played': games_played
        }

    if debug:
        print("\nPost-Trade Statistics Validation:")
        print(tabulate(pd.DataFrame(validation_results).T, headers='keys', tablefmt='grid'))

    return validation_results

import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.static import teams

def load_team_data():
    nba_teams = teams.get_teams()
    team_df = pd.DataFrame(nba_teams)
    return team_df[['id', 'full_name', 'abbreviation']]

def load_player_data(start_year, end_year):
    player_data = pd.DataFrame()
    for year in range(start_year, end_year + 1):
        data = fetch_season_data_by_year(year)
        if data is not None:
            player_data = pd.concat([player_data, data], ignore_index=True)
    return player_data

def trade_impact_simulator():
    st.subheader("NBA Trade Impact Simulator")

    # Load team and player data
    team_data = load_team_data()
    player_data = load_player_data(2020, 2023)  # Adjust years as needed

    # User inputs
    trade_date = st.date_input('Trade Date', datetime(2023, 12, 20))

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox('Select Team 1', team_data['full_name'].tolist())
    with col2:
        team2 = st.selectbox('Select Team 2', team_data['full_name'].tolist(), index=1)

    team1_players = player_data[player_data['TEAM_NAME'] == team1]['PLAYER_NAME'].unique()
    team2_players = player_data[player_data['TEAM_NAME'] == team2]['PLAYER_NAME'].unique()

    col1, col2 = st.columns(2)
    with col1:
        players1 = st.multiselect(f'Select Players from {team1}', team1_players)
    with col2:
        players2 = st.multiselect(f'Select Players from {team2}', team2_players)

    if st.button('Simulate Trade'):
        # Convert trade_date to pandas Timestamp for comparison
        trade_date = pd.Timestamp(trade_date)

        # Prepare traded players data
        traded_players = {}
        for player in players1:
            player_id = player_data[player_data['PLAYER_NAME'] == player]['PLAYER_ID'].iloc[0]
            traded_players[player_id] = (team2, player)
        for player in players2:
            player_id = player_data[player_data['PLAYER_NAME'] == player]['PLAYER_ID'].iloc[0]
            traded_players[player_id] = (team1, player)

        # Fetch champion data
        champions = get_champions(2020, 2023)

        # Process champion team data
        champion_team_data = process_champion_team_data(player_data, champions)

        # Calculate pre-trade and post-trade team statistics
        pre_trade_team_stats = calculate_team_stats(player_data[player_data['GAME_DATE'] < trade_date], 'Pre-trade')
        post_trade_team_stats = calculate_post_trade_team_stats(player_data, traded_players, trade_date, player_data)

        # Combine pre-trade and post-trade stats
        combined_stats = pd.concat([pre_trade_team_stats, post_trade_team_stats], ignore_index=True)

        # Calculate eFG% for the combined dataset
        combined_stats['eFG%_per_game'] = (combined_stats['FGM_per_game'] + 0.5 * combined_stats['FG3M_per_game']) / combined_stats['FGA_per_game']

        # Calculate percentiles for the combined stats
        combined_stats = calculate_percentiles(combined_stats)

        # Calculate average champion stats
        average_champion_stats = calculate_average_champion_stats(champion_team_data)

        # Compare pre-trade and post-trade stats for traded teams
        traded_teams = [team1, team2]
        comparison_table = compare_team_performance(combined_stats, average_champion_stats, traded_teams)

        # Display the comparison table
        st.subheader('Trade Impact Comparison')
        st.dataframe(comparison_table)

        # Visualize the results
        st.subheader('Visual Comparison')
        metric = st.selectbox('Select Metric', ['PTS', 'AST', 'TOV', 'STL', 'BLK', 'OREB', 'DREB', 'FGM', 'FG3M', 'FGA', 'eFG%'])

        fig = go.Figure()
        for team in traded_teams:
            team_data = comparison_table[comparison_table['Team'] == team]
            fig.add_trace(go.Bar(x=[f'{team} Pre-trade'], y=[team_data[f'{metric} Pre-trade'].values[0]], name=f'{team} Pre-trade'))
            fig.add_trace(go.Bar(x=[f'{team} Post-trade'], y=[team_data[f'{metric} Post-trade'].values[0]], name=f'{team} Post-trade'))
            fig.add_trace(go.Bar(x=[f'{team} Champion'], y=[team_data[f'{metric} Champion'].values[0]], name=f'{team} Champion'))

        fig.update_layout(title=f'{metric} Comparison', xaxis_title='Team', yaxis_title=metric)
        st.plotly_chart(fig)


def main(debug=True):
    start_year = 2020
    end_year = 2023
    trade_date = '2023-12-20'  # Example trade date
    
    # Traded players with new team names
    traded_players = {
        1628369: ('Los Angeles Lakers', 'Jayson Tatum'),  # Example Player ID and new team
        1630559: ('Boston Celtics', 'Austin Reaves')      # Example Player ID and new team
    }
    
    # Fetch player names
    for player_id in traded_players.keys():
        player_info = fetch_player_info(player_id, debug)
        if player_info is not None:
            traded_players[player_id] = (traded_players[player_id][0], player_info['DISPLAY_FIRST_LAST'].values[0])
    
    # Fetch champion data
    champions = get_champions(start_year, end_year, debug)
    
    # Fetch player data for each season
    player_data = pd.DataFrame()
    season_data = pd.DataFrame()  # To store the full season data
    for year in range(start_year, end_year + 1):
        data = fetch_season_data_by_year(year, debug)
        if data is not None:
            player_data = pd.concat([player_data, data], ignore_index=True)
            season_data = player_data  # Assuming season_data should hold the entire season's data

    if player_data.empty:
        print("Failed to fetch player data. Exiting.")
        return

    # Process champion team data
    champion_team_data = process_champion_team_data(player_data, champions, debug)

    if debug:
        print("\nChampion Team Stats and Percentiles:")
        print(tabulate(champion_team_data, headers='keys', tablefmt='grid'))

    # Debug: Print pre-trade stats for traded players and their teams
    if debug:
        print("\nPre-trade stats:")
        for player_id, (new_team_name, player_name) in traded_players.items():
            # Use all available data if trade date is before the season starts
            if player_data['GAME_DATE'].min() > pd.to_datetime(trade_date):
                player_pre_trade = player_data[player_data['PLAYER_ID'] == player_id]
            else:
                player_pre_trade = player_data[(player_data['PLAYER_ID'] == player_id) & (player_data['GAME_DATE'] < pd.to_datetime(trade_date))]
            
            if not player_pre_trade.empty:
                old_team_name = player_pre_trade['TEAM_NAME'].iloc[0]
                player_total_points = player_pre_trade['PTS'].sum()
                team_total_points = player_data[(player_data['TEAM_NAME'] == old_team_name) & (player_data['GAME_DATE'] < pd.to_datetime(trade_date))]['PTS'].sum()
                print(f"{player_name} (Old team: {old_team_name}):")
                print(f"  Player total points: {player_total_points}")
                print(f"  Team total points: {team_total_points}")
            else:
                print(f"No data available for {player_name}.")

    # Calculate pre-trade and post-trade team statistics
    if player_data['GAME_DATE'].min() > pd.to_datetime(trade_date):
        pre_trade_team_stats = calculate_team_stats(player_data, 'Pre-trade', debug)
    else:
        pre_trade_team_stats = calculate_team_stats(player_data[player_data['GAME_DATE'] < pd.to_datetime(trade_date)], 'Pre-trade', debug)
        
    post_trade_team_stats = calculate_post_trade_team_stats(player_data, traded_players, trade_date, season_data, debug)

    # Debug: Print post-trade stats for traded players and their new teams
    if debug:
        print("\nPost-trade stats:")
        for player_id, (new_team_name, player_name) in traded_players.items():
            player_post_trade = player_data[(player_data['PLAYER_ID'] == player_id) & (player_data['GAME_DATE'] >= pd.to_datetime(trade_date))]
            if not player_post_trade.empty:
                player_total_points = player_post_trade['PTS'].sum()
                team_total_points = player_data[(player_data['TEAM_NAME'] == new_team_name) & (player_data['GAME_DATE'] >= pd.to_datetime(trade_date))]['PTS'].sum()
                print(f"{player_name} (New team: {new_team_name}):")
                print(f"  Player total points: {player_total_points}")
                print(f"  Team total points: {team_total_points}")
            else:
                print(f"No post-trade data found for {player_name}.")

    # Combine pre-trade and post-trade stats
    combined_stats = pd.concat([pre_trade_team_stats, post_trade_team_stats], ignore_index=True)

    # Calculate eFG% for the combined dataset
    combined_stats['eFG%_per_game'] = (combined_stats['FGM_per_game'] + 0.5 * combined_stats['FG3M_per_game']) / combined_stats['FGA_per_game']

    # Calculate percentiles for the combined stats
    percentiles = calculate_percentiles(combined_stats, debug)
    
    if debug:
        print("\nCombined Team Stats and Percentiles:")
        print(tabulate(percentiles, headers='keys', tablefmt='grid'))
    
    # Calculate average champion stats
    average_champion_stats = calculate_average_champion_stats(champion_team_data, debug)

    # Compare pre-trade and post-trade stats for traded teams
    traded_teams = list(set([team_name for _, (team_name, _) in traded_players.items()]))
    comparison_table = compare_team_performance(percentiles, average_champion_stats, traded_teams, debug)
    
    # Print the comparison table
    if debug:
        print("\nTrade Impact Comparison:")
        print(tabulate(comparison_table, headers='keys', tablefmt='grid'))

    # Validate post-trade statistics
    validation_results = validate_post_trade_stats(player_data, trade_date, traded_teams, post_trade_team_stats, debug)

    return validation_results

if __name__ == "__main__":
    main(debug=True)
