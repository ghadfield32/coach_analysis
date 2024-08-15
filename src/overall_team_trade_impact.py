
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelogs, leaguegamefinder
from tabulate import tabulate
import time
from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.static import teams, players

# Constants
RELEVANT_STATS = ['PTS', 'AST', 'TOV', 'STL', 'BLK', 'OREB', 'DREB', 'FGM', 'FG3M', 'FGA']

def load_team_data():
    nba_teams = teams.get_teams()
    team_df = pd.DataFrame(nba_teams)
    return team_df[['id', 'full_name', 'abbreviation']]

def fetch_player_id_by_name(player_name, debug=False):
    """Fetch player ID based on player name."""
    try:
        player = players.find_players_by_full_name(player_name)[0]
        if debug:
            print(f"Fetched ID for player {player_name}: {player['id']}")
        return player['id']
    except Exception as e:
        if debug:
            print(f"Error fetching ID for player {player_name}: {e}")
        return None

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

def fetch_season_data_by_year(year):
    """Fetch player game logs data for a given starting year of the NBA season."""
    season = f"{year}-{str(year+1)[-2:]}"
    player_logs = playergamelogs.PlayerGameLogs(season_nullable=season).get_data_frames()[0]
    player_logs['SEASON'] = season
    player_logs['GAME_DATE'] = pd.to_datetime(player_logs['GAME_DATE'])
    return player_logs

def calculate_team_averages(season_data):
    """Calculate team averages for the relevant stats."""
    team_stats = (
        season_data.groupby(['SEASON', 'TEAM_NAME'])[RELEVANT_STATS]
        .sum()
        .reset_index()
    )
    games_played = season_data.groupby(['SEASON', 'TEAM_NAME'])['GAME_ID'].nunique().reset_index(name='GAMES_PLAYED')
    team_stats = pd.merge(team_stats, games_played, on=['SEASON', 'TEAM_NAME'])

    for stat in RELEVANT_STATS:
        team_stats[f'{stat}_per_game'] = team_stats[stat] / team_stats['GAMES_PLAYED']
    
    team_stats['eFG%_per_game'] = (
        (team_stats['FGM_per_game'] + 0.5 * team_stats['FG3M_per_game']) / team_stats['FGA_per_game']
    )
    
    return team_stats

def calculate_percentiles(stats_df, debug=False):
    """Calculate percentiles for the stats within each season."""
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

def get_champion_team_stats(start_year, end_year):
    """Fetch and process champion team stats for the given range of years."""
    all_team_stats = pd.DataFrame()

    for year in range(start_year, end_year + 1):
        season_data = fetch_season_data_by_year(year)
        team_stats = calculate_team_averages(season_data)
        team_stats = calculate_percentiles(team_stats)
        
        # Identify the champion team
        champ_name = get_champion(f"{year}-{str(year+1)[-2:]}")
        if champ_name:
            champ_stats = team_stats[team_stats['TEAM_NAME'] == champ_name]
            all_team_stats = pd.concat([all_team_stats, champ_stats])

    # Calculate the average stats for the champions (only numeric columns)
    numeric_cols = all_team_stats.select_dtypes(include=[np.number]).columns
    avg_stats = all_team_stats[numeric_cols].mean()

    # Create a DataFrame for the average stats and append to the results
    avg_stats_df = pd.DataFrame([avg_stats])
    avg_stats_df['SEASON'] = 'Average'
    avg_stats_df['TEAM_NAME'] = 'Average Champion'
    
    # Append the average row to the champion team data
    all_team_stats = pd.concat([all_team_stats, avg_stats_df], ignore_index=True)
    
    return all_team_stats

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
    for player_name, new_team_name in traded_players.items():
        player_id = fetch_player_id_by_name(player_name, debug)
        player_post_trade_stats = post_trade_data[post_trade_data['PLAYER_ID'] == player_id][RELEVANT_STATS].mean()
        traded_player_stats[player_name] = player_post_trade_stats.to_dict()
        if debug:
            print(f"{player_name} averages post-trade (to {new_team_name}): {traded_player_stats[player_name]}")

    # Adjust post-trade stats based on traded players
    for player_name, new_team_name in traded_players.items():
        player_id = fetch_player_id_by_name(player_name, debug)
        old_team_name = player_data[player_data['PLAYER_ID'] == player_id]['TEAM_NAME'].iloc[0]
        post_trade_games = post_trade_stats.loc[post_trade_stats['TEAM_NAME'] == new_team_name, 'GAMES_PLAYED'].values[0]

        if debug:
            print(f"\nAdjusting stats for trade: {player_name} from {old_team_name} to {new_team_name}")

        # Remove player's stats from old team
        for stat in RELEVANT_STATS:
            # Ensure the column is a float before performing the operation
            post_trade_stats[stat] = post_trade_stats[stat].astype(float)
            if debug:
                print(f"  Before adjustment - {old_team_name} {stat}: {post_trade_stats.loc[post_trade_stats['TEAM_NAME'] == old_team_name, stat].values[0]}")
            post_trade_stats.loc[post_trade_stats['TEAM_NAME'] == old_team_name, stat] -= traded_player_stats[player_name][stat] * post_trade_games
            if debug:
                print(f"  After adjustment - {old_team_name} {stat}: {post_trade_stats.loc[post_trade_stats['TEAM_NAME'] == old_team_name, stat].values[0]}")

        # Add player's stats to new team
        for stat in RELEVANT_STATS:
            if debug:
                print(f"  Before adjustment - {new_team_name} {stat}: {post_trade_stats.loc[post_trade_stats['TEAM_NAME'] == new_team_name, stat].values[0]}")
            post_trade_stats.loc[post_trade_stats['TEAM_NAME'] == new_team_name, stat] += traded_player_stats[player_name][stat] * post_trade_games
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


def compare_team_performance(percentiles, champion_team_data, traded_teams, champion_filter='Average Champion', debug=False):
    """Generate a comparison table for team performance before and after trades."""
    if debug:
        print("Comparing team performance:")
        print("Percentiles data head:")
        print(percentiles.head())
    
    # Filter for the selected champion
    if champion_filter == 'Average Champion':
        champion_row = champion_team_data[champion_team_data['TEAM_NAME'] == 'Average Champion'].iloc[0]
    else:
        champion_row = champion_team_data[(champion_team_data['SEASON'] == champion_filter) & (champion_team_data['TEAM_NAME'] != 'Average Champion')].iloc[0]

    comparison_data = []
    
    for team in traded_teams:
        pre_trade_stats = percentiles[(percentiles['TEAM_NAME'] == team) & (percentiles['PERIOD'] == 'Pre-trade')]
        post_trade_stats = percentiles[(percentiles['TEAM_NAME'] == team) & (percentiles['PERIOD'] == 'Post-trade')]
        
        if not pre_trade_stats.empty and not post_trade_stats.empty:
            team_comparison = {'Team': team}
            for stat in RELEVANT_STATS + ['eFG%']:
                team_comparison[f'{stat} Pre-trade'] = pre_trade_stats[f'{stat}_per_game'].values[0]
                team_comparison[f'{stat} Pre-trade Percentile'] = pre_trade_stats[f'{stat}_percentile'].values[0]
                team_comparison[f'{stat} Post-trade'] = post_trade_stats[f'{stat}_per_game'].values[0]
                team_comparison[f'{stat} Post-trade Percentile'] = post_trade_stats[f'{stat}_percentile'].values[0]
                team_comparison[f'{stat} Champion'] = champion_row[f'{stat}_per_game']
                team_comparison[f'{stat} Champion Percentile'] = champion_row[f'{stat}_percentile']
            
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
        print(tabulate(comparison_df, headers='keys', tablefmt='grid'))

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

def fetch_and_process_champion_data(start_year, end_year, debug=False):
    """Fetch and process champion data for the given range of years."""
    champions = get_champions(start_year, end_year, debug)
    champion_team_data = get_champion_team_stats(start_year, end_year)
    
    if debug:
        print("\nChampion Team Stats with Average:")
        print(tabulate(champion_team_data, headers='keys', tablefmt='grid'))
    
    return champion_team_data

def fetch_and_process_player_data(start_year, end_year, debug=False):
    """Fetch and combine player data for the given range of years."""
    player_data = pd.DataFrame()
    season_data = pd.DataFrame()  # To store the full season data

    for year in range(start_year, end_year + 1):
        data = fetch_season_data_by_year(year)
        if data is not None:
            player_data = pd.concat([player_data, data], ignore_index=True)
            season_data = player_data  # Assuming season_data should hold the entire season's data
    
    if player_data.empty:
        raise ValueError("Failed to fetch player data.")
    
    if debug:
        print("\nFetched Player Data:")
        print(player_data.head())
    
    return player_data, season_data

def calculate_combined_team_stats(player_data, trade_date, traded_players, season_data, debug=False):
    """Calculate combined pre-trade and post-trade team statistics."""
    trade_date = pd.to_datetime(trade_date)

    if player_data['GAME_DATE'].min() > trade_date:
        pre_trade_team_stats = calculate_team_stats(player_data, 'Pre-trade', debug)
    else:
        pre_trade_team_stats = calculate_team_stats(player_data[player_data['GAME_DATE'] < trade_date], 'Pre-trade', debug)
        
    post_trade_team_stats = calculate_post_trade_team_stats(player_data, traded_players, trade_date, season_data, debug)
    
    combined_stats = pd.concat([pre_trade_team_stats, post_trade_team_stats], ignore_index=True)
    
    combined_stats['eFG%_per_game'] = (
        (combined_stats['FGM_per_game'] + 0.5 * combined_stats['FG3M_per_game']) / combined_stats['FGA_per_game']
    )
    
    if debug:
        print("\nCombined Team Stats:")
        print(tabulate(combined_stats, headers='keys', tablefmt='grid'))
    
    return combined_stats

def trade_impact_analysis(start_year, end_year, trade_date, traded_players, champion_filter='Average Champion', debug=False):
    """Perform trade impact analysis and return the comparison table."""
    # Fetch and process data
    champion_team_data = fetch_and_process_champion_data(start_year, end_year, debug)
    player_data, season_data = fetch_and_process_player_data(start_year, end_year, debug)
    
    # Calculate combined team stats
    combined_stats = calculate_combined_team_stats(player_data, trade_date, traded_players, season_data, debug)
    
    # Calculate percentiles for combined stats
    percentiles = calculate_percentiles(combined_stats, debug)
    
    # Compare pre-trade and post-trade stats for traded teams
    traded_teams = list(set([team_name for _, team_name in traded_players.items()]))
    comparison_table = compare_team_performance(percentiles, champion_team_data, traded_teams, champion_filter, debug)
    
    if debug:
        print("\nTrade Impact Comparison:")
        print(tabulate(comparison_table, headers='keys', tablefmt='grid'))
    
    return comparison_table

def main(debug=True):
    start_year = 2023
    end_year = 2023
    trade_date = '2023-4-20'  # Example trade date
    
    # Traded players with new team names
    traded_players = {
        'Jayson Tatum': 'Boston Celtics',  # Example Player and new team
        'Devin Booker': 'Phoenix Suns'     # Example Player and new team
    }
    
    # Perform trade impact analysis
    comparison_table = trade_impact_analysis(
        start_year, end_year, trade_date, traded_players, 
        champion_filter='Average Champion', debug=debug
    )
    
    # Print the comparison table
    print(comparison_table)
    

if __name__ == "__main__":
    main(debug=True)


