
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelogs, leaguegamefinder
from nba_api.stats.static import players, teams
import os
import pickle

# Set the cache directory and file path
CACHE_DIR = "../data/processed/"
CACHE_FILE = os.path.join(CACHE_DIR, "champion_stats_cache.pkl")

# Ensure the directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def load_cache():
    """Load the cached champion stats if available."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
        return cache
    return {}

def save_cache(cache):
    """Save the champion stats to cache."""
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)

def get_champion(season, debug=False):
    """Fetch the champion team for a given NBA season (cached + retried)."""
    from trade_impact.utils.nba_api_utils import get_champion_team_name, normalize_season
    season_norm = normalize_season(season)
    winner = get_champion_team_name(season_norm, timeout=90, retries=3, use_live=True, debug=debug)
    if debug:
        print(f"Champion for season {season_norm}: {winner}")
    return winner


def get_champion_team_stats(seasons, relevant_stats, debug=False):
    """Fetch and process champion team stats for the selected seasons, using caching."""
    # Load the cache
    cache = load_cache()

    all_team_stats = pd.DataFrame()

    # List of seasons that need to be fetched (not in cache)
    missing_seasons = [season for season in seasons if season not in cache]

    if missing_seasons:
        if debug:
            print(f"Fetching data for missing seasons: {missing_seasons}")
        
        # Fetch data for missing seasons
        for season in missing_seasons:
            season_data = fetch_season_data_by_year(season, debug)
            if season_data is None:
                continue  # Skip if no data

            team_stats = calculate_team_stats(season_data, 'No-trade', relevant_stats, debug)
            team_stats = calculate_percentiles(team_stats, relevant_stats, debug)
            
            # Identify the champion team
            champ_name = get_champion(season, debug)
            if champ_name:
                champ_stats = team_stats[team_stats['TEAM_NAME'] == champ_name]
                cache[season] = champ_stats  # Store the stats in the cache

        # Save the updated cache
        save_cache(cache)

    # Collect data from the cache for the requested seasons
    for season in seasons:
        if season in cache:
            all_team_stats = pd.concat([all_team_stats, cache[season]])

    # Calculate average champion stats
    if not all_team_stats.empty:
        numeric_cols = all_team_stats.select_dtypes(include=[np.number]).columns
        average_champion = all_team_stats[numeric_cols].mean().to_frame().T
        average_champion['TEAM_NAME'] = 'Average Champion'
        average_champion['SEASON'] = 'Multiple Seasons'
        all_team_stats = pd.concat([all_team_stats, average_champion])

    return all_team_stats


def fetch_player_id_by_name(player_name, debug=False):
    try:
        player = players.find_players_by_full_name(player_name)[0]
        if debug:
            print(f"Fetched ID for player {player_name}: {player['id']}")
        return player['id']
    except Exception as e:
        if debug:
            print(f"Error fetching ID for player {player_name}: {e}")
        return None

def fetch_season_data_by_year(year, debug=False):
    """Fetch league-wide player game logs for a given year or season string using retries/cache."""
    from trade_impact.utils.nba_api_utils import get_playergamelogs_df, normalize_season
    season_norm = normalize_season(year)
    if debug:
        print(f"Fetching data for season: {season_norm}")
    try:
        player_logs = get_playergamelogs_df(season_norm, timeout=90, retries=3, use_live=True, debug=debug)
    except Exception as e:
        if debug:
            print(f"Error fetching data for season {season_norm}: {e}")
        return None
    player_logs['SEASON'] = season_norm
    player_logs['GAME_DATE'] = pd.to_datetime(player_logs['GAME_DATE'])
    if debug:
        print(f"Fetched season data with {len(player_logs)} records.")
    return player_logs



def calculate_team_stats(player_data, period, relevant_stats, debug=False):
    if player_data.empty:
        if debug:
            print(f"No data available for {period}. Returning 'N/A' values.")
        return pd.DataFrame({"SEASON": ["N/A"], "TEAM_NAME": ["N/A"], "GAMES_PLAYED": ["N/A"], **{f'{stat}_per_game': ["N/A"] for stat in relevant_stats}})

    missing_stats = [stat for stat in relevant_stats if stat not in player_data.columns]
    if missing_stats:
        raise KeyError(f"Missing columns in player_data: {missing_stats}")

    if debug:
        print(f"Calculating {period} team-level statistics.")
    
    valid_player_data = player_data.dropna(subset=relevant_stats)
    
    if valid_player_data.empty:
        if debug:
            print(f"No valid data after dropping NA for {period}. Returning 'N/A' values.")
        return pd.DataFrame({"SEASON": ["N/A"], "TEAM_NAME": ["N/A"], "GAMES_PLAYED": ["N/A"], **{f'{stat}_per_game': ["N/A"] for stat in relevant_stats}})
    
    team_stats = (
        valid_player_data.groupby(['SEASON', 'TEAM_NAME'])[relevant_stats]
        .sum()
        .reset_index()
    )
    
    games_played = valid_player_data.groupby(['SEASON', 'TEAM_NAME'])['GAME_ID'].nunique().reset_index(name='GAMES_PLAYED')
    
    team_stats = pd.merge(team_stats, games_played, on=['SEASON', 'TEAM_NAME'])
    for stat in relevant_stats:
        team_stats[f'{stat}_per_game'] = team_stats[stat] / team_stats['GAMES_PLAYED']
    
    team_stats['PERIOD'] = period
    
    if debug:
        print(f"{period} team-level statistics:")
        display_cols = ['SEASON', 'TEAM_NAME', 'GAMES_PLAYED'] + [f'{stat}_per_game' for stat in relevant_stats]
        print(team_stats[display_cols].head(), "\n")
    
    return team_stats


def calculate_percentiles(stats_df, relevant_stats, debug=False):
    if debug:
        print("Calculating percentiles for each team and season.\n")
    
    for season in stats_df['SEASON'].unique():
        season_data = stats_df[stats_df['SEASON'] == season]
        for stat in relevant_stats:
            stat_per_game = f'{stat}_per_game'
            if stat_per_game in season_data.columns:
                percentile_col = f'{stat}_percentile'
                stats_df.loc[stats_df['SEASON'] == season, percentile_col] = season_data[stat_per_game].rank(pct=True)
                if debug:
                    print(f"Calculated percentiles for {stat} in season {season}:")
                    print(stats_df.loc[stats_df['SEASON'] == season, [stat_per_game, percentile_col]].head(), "\n")
    
    return stats_df


def calculate_player_averages(post_trade_data, traded_players, relevant_stats, debug=False):
    player_averages = {}
    for player_name, new_team_name in traded_players.items():
        player_id = fetch_player_id_by_name(player_name, debug)
        if player_id is None:
            if debug:
                print(f"Skipping player {player_name} due to missing ID.")
            continue
        
        # Calculate average stats for the player post-trade
        player_data = post_trade_data[post_trade_data['PLAYER_ID'] == player_id]
        if player_data.empty:
            if debug:
                print(f"No post-trade data found for player {player_name}.")
            continue
        
        avg_stats = player_data[relevant_stats].mean()
        player_averages[player_id] = avg_stats
        if debug:
            print(f"Averages for {player_name} after trade: {avg_stats.to_dict()}")
    
    return player_averages

def simulate_game_logs(post_trade_data, player_averages, traded_players, no_trade_data, trade_date, relevant_stats, debug=False):
    simulated_logs_list = []  # Use a list to collect simulated logs
    
    for player_name, new_team_name in traded_players.items():
        player_id = fetch_player_id_by_name(player_name, debug)
        if player_id is None or player_id not in player_averages:
            if debug:
                print(f"Skipping simulation for player {player_name}.")
            continue
        
        # Remove original player's logs from the post-trade dataset
        post_trade_data = post_trade_data[post_trade_data['PLAYER_ID'] != player_id]
        
        # Get the team's unique schedule post-trade (one entry per game)
        team_schedule = no_trade_data[
            (no_trade_data['TEAM_NAME'] == new_team_name) & 
            (no_trade_data['GAME_DATE'] >= trade_date)
        ].drop_duplicates(subset=['GAME_ID', 'TEAM_NAME'])
        
        if team_schedule.empty:
            if debug:
                print(f"No games found for team {new_team_name} after trade date {trade_date}.")
            continue
        
        # Create simulated logs based on the player's average stats
        for _, game in team_schedule.iterrows():
            simulated_log = {
                'SEASON': game['SEASON'],
                'PLAYER_ID': player_id,
                'PLAYER_NAME': player_name,
                'TEAM_ID': game['TEAM_ID'],
                'TEAM_ABBREVIATION': game['TEAM_ABBREVIATION'],
                'TEAM_NAME': new_team_name,
                'GAME_ID': game['GAME_ID'],
                'GAME_DATE': game['GAME_DATE'],
                'MATCHUP': game['MATCHUP'],
                **{stat: player_averages[player_id][stat] for stat in relevant_stats}
            }
            simulated_logs_list.append(simulated_log)
        
        if debug:
            print(f"Simulated {len(team_schedule)} logs for {player_name} with {new_team_name}.\n")
    
    # Combine the simulated logs with the original post-trade data
    if simulated_logs_list:
        simulated_logs = pd.DataFrame(simulated_logs_list)
        if debug:
            print(f"Total simulated logs created: {len(simulated_logs)}")
            print(simulated_logs.head(), "\n")
        
        post_trade_data = pd.concat([post_trade_data, simulated_logs], ignore_index=True)
    
    if debug:
        print(f"Post-trade data now has {len(post_trade_data)} records after simulation.\n")
    
    return post_trade_data



def trade_impact_analysis(start_season, end_season, trade_date, traded_players, team_a_name, team_b_name, champion_seasons, relevant_stats, debug=False):
    player_data = pd.DataFrame()

    # Fetch full season data
    start_year = int(start_season.split('-')[0])
    end_year = int(end_season.split('-')[0])
    for season in range(start_year, end_year + 1):
        season_str = f"{season}-{str(season + 1)[-2:]}"
        data = fetch_season_data_by_year(season_str, debug)
        if data is not None:
            player_data = pd.concat([player_data, data], ignore_index=True)
    
    if debug:
        print(f"\nTotal player data records: {len(player_data)}")
        print(f"Sample data:\n{player_data.head()}\n")
    
    # Convert trade date to datetime
    trade_date = pd.to_datetime(trade_date)
    trade_month = trade_date.month
    
    # NBA season typically runs from October (month 10) to June (month 6)
    in_season_trade = trade_month in [10, 11, 12, 1, 2, 3, 4, 5, 6]

    # Determine if the trade is during the season or offseason
    if not in_season_trade:
        if debug:
            print("Trade date is in the offseason. Considering the full season for analysis.")
        pre_trade_data = pd.DataFrame()  # No pre-trade data since it's offseason
        post_trade_data = player_data.copy()  # Consider the full season as post-trade
    else:
        # Step 1: Create pre-trade and post-trade datasets
        pre_trade_data = player_data[player_data['GAME_DATE'] < trade_date].copy()
        post_trade_data = player_data[player_data['GAME_DATE'] >= trade_date].copy()

        if debug:
            pre_trade_points = pre_trade_data['PTS'].sum()
            pre_trade_games = pre_trade_data['GAME_ID'].nunique()
            print("Pre-trade Dataset:")
            print(f"Total Points: {pre_trade_points}")
            print(f"Total Games Played: {pre_trade_games}")
        
            post_trade_points = post_trade_data['PTS'].sum()
            post_trade_games = post_trade_data['GAME_ID'].nunique()
            print("Post-trade Dataset:")
            print(f"Total Points: {post_trade_points}")
            print(f"Total Games Played: {post_trade_games}\n")
    
    if pre_trade_data.empty and in_season_trade:
        print("No data available before the trade date.")
        pre_trade_stats = pd.DataFrame({"TEAM_NAME": [team_a_name, team_b_name], "GAMES_PLAYED": ["N/A", "N/A"], **{f"{stat}_per_game": ["N/A", "N/A"] for stat in relevant_stats}})
    else:
        pre_trade_stats = calculate_team_stats(pre_trade_data, 'Pre-trade', relevant_stats, debug)
        pre_trade_stats = calculate_percentiles(pre_trade_stats, relevant_stats, debug)

    # Get champion data for the selected seasons
    champion_team_data = get_champion_team_stats(champion_seasons, relevant_stats, debug)
    
    # Step 2: Calculate and simulate player averages post-trade
    for player_name, new_team_name in traded_players.items():
        player_averages = calculate_player_averages(post_trade_data, {player_name: new_team_name}, relevant_stats, debug)
        post_trade_data = simulate_game_logs(post_trade_data, player_averages, {player_name: new_team_name}, player_data, trade_date, relevant_stats, debug)
    
    # Step 3: Recalculate team statistics after all player simulations
    post_trade_stats = calculate_team_stats(post_trade_data, 'Post-trade', relevant_stats, debug)
    post_trade_stats = calculate_percentiles(post_trade_stats, relevant_stats, debug)

    no_trade_stats = calculate_team_stats(player_data, 'No-trade', relevant_stats, debug)
    no_trade_stats = calculate_percentiles(no_trade_stats, relevant_stats, debug)

    overall_trade_stats = calculate_team_stats(pd.concat([pre_trade_data, post_trade_data]), 'Overall-trade', relevant_stats, debug)
    overall_trade_stats = calculate_percentiles(overall_trade_stats, relevant_stats, debug)
    
    # Step 4: Final Comparison - Create a separate table for each metric
    comparison_tables = {}

    for stat in relevant_stats:
        comparison_data = []

        for team in [team_a_name, team_b_name]:
            pre_trade = pre_trade_stats[pre_trade_stats['TEAM_NAME'] == team]
            post_trade = post_trade_stats[post_trade_stats['TEAM_NAME'] == team]
            no_trade = no_trade_stats[no_trade_stats['TEAM_NAME'] == team]
            overall_trade = overall_trade_stats[overall_trade_stats['TEAM_NAME'] == team]
            
            if pre_trade.empty:
                pre_trade = pd.DataFrame({"GAMES_PLAYED": ["N/A"], f'{stat}_per_game': ["N/A"], f'{stat}_percentile': ["N/A"]})
            if post_trade.empty:
                post_trade = pd.DataFrame({"GAMES_PLAYED": ["N/A"], f'{stat}_per_game': ["N/A"], f'{stat}_percentile': ["N/A"]})
            if no_trade.empty:
                no_trade = pd.DataFrame({"GAMES_PLAYED": ["N/A"], f'{stat}_per_game': ["N/A"], f'{stat}_percentile': ["N/A"]})
            if overall_trade.empty:
                overall_trade = pd.DataFrame({"GAMES_PLAYED": ["N/A"], f'{stat}_per_game': ["N/A"], f'{stat}_percentile': ["N/A"]})

            # Build the comparison data
            comparison_entry = {
                'Team': team,
                'Pre-trade Metric': pre_trade[f'{stat}_per_game'].values[0],
                'Post-trade Metric': post_trade[f'{stat}_per_game'].values[0],
                'Overall-trade Metric': overall_trade[f'{stat}_per_game'].values[0],
                'No-trade Metric': no_trade[f'{stat}_per_game'].values[0],
                'Champion Metric': champion_team_data[champion_team_data['TEAM_NAME'] == 'Average Champion'][f'{stat}_per_game'].max(),
                'Pre-trade Percentile': pre_trade[f'{stat}_percentile'].values[0],
                'Post-trade Percentile': post_trade[f'{stat}_percentile'].values[0],
                'Overall-trade Percentile': overall_trade[f'{stat}_percentile'].values[0],
                'No-trade Percentile': no_trade[f'{stat}_percentile'].values[0],
                'Champion Percentile': champion_team_data[champion_team_data['TEAM_NAME'] == 'Average Champion'][f'{stat}_percentile'].max(),
            }
            
            # Optionally include games and totals if debug is true
            if debug:
                comparison_entry.update({
                    'Pre-trade Games': pre_trade['GAMES_PLAYED'].values[0],
                    'Post-trade Games': post_trade['GAMES_PLAYED'].values[0],
                    'Overall-trade Games': overall_trade['GAMES_PLAYED'].values[0],
                    'No-trade Games': no_trade['GAMES_PLAYED'].values[0],
                    'Pre-trade Total': pre_trade[stat].sum() if stat in pre_trade.columns else "N/A",
                    'Post-trade Total': post_trade[stat].sum() if stat in post_trade.columns else "N/A",
                    'Overall-trade Total': overall_trade[stat].sum() if stat in overall_trade.columns else "N/A",
                    'No-trade Total': no_trade[stat].sum() if stat in no_trade.columns else "N/A",
                })

            comparison_data.append(comparison_entry)
        
        comparison_tables[stat] = pd.DataFrame(comparison_data)
        
        if debug:
            print(f"Comparison Table for {stat}:")
            print(comparison_tables[stat], "\n")
    
    return comparison_tables




def main(debug=True):
    start_season = "2023-24"
    end_season = "2023-24"
    trade_date = '2023-09-20'  # Adjusted trade date to be within the season
    
    # Team A and Team B selection
    team_a_name = "Dallas Mavericks"
    team_b_name = "Charlotte Hornets"
    
    # Fetch players for each team
    players_from_team_a = get_players_for_team(team_a_name, start_season)
    players_from_team_b = get_players_for_team(team_b_name, start_season)
    
    # Player selection - ensure these are split into lists
    selected_players_team_a = ["Grant Williams", "Seth Curry"]  # List of players from Dallas Mavericks
    selected_players_team_b = ["P.J. Washington"]  # List of players from Charlotte Hornets
    
    # Combine selected players into the traded_players dictionary
    traded_players = {player: team_b_name for player in selected_players_team_a}
    traded_players.update({player: team_a_name for player in selected_players_team_b})
    
    # Specify the seasons to consider for champions
    champion_seasons = ["2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2021-22", "2022-23", "2023-24"]
    
    # Adjust the relevant stats to analyze
    relevant_stats = ['PTS']  # This can be modified
    
    # Perform the trade impact analysis
    comparison_tables = trade_impact_analysis(
        start_season, end_season, trade_date, traded_players, team_a_name, team_b_name, champion_seasons, relevant_stats, debug=debug
    )
    
    # Print all comparison tables
    for stat, table in comparison_tables.items():
        print(f"Comparison Table for {stat}:")
        print(table)

if __name__ == "__main__":
    main(debug=True)


