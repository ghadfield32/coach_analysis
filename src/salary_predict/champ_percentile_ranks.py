
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder, leaguedashplayerstats
import time
from requests.exceptions import RequestException
from json.decoder import JSONDecodeError
from concurrent.futures import ThreadPoolExecutor, as_completed
from IPython.display import display

MAX_REQUESTS_PER_MINUTE = 30
DELAY_BETWEEN_REQUESTS = 2

RELEVANT_STATS = [
    'PTS', 'AST', 'OREB', 'DREB', 'FG3M', 'FG3_PCT', 'FGM', 'FG_PCT', 'FTM', 'FT_PCT'
]

def fetch_with_retry(endpoint, max_retries=5, delay=5, **kwargs):
    for attempt in range(max_retries):
        try:
            print(f"Fetching data using {endpoint.__name__} (Attempt {attempt + 1}) with parameters: {kwargs}")
            data = endpoint(**kwargs).get_data_frames()
            return data[0] if isinstance(data, list) else data
        except (RequestException, JSONDecodeError, KeyError) as e:
            print(f"Error occurred: {e}")
            if attempt == max_retries - 1:
                print(f"Failed to fetch data after {max_retries} attempts")
                return None
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)

def get_champion(season):
    games = fetch_with_retry(leaguegamefinder.LeagueGameFinder, 
                             season_nullable=season, 
                             season_type_nullable='Playoffs')
    
    if games is None or games.empty:
        print(f"No data found for season {season}")
        return None

    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    last_game = games.sort_values('GAME_DATE').iloc[-2:]
    winner = last_game[last_game['WL'] == 'W'].iloc[0]
    
    return {
        'Season': season,
        'TeamID': winner['TEAM_ID'],
        'TeamName': winner['TEAM_NAME']
    }

def get_champions(start_year, end_year, reload=False):
    if not reload:
        try:
            champions_df = pd.read_csv('../data/processed/nba_champions.csv')
            champions = {row['Season']: {'TeamID': row['ChampionTeamID'], 'TeamName': row['ChampionTeamName']}
                         for _, row in champions_df.iterrows()}
            print("Loaded champions data from file.")
            return champions
        except FileNotFoundError:
            print("Champions data file not found. Fetching new data.")
    
    champions = {}
    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year+1)[-2:]}"
        champion = get_champion(season)
        if champion:
            champions[season] = champion
        time.sleep(DELAY_BETWEEN_REQUESTS)
    
    # Save to CSV
    df_champions = pd.DataFrame(champions.values())
    df_champions.to_csv('../data/processed/nba_champions.csv', index=False)
    print("Champions data saved to 'nba_champions.csv'")
    
    return champions

def get_player_stats(season):
    player_stats = fetch_with_retry(leaguedashplayerstats.LeagueDashPlayerStats, season=season, measure_type_detailed_defense='Base')
    if player_stats is not None:
        return player_stats
    return None

def calculate_percentiles(stats, min_minutes_per_game=10, min_games=41):
    qualified_players = stats[(stats['MIN'] / stats['GP'] >= min_minutes_per_game) & (stats['GP'] >= min_games)].copy()
    
    for column in RELEVANT_STATS:
        if column in qualified_players.columns and qualified_players[column].dtype in [np.float64, np.int64]:
            qualified_players.loc[:, f'{column}_PERCENTILE'] = qualified_players[column].rank(pct=True) * 100
    return qualified_players

def analyze_team_percentiles(champions, all_player_stats, min_minutes_per_game=10, min_games=41):
    team_percentiles = {stat: [] for stat in RELEVANT_STATS}
    summary_stats = {}

    for season, champion_info in champions.items():
        if season in all_player_stats:
            season_stats = all_player_stats[season]
            champion_players = season_stats[season_stats['TEAM_ID'] == champion_info['TeamID']].copy()
            
            champion_players = champion_players[(champion_players['MIN'] / champion_players['GP'] >= min_minutes_per_game) & (champion_players['GP'] >= min_games)]
            
            season_summary = {'TeamName': champion_info['TeamName']}
            for stat in RELEVANT_STATS:
                percentile_column = f'{stat}_PERCENTILE'
                if percentile_column in champion_players.columns:
                    percentiles = champion_players[percentile_column].values
                    if percentiles.size > 0:
                        team_percentiles[stat].extend(percentiles)
                        
                        season_summary[stat] = {
                            'min': np.min(percentiles),
                            'max': np.max(percentiles),
                            'mean': np.mean(percentiles),
                            'std': np.std(percentiles),
                            'above_average': np.sum(percentiles > 50),
                            'total_players': len(percentiles)
                        }
                    else:
                        season_summary[stat] = {
                            'min': None,
                            'max': None,
                            'mean': None,
                            'std': None,
                            'above_average': 0,
                            'total_players': 0
                        }
            
            summary_stats[season] = season_summary

    overall_summary = {stat: {
        'min': np.min(team_percentiles[stat]) if len(team_percentiles[stat]) > 0 else None,
        'max': np.max(team_percentiles[stat]) if len(team_percentiles[stat]) > 0 else None,
        'mean': np.mean(team_percentiles[stat]) if len(team_percentiles[stat]) > 0 else None,
        'std': np.std(team_percentiles[stat]) if len(team_percentiles[stat]) > 0 else None,
        'above_average': np.sum(np.array(team_percentiles[stat]) > 50),
        'total_players': len(team_percentiles[stat])
    } for stat in RELEVANT_STATS}

    return summary_stats, overall_summary

def fetch_season_data(year):
    season = f"{year}-{str(year+1)[-2:]}"
    print(f"Processing {season}...")
    player_stats = get_player_stats(season)
    if player_stats is not None:
        return season, calculate_percentiles(player_stats)
    return season, None

def main_championship_analysis(start_year, end_year, min_minutes_per_game=10, min_games=41, reload_champions=False):
    champions = get_champions(start_year, end_year, reload=reload_champions)
    
    print("Champions by season:")
    for season, info in champions.items():
        print(f"{season}: {info['TeamName']}")

    all_player_stats = {}
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_season = {executor.submit(fetch_season_data, year): year for year in range(start_year, end_year + 1)}
        for future in as_completed(future_to_season):
            season, percentile_player_stats = future.result()
            if percentile_player_stats is not None:
                all_player_stats[season] = percentile_player_stats
    
    summary_stats, overall_summary = analyze_team_percentiles(champions, all_player_stats, min_minutes_per_game, min_games)
    
    return summary_stats, overall_summary, all_player_stats, champions

def print_team_players(season, champions, all_player_stats, min_minutes_per_game=10, min_games=41):
    champion_info = champions.get(season)
    if champion_info and season in all_player_stats:
        season_stats = all_player_stats[season]
        champion_players = season_stats[season_stats['TEAM_ID'] == champion_info['TeamID']]
        filtered_players = champion_players[(champion_players['MIN'] / champion_players['GP'] >= min_minutes_per_game) & (champion_players['GP'] >= min_games)]
        
        print(f"\nChampionship team players for {season} ({champion_info['TeamName']}):")
        display(filtered_players[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'GP', 'MIN']])
        print(f"Total filtered players: {len(filtered_players)}")
    else:
        print(f"No data available for {season}")

def calculate_team_percentiles(team_players, all_players):
    team_percentiles = {}
    for stat in RELEVANT_STATS:
        percentile_column = f'{stat}_PERCENTILE'
        if percentile_column in team_players.columns:
            percentiles = team_players[percentile_column].values
            team_percentiles[stat] = {
                'min': np.min(percentiles),
                'max': np.max(percentiles),
                'mean': np.mean(percentiles),
                'std': np.std(percentiles),
                'above_average': np.sum(percentiles > 50),
                'total_players': len(percentiles)
            }
    return team_percentiles

def simulate_trade(team_players, new_player_stats):
    # Remove a player (e.g., the lowest-ranked player) to make room for the new player
    team_players = team_players.sort_values('PTS', ascending=True).iloc[1:]
    
    # Add the new player to the team
    new_team = pd.concat([team_players, new_player_stats], ignore_index=True)
    
    return new_team

def compare_percentiles(current_percentiles, simulated_percentiles, champ_percentiles):
    comparison = {}
    for stat in RELEVANT_STATS:
        comparison[stat] = {
            'Current': current_percentiles[stat]['mean'],
            'With New Player': simulated_percentiles[stat]['mean'],
            'Champ Average': champ_percentiles[stat]['mean'],
            'Current Diff': current_percentiles[stat]['mean'] - champ_percentiles[stat]['mean'],
            'Simulated Diff': simulated_percentiles[stat]['mean'] - champ_percentiles[stat]['mean']
        }
    return comparison

def analyze_trade_impact(team_abbr, new_player_name, season, min_minutes_per_game=10, min_games=20):
    # Fetch player stats
    all_player_stats = get_player_stats(season)
    if all_player_stats is None:
        print("Failed to fetch player stats.")
        return

    # Calculate percentiles
    percentile_stats = calculate_percentiles(all_player_stats, min_minutes_per_game, min_games)

    # Get team players
    team_players = percentile_stats[percentile_stats['TEAM_ABBREVIATION'] == team_abbr]

    # Get new player's stats
    new_player_stats = percentile_stats[percentile_stats['PLAYER_NAME'] == new_player_name]

    if new_player_stats.empty:
        print(f"Could not find {new_player_name}'s stats.")
        return

    # Calculate current team percentiles
    current_team_percentiles = calculate_team_percentiles(team_players, percentile_stats)

    # Simulate trade
    team_with_new_player = simulate_trade(team_players, new_player_stats)

    # Calculate simulated team percentiles
    simulated_team_percentiles = calculate_team_percentiles(team_with_new_player, percentile_stats)

    # Get championship team percentiles
    champions = get_champions(2014, 2023)
    _, champ_percentiles = analyze_team_percentiles(champions, {season: percentile_stats for season in champions})

    # Compare percentiles
    comparison = compare_percentiles(current_team_percentiles, simulated_team_percentiles, champ_percentiles)

    # Print results
    print(f"Comparison of {team_abbr} Percentiles with and without {new_player_name}:")
    print("{:<10} {:<15} {:<15} {:<20} {:<15} {:<15}".format(
        "Stat", "Current", f"With {new_player_name}", "Champ Average", "Current Diff", "Simulated Diff"))
    print("-" * 90)
    for stat, values in comparison.items():
        print("{:<10} {:<15.2f} {:<15.2f} {:<20.2f} {:<15.2f} {:<15.2f}".format(
            stat, 
            values['Current'], 
            values['With New Player'], 
            values['Champ Average'],
            values['Current Diff'],
            values['Simulated Diff']
        ))

    return comparison

if __name__ == "__main__":
    # Example usage
    start_year = 2014
    end_year = 2023
    min_minutes_per_game = 10
    min_games = 41
    reload_champions = False
    
    summary_stats, overall_summary, all_player_stats, champions = main_championship_analysis(
        start_year, end_year, min_minutes_per_game, min_games, reload_champions
    )
    
    # Print players for the 2022-23 champion
    print_team_players("2022-23", champions, all_player_stats, min_minutes_per_game, min_games)
    
    # Analyze trade impact
    analyze_trade_impact('LAL', 'Stephen Curry', '2023-24')
