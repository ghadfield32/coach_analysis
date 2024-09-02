
import pandas as pd
import numpy as np
import time
from nba_api.stats.endpoints import leaguegamefinder, playergamelogs
from nba_api.stats.static import teams, players

# Constants
RELEVANT_STATS = ['PTS', 'AST', 'TOV', 'STL', 'BLK', 'OREB', 'DREB', 'FGM', 'FG3M', 'FGA']
PERCENTILE_THRESHOLDS = [1, 2, 3, 4, 5, 10, 25, 50]

def load_team_data():
    nba_teams = teams.get_teams()
    team_df = pd.DataFrame(nba_teams)
    return team_df[['id', 'full_name', 'abbreviation']]

# Helper Functions

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

def get_champion_for_percentile(season, debug=False):
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

def get_champions_for_percentile(start_year, end_year, debug=False):
    """Fetch champions for each season from start_year to end_year."""
    champions = []
    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year+1)[-2:]}"
        champ_name = get_champion_for_percentile(season, debug)
        if champ_name:
            champions.append({'Season': season, 'ChampionTeamName': champ_name})
        elif debug:
            print(f"Champion data not available for season {season}")
        time.sleep(1)  # To avoid overwhelming the API
    if debug:
        print(f"Champions data: {champions}")
    return pd.DataFrame(champions)

def calculate_average_top_percentiles(top_percentile_counts_df, debug=False):
    """Calculate the average percentiles for all champion teams, grouped by season."""
    average_percentiles = {}

    for col in RELEVANT_STATS:
        for threshold in PERCENTILE_THRESHOLDS:
            count_key = f'{col}_Top_{threshold}_count'
            avg_key = f'{col}_Avg_Top_{threshold}_percentile'
            
            # Calculate the mean of counts grouped by 'Season' and then average these means
            avg_value = top_percentile_counts_df.groupby('Season')[count_key].mean().mean()
            
            avg_value = avg_value if pd.notnull(avg_value) else 0
            average_percentiles[avg_key] = avg_value
            
            if debug:
                print(f"{col} Avg Top {threshold}% Count across seasons: {avg_value}")
    
    return pd.DataFrame([average_percentiles])


def calculate_champion_percentiles(league_percentiles, champions, debug=False):
    """Extract percentiles for players in champion teams based on league percentiles."""
    champion_data = league_percentiles[league_percentiles['TEAM_NAME'].isin(champions['ChampionTeamName'])].copy()
    
    # Merge with champions to get the Season associated with each champion team
    champion_data = pd.merge(champion_data, champions, left_on='TEAM_NAME', right_on='ChampionTeamName')
    
    if debug:
        print("Champion Data Percentiles with Season:")
        print(champion_data[['TEAM_NAME', 'Season', 'PLAYER_NAME']].head())
    
    return champion_data


def fetch_all_player_data(seasons, debug=False):
    """Fetch player game logs data for all players across multiple seasons."""
    all_data = pd.DataFrame()
    for season in seasons:
        try:
            player_logs = playergamelogs.PlayerGameLogs(season_nullable=season).get_data_frames()[0]
            player_logs['SEASON'] = season
            all_data = pd.concat([all_data, player_logs], ignore_index=True)
            if debug:
                print(f"Fetched {len(player_logs)} player logs for the league in season {season}")
        except Exception as e:
            if debug:
                print(f"Error fetching player data for the league in season {season}: {e}")
    if debug:
        print(f"Total logs fetched: {len(all_data)}")
    return all_data

def calculate_player_stats(player_data, debug=False):
    """Calculate average player statistics from game logs."""
    # Calculate stats per game for players
    player_stats = player_data.groupby(['SEASON', 'TEAM_NAME', 'PLAYER_NAME'])[RELEVANT_STATS].mean().reset_index()
    
    # Rename columns to include '_per_game'
    for stat in RELEVANT_STATS:
        player_stats.rename(columns={stat: f'{stat}_per_game'}, inplace=True)

    if debug:
        print("Sample player stats (entire league):")
        print(player_stats.head())  # Show head of the player stats
    return player_stats

def calculate_player_percentiles(stats_df, debug=False):
    """Calculate percentile ranks for each stat in the DataFrame by season."""
    percentiles = {}

    for col in RELEVANT_STATS:
        col_per_game = f'{col}_per_game'
        if col_per_game in stats_df.columns:
            # Calculate percentiles across the entire dataset
            stats_df[f'{col}_percentile'] = stats_df[col_per_game].rank(pct=True, method='min')
            # Ensure no NaN values before calculating percentiles
            if not stats_df[col_per_game].isna().any():
                percentiles[col] = np.percentile(stats_df[col_per_game], [100 - t for t in PERCENTILE_THRESHOLDS])
            else:
                if debug:
                    print(f"NaN values found in {col_per_game} column.")
            if debug:
                print(f"Calculated percentiles for {col_per_game}:")
                print(stats_df[['TEAM_NAME', 'PLAYER_NAME', col_per_game, f'{col}_percentile']].head())
    return stats_df, percentiles

def count_top_percentiles(player_percentiles, percentiles, team_name, season, debug=False):
    """Count how many players in a specific team fall within top percentiles, filtered by season."""
    top_counts = {f'{stat}_Top_{threshold}_count': 0 for stat in RELEVANT_STATS for threshold in PERCENTILE_THRESHOLDS}
    
    # Filter the data by team and season
    team_data = player_percentiles[(player_percentiles['TEAM_NAME'] == team_name) & (player_percentiles['SEASON'] == season)]
    
    if debug:
        print(f"\n{team_name} player percentiles data for season {season}:\n{team_data[['PLAYER_NAME', 'FG3M_per_game', 'FG3M_percentile']]}")
    
    for col in RELEVANT_STATS:
        col_per_game = f'{col}_per_game'
        if col in percentiles:  # Ensure we have valid percentiles calculated
            for idx, threshold in enumerate(PERCENTILE_THRESHOLDS):
                count_key = f'{col}_Top_{threshold}_count'
                top_counts[count_key] = (team_data[col_per_game] >= percentiles[col][idx]).sum()

                if debug and col == 'FG3M':
                    print(f"{col} Top {threshold}% Count for season {season}: {top_counts[count_key]}")
                    print(f"Players in Top {threshold}% for {col} in season {season}: {team_data[team_data[col_per_game] >= percentiles[col][idx]][['PLAYER_NAME', col_per_game, f'{col}_percentile']]}")

    return top_counts


def simulate_trade(player_stats, players_from_team_a, players_from_team_b, team_a_name, team_b_name, debug=False):
    """Simulate a trade by swapping players between two teams."""
    if debug:
        print("\nBefore trade simulation:")
        print(player_stats[(player_stats['PLAYER_NAME'].isin(players_from_team_a + players_from_team_b))][['PLAYER_NAME', 'TEAM_NAME']])
    
    # Swap players between the two teams
    player_stats.loc[player_stats['PLAYER_NAME'].isin(players_from_team_a), 'TEAM_NAME'] = team_b_name
    player_stats.loc[player_stats['PLAYER_NAME'].isin(players_from_team_b), 'TEAM_NAME'] = team_a_name
    
    if debug:
        print("\nAfter trade simulation:")
        print(player_stats[(player_stats['PLAYER_NAME'].isin(players_from_team_a + players_from_team_b))][['PLAYER_NAME', 'TEAM_NAME']])
    
    return player_stats

def create_comparison_table(before_trade, after_trade, average_percentiles, team_name):
    """Create a comparison table for a team before and after the trade."""
    data = {'Team': [team_name] * len(PERCENTILE_THRESHOLDS), 'Percentile': PERCENTILE_THRESHOLDS}
    
    for stat in RELEVANT_STATS:
        before_counts = [before_trade[f'{stat}_Top_{threshold}_count'] for threshold in PERCENTILE_THRESHOLDS]
        after_counts = [after_trade[f'{stat}_Top_{threshold}_count'] for threshold in PERCENTILE_THRESHOLDS]
        champ_avg = [average_percentiles[f'{stat}_Avg_Top_{threshold}_percentile'][0] for threshold in PERCENTILE_THRESHOLDS]
        
        data[f'{stat}_Before'] = before_counts
        data[f'{stat}_After'] = after_counts
        data[f'{stat}_Champ_Avg'] = champ_avg
    
    df = pd.DataFrame(data)
    df.set_index('Percentile', inplace=True)
    return df

def fetch_and_process_season_data(seasons, debug=False):
    # Fetch player data for all specified seasons
    all_player_data = fetch_all_player_data(seasons, debug)
    
    # Calculate player-level stats
    player_stats = calculate_player_stats(all_player_data, debug)
    
    # Calculate percentiles for all players in the league
    league_percentiles, league_percentiles_ref = calculate_player_percentiles(player_stats, debug)
    
    return player_stats, league_percentiles, league_percentiles_ref

def get_champion_percentiles(seasons, debug=False):
    start_year = int(seasons[0].split('-')[0])
    end_year = int(seasons[-1].split('-')[0])

    champion_info = get_champions_for_percentile(start_year, end_year, debug)
    player_stats, league_percentiles, league_percentiles_ref = fetch_and_process_season_data(seasons, debug)
    
    # Calculate champion percentiles including the Season column
    champion_percentiles = calculate_champion_percentiles(league_percentiles, champion_info, debug)
    
    # Group by TEAM_NAME and Season, then calculate top percentiles
    top_percentile_counts = champion_percentiles.groupby(['TEAM_NAME', 'Season']).apply(
        lambda x: count_top_percentiles(x, league_percentiles_ref, x.iloc[0]['TEAM_NAME'], x.iloc[0]['Season'], debug)
    ).apply(pd.Series).reset_index()

    # Calculate average percentiles across all seasons for each champion team
    average_top_percentiles_df = calculate_average_top_percentiles(top_percentile_counts, debug)
    
    return average_top_percentiles_df


def compare_teams_before_after_trade(season, team_a_name, team_b_name, players_from_team_a, players_from_team_b, debug=False):
    player_stats, league_percentiles, league_percentiles_ref = fetch_and_process_season_data([season], debug)
    
    # Count top percentiles before the trade
    team_a_top_percentile_counts = count_top_percentiles(league_percentiles, league_percentiles_ref, team_a_name, season, debug)
    team_b_top_percentile_counts = count_top_percentiles(league_percentiles, league_percentiles_ref, team_b_name, season, debug)
    
    # Simulate the trade
    player_stats = simulate_trade(player_stats, players_from_team_a, players_from_team_b, team_a_name, team_b_name, debug)
    
    # Recalculate percentiles after the trade
    league_percentiles_after_trade, _ = calculate_player_percentiles(player_stats, debug)
    
    if debug:
        print("\nAfter trade percentiles calculation:")
        print(league_percentiles_after_trade[['TEAM_NAME', 'PLAYER_NAME', 'FG3M_per_game', 'FG3M_percentile']])
    
    # Count top percentiles after the trade
    team_a_top_percentile_counts_after = count_top_percentiles(league_percentiles_after_trade, league_percentiles_ref, team_a_name, season, debug)
    team_b_top_percentile_counts_after = count_top_percentiles(league_percentiles_after_trade, league_percentiles_ref, team_b_name, season, debug)
    
    return team_a_top_percentile_counts, team_a_top_percentile_counts_after, team_b_top_percentile_counts, team_b_top_percentile_counts_after


def generate_comparison_tables(season, team_a_name, team_b_name, players_from_team_a, players_from_team_b, average_top_percentiles_df, debug=False):
    team_a_top_before, team_a_top_after, team_b_top_before, team_b_top_after = compare_teams_before_after_trade(
        season, team_a_name, team_b_name, players_from_team_a, players_from_team_b, debug
    )
    
    # Create comparison tables with champion average percentiles
    celtics_comparison_table = create_comparison_table(team_a_top_before, team_a_top_after, average_top_percentiles_df, team_a_name)
    warriors_comparison_table = create_comparison_table(team_b_top_before, team_b_top_after, average_top_percentiles_df, team_b_name)
    
    return celtics_comparison_table, warriors_comparison_table


def main(debug=False):
    seasons = ["2019-20", "2020-21","2021-22", "2022-23", "2023-24"]

    # Fetch champion percentiles and calculate averages
    average_top_percentiles_df = get_champion_percentiles(seasons, debug)
    
    if debug:
        print("\nAverage Champion Percentiles:")
        print(average_top_percentiles_df)
    
    team_a_name = "Boston Celtics"
    team_b_name = "Atlanta Hawks"
    team_a_players = ["Jaylen Brown"]
    team_b_players = ["Trae Young"]
    
    # Generate comparison tables before and after the trade
    celtics_comparison_table, warriors_comparison_table = generate_comparison_tables(
        seasons[-1], team_a_name, team_b_name, team_a_players, team_b_players, average_top_percentiles_df, debug
    )
    
    # Display tables
    print("\nTeam A Comparison Table:")
    print(celtics_comparison_table)
    
    print("\nTeam B Comparison Table:")
    print(warriors_comparison_table)

if __name__ == "__main__":
    main(debug=True)


