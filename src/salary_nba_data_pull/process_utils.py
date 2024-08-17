import pandas as pd
import numpy as np
from datetime import datetime
import cpi
from fetch_utils import fetch_player_info, fetch_career_stats, fetch_league_standings
from scrape_utils import scrape_advanced_metrics

def inflate_value(value, year_str, debug=False):
    try:
        year = int(year_str[:4])
        current_year = datetime.now().year
       
        if year >= current_year:
            return value  # Return the original value for future years
        # Adjust to 2022 dollars to match the original data
        inflated_value = cpi.inflate(value, year, to=2022)
        if debug:
            print(f"Inflated value {value} from {year} to {inflated_value} (2022 dollars)")
        return inflated_value
    except ValueError:
        if debug:
            print(f"Invalid year format: {year_str}")
        return value
    except cpi.errors.CPIObjectDoesNotExist:
        # If data for the specific year is not available, use the earliest available year
        earliest_year = min(cpi.SURVEYS['CPI-U'].indexes['annual'].keys()).year
        inflated_value = cpi.inflate(value, earliest_year, to=2022)
        if debug:
            print(f"Used earliest available year {earliest_year} for inflation calculation")
        return inflated_value
    except Exception as e:
        if debug:
            print(f"Error inflating value for year {year_str}: {e}")
        return value

def calculate_percentages(df, debug=False):
    df['FG%'] = df['FG'] / df['FGA'].replace(0, np.nan)
    df['3P%'] = df['3P'] / df['3PA'].replace(0, np.nan)
    df['2P%'] = df['2P'] / df['2PA'].replace(0, np.nan)
    df['FT%'] = df['FT'] / df['FTA'].replace(0, np.nan)
    df['eFG%'] = (df['FG'] + 0.5 * df['3P']) / df['FGA'].replace(0, np.nan)
    if debug:
        print("Calculated percentages:")
        print(df[['FG%', '3P%', '2P%', 'FT%', 'eFG%']].head())
    return df

def process_player_data(player, season, all_players, debug=False):
    player_lower = player.lower()
    if player_lower not in all_players:
        if debug:
            print(f"No player ID found for {player} in all_players. Player might be missing or the name format might differ.")
        # Print the first few keys from all_players to check name formatting
        if debug:
            print(f"First few player names in all_players: {list(all_players.keys())[:5]}")
        return None

    player_id = all_players[player_lower]['player_id']
    team_id = all_players[player_lower]['team_id']

    if debug:
        print(f"Processing data for player: {player} (ID: {player_id}, Team ID: {team_id})")

    player_info = fetch_player_info(player_id, debug=debug)
    career_stats = fetch_career_stats(player_id, debug=debug)
    league_standings = fetch_league_standings(season, debug=debug)

    # Scrape advanced metrics from Basketball Reference
    advanced_metrics = scrape_advanced_metrics(player, season, debug=debug)

    if player_info is None or career_stats is None or career_stats.empty:
        if debug:
            print(f"Unable to fetch complete data for {player}")
        return None

    season_stats = career_stats[career_stats['SEASON_ID'].str.contains(season.split('-')[0], na=False)]
    if season_stats.empty:
        if debug:
            print(f"No stats found for {player} in season {season}")
        return None

    latest_season_stats = season_stats.iloc[0]
    
    try:
        draft_year = int(player_info['DRAFT_YEAR'].iloc[0])
    except ValueError:
        draft_year = int(player_info['FROM_YEAR'].iloc[0])

    current_season_year = int(season.split('-')[0])
    years_of_service = max(0, current_season_year - draft_year)

    # Handle missing league standings gracefully
    if league_standings is not None and not league_standings.empty:
        player_stats = calculate_player_stats(latest_season_stats, player_info, years_of_service, team_id, league_standings, advanced_metrics)
    else:
        player_stats = calculate_player_stats(latest_season_stats, player_info, years_of_service, team_id, pd.DataFrame(), advanced_metrics)

    player_stats.update({'Player': player, 'Season': season})

    if debug:
        print(f"Processed data for {player} in season {season}: {player_stats}")
    return player_stats


def calculate_player_stats(stats, player_info, years_of_service, team_id, league_standings, advanced_metrics):
    fg = stats.get('FGM', 0) or 0
    fga = stats.get('FGA', 0) or 0
    fg3 = stats.get('FG3M', 0) or 0
    fg3a = stats.get('FG3A', 0) or 0
    efg = (fg + 0.5 * fg3) / fga if fga != 0 else 0
    fg2 = fg - fg3
    fg2a = fga - fg3a
    fg2_pct = (fg2 / fg2a) if fg2a != 0 else 0

    player_stats = {
        'Position': player_info.iloc[0]['POSITION'],
        'Age': stats.get('PLAYER_AGE', None),
        'Team': stats.get('TEAM_ABBREVIATION', None),
        'TeamID': team_id,
        'Years of Service': years_of_service,
        'GP': stats.get('GP', None),
        'GS': stats.get('GS', None),
        'MP': stats.get('MIN', None),
        'FG': fg,
        'FGA': fga,
        'FG%': stats.get('FG_PCT', None),
        '3P': fg3,
        '3PA': fg3a,
        '3P%': stats.get('FG3_PCT', None),
        '2P': fg2,
        '2PA': fg2a,
        '2P%': fg2_pct,
        'eFG%': efg,
        'FT': stats.get('FTM', None),
        'FTA': stats.get('FTA', None),
        'FT%': stats.get('FT_PCT', None),
        'ORB': stats.get('OREB', None),
        'DRB': stats.get('DREB', None),
        'TRB': stats.get('REB', None),
        'AST': stats.get('AST', None),
        'STL': stats.get('STL', None),
        'BLK': stats.get('BLK', None),
        'TOV': stats.get('TOV', None),
        'PF': stats.get('PF', None),
        'PTS': stats.get('PTS', None),
    }
    
    # Add advanced metrics
    player_stats.update(advanced_metrics)

    if league_standings is not None and not league_standings.empty:
        team_standings = league_standings[league_standings['TeamID'] == team_id]
        if not team_standings.empty:
            player_stats.update({
                'Wins': team_standings['WINS'].values[0],
                'Losses': team_standings['LOSSES'].values[0]
            })

    return player_stats

if __name__ == "__main__":
    # Example usage
    debug = True
    season = "2022-23"
    sample_value = 1000000
    sample_year = "2022"
    sample_player = "LeBron James"
    
    # Test inflate_value
    inflated_value = inflate_value(sample_value, sample_year, debug=debug)
    print(f"Inflated value: {inflated_value}")
    
    # Test calculate_percentages
    sample_df = pd.DataFrame({
        'FG': [100], 'FGA': [200],
        '3P': [50], '3PA': [100],
        '2P': [50], '2PA': [100],
        'FT': [75], 'FTA': [100]
    })
    calculated_df = calculate_percentages(sample_df, debug=debug)
    print("Calculated percentages:")
    print(calculated_df)
    
    # Test process_player_data
    # Note: This requires actual data from fetch_utils and scrape_utils
    # Here's a mock-up of how it would work:
    # from fetch_utils import fetch_all_players
    all_players = fetch_all_players(season, debug=debug)
    if sample_player.lower() in all_players:
        player_data = process_player_data(sample_player, season, all_players, debug=debug)
        print(f"Processed data for {sample_player}:")
        print(player_data)
    else:
        print(f"Player {sample_player} not found in the {season} season data.")
