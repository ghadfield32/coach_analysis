import time
from nba_api.stats.endpoints import commonallplayers, commonplayerinfo, playercareerstats, leaguestandings
from requests.exceptions import RequestException
from json.decoder import JSONDecodeError

# Define the maximum requests allowed per minute and delay between requests
MAX_REQUESTS_PER_MINUTE = 20
DELAY_BETWEEN_REQUESTS = 3  # seconds

def fetch_with_retry(endpoint, max_retries=10, initial_delay=5, max_delay=60, timeout=60, debug=False, **kwargs):
    for attempt in range(max_retries):
        try:
            if debug:
                print(f"Fetching data using {endpoint.__name__} (Attempt {attempt + 1}) with parameters: {kwargs}")
            data = endpoint(**kwargs, timeout=timeout).get_data_frames()
            time.sleep(DELAY_BETWEEN_REQUESTS)  # Add delay between requests
            return data[0] if isinstance(data, list) else data
        except (RequestException, JSONDecodeError, KeyError) as e:
            if debug:
                print(f"Error occurred during fetching {endpoint.__name__}: {e}")
            if attempt == max_retries - 1:
                if debug:
                    print(f"Failed to fetch data from {endpoint.__name__} after {max_retries} attempts")
                return None
            delay = min(initial_delay * (2 ** attempt), max_delay)
            if debug:
                print(f"Retrying in {delay} seconds...")
            time.sleep(delay)

def fetch_all_players(season, debug=False):
    all_players_data = fetch_with_retry(commonallplayers.CommonAllPlayers, season=season, debug=debug)
    all_players = {}
    if all_players_data is not None:
        for _, row in all_players_data.iterrows():
            player_name = row['DISPLAY_FIRST_LAST'].strip().lower()
            player_id = row['PERSON_ID']
            team_id = row['TEAM_ID']
            all_players[player_name] = {
                'player_id': player_id,
                'team_id': team_id
            }
    if debug:
        print(f"Fetched {len(all_players)} players for season {season}")
    return all_players

def fetch_player_info(player_id, debug=False):
    return fetch_with_retry(commonplayerinfo.CommonPlayerInfo, player_id=player_id, debug=debug)

def fetch_career_stats(player_id, debug=False):
    return fetch_with_retry(playercareerstats.PlayerCareerStats, player_id=player_id, debug=debug)

def fetch_league_standings(season, debug=False):
    return fetch_with_retry(leaguestandings.LeagueStandings, season=season, debug=debug)

if __name__ == "__main__":
    # Example usage
    debug = True
    season = "2022-23"
    sample_player_name = "LeBron James"
    
    # Fetch all players
    all_players = fetch_all_players(season, debug=debug)
    print(f"Total players fetched: {len(all_players)}")
    
    # Fetch player info for a sample player
    if sample_player_name.lower() in all_players:
        sample_player_id = all_players[sample_player_name.lower()]['player_id']
        player_info = fetch_player_info(sample_player_id, debug=debug)
        print(f"Sample player info for {sample_player_name}:")
        print(player_info)
        
        # Fetch career stats for the sample player
        career_stats = fetch_career_stats(sample_player_id, debug=debug)
        print(f"Sample player career stats for {sample_player_name}:")
        print(career_stats)
    else:
        print(f"Player {sample_player_name} not found in the {season} season data.")
    
    # Fetch league standings
    standings = fetch_league_standings(season, debug=debug)
    print("League standings:")
    print(standings)
