
import logging
from nba_api.stats.endpoints import commonallplayers
from requests.exceptions import RequestException
from json.decoder import JSONDecodeError
import time

# Define constants
SEASON = "2022-23"
DEBUG = True
MAX_RETRIES = 3
INITIAL_DELAY = 5

# Function to fetch data with retry logic
def fetch_with_retry(endpoint, max_retries=5, initial_delay=5, debug=False, **kwargs):
    for attempt in range(max_retries):
        try:
            if debug:
                logging.debug(f"Fetching data using {endpoint.__name__} (Attempt {attempt + 1}) with parameters: {kwargs}")
            data = endpoint(**kwargs).get_data_frames()

            if debug:
                logging.debug(f"Raw API Response: {endpoint(**kwargs).get_json()}")

            return data[0] if isinstance(data, list) else data
        except (RequestException, JSONDecodeError, KeyError) as e:
            if debug:
                logging.debug(f"Error occurred: {e}")
            if attempt == max_retries - 1:
                if debug:
                    logging.debug(f"Failed after {max_retries} attempts")
                return None
            time.sleep(initial_delay * (2 ** attempt))

# Function to fetch all players for a season
def fetch_all_players(season, debug=False):
    all_players_data = fetch_with_retry(commonallplayers.CommonAllPlayers, season=season, debug=debug)
    if all_players_data is not None and not all_players_data.empty:
        logging.debug(f"Fetched {len(all_players_data)} players for season {season}")
        print(all_players_data.head())
    else:
        print("No players data found or failed to fetch data.")
    return all_players_data

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
    print(f"Fetching player data for season {SEASON}")
    player_data = fetch_all_players(SEASON, debug=DEBUG)
