
import pandas as pd
from nba_api.stats.endpoints import (
    commonallplayers, commonplayerinfo, teaminfocommon, boxscoresummaryv2,
    boxscoreplayertrackv2, commonteamroster, shotchartdetail, leaguegamefinder,
    boxscorematchupsv3, boxscoreadvancedv3, playbyplayv3, playerdashptshots
)
from nba_api.stats.library.parameters import SeasonAll
import time
from requests.exceptions import RequestException
from json.decoder import JSONDecodeError

def fetch_with_retry(endpoint, max_retries=3, delay=5, **kwargs):
    """
    Fetches data from an endpoint with retry logic.
    """
    for attempt in range(max_retries):
        try:
            data = endpoint(**kwargs).get_data_frames()
            return data
        except (RequestException, JSONDecodeError) as e:
            if attempt == max_retries - 1:
                print(f"Failed to fetch data after {max_retries} attempts: {str(e)}")
                return None
            print(f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...")
            time.sleep(delay)

def fetch_all_teams(season=SeasonAll.default):
    """
    Fetches the list of all teams for a given season.
    """
    return fetch_with_retry(commonallplayers.CommonAllPlayers, season=season, is_only_current_season=1)

def fetch_team_info(team_id):
    """
    Fetches detailed information for a specific team.
    """
    return fetch_with_retry(teaminfocommon.TeamInfoCommon, team_id=team_id)

def fetch_player_info(player_id):
    """
    Fetches detailed information for a specific player.
    """
    return fetch_with_retry(commonplayerinfo.CommonPlayerInfo, player_id=player_id)

def fetch_game_summary(game_id):
    """
    Fetches the summary for a specific game.
    """
    data = fetch_with_retry(boxscoresummaryv2.BoxScoreSummaryV2, game_id=game_id)
    if data:
        return {i: df for i, df in enumerate(data)}
    return None

def fetch_player_game_stats(game_id):
    """
    Fetches the player tracking stats for a specific game.
    """
    return fetch_with_retry(boxscoreplayertrackv2.BoxScorePlayerTrackV2, game_id=game_id)

def fetch_team_roster(team_id, season):
    """
    Fetches the roster and coaches for a specific team.
    """
    return fetch_with_retry(commonteamroster.CommonTeamRoster, team_id=team_id, season=season)

def fetch_shot_chart(team_id, season, season_type="Regular Season"):
    """
    Fetches the shot chart data for a team.
    """
    return fetch_with_retry(shotchartdetail.ShotChartDetail,
                            team_id=team_id,
                            player_id=0,
                            season_nullable=season,
                            season_type_all_star=season_type)

def fetch_team_games(team_id, season, season_type="Regular Season"):
    """
    Fetches all games for a team in a given season.
    """
    return fetch_with_retry(leaguegamefinder.LeagueGameFinder,
                            team_id_nullable=team_id,
                            season_nullable=season,
                            season_type_nullable=season_type)

def fetch_matchups(game_id):
    """
    Fetches matchup data for a specific game.
    """
    return fetch_with_retry(boxscorematchupsv3.BoxScoreMatchupsV3, game_id=game_id)

def fetch_advanced_box_score(game_id):
    """
    Fetches advanced box score data for a specific game.
    """
    return fetch_with_retry(boxscoreadvancedv3.BoxScoreAdvancedV3, game_id=game_id)

def fetch_play_by_play(game_id):
    """
    Fetches play-by-play data for a specific game.
    """
    return fetch_with_retry(playbyplayv3.PlayByPlayV3, game_id=game_id)

def fetch_player_shot_dashboard(player_id, team_id, season, season_type="Regular Season"):
    """
    Fetches detailed shot dashboard data for a specific player.
    """
    return fetch_with_retry(playerdashptshots.PlayerDashPtShots,
                            player_id=player_id,
                            team_id=team_id,
                            season=season,
                            season_type_all_star=season_type)
