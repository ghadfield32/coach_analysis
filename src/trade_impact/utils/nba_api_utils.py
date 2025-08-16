import os
import time
import random
from typing import Optional, Tuple

import pandas as pd

from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import (
    playergamelogs as _playergamelogs,
    commonteamroster as _commonteamroster,
    leaguegamefinder as _leaguegamefinder,
)

# -------------------------
# Paths & simple disk cache
# -------------------------
_CACHE_DIR = os.path.join("..", "data", "processed", "cache_nba_api")
os.makedirs(_CACHE_DIR, exist_ok=True)

def _cache_path(key: str) -> str:
    safe = key.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return os.path.join(_CACHE_DIR, f"{safe}.csv")

def _load_cache_df(key: str) -> Optional[pd.DataFrame]:
    path = _cache_path(key)
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None

def _save_cache_df(key: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    df.to_csv(_cache_path(key), index=False)

# -------------------------
# Season normalization
# -------------------------
def normalize_season(season_or_year: str | int) -> str:
    """
    Accepts '2023-24' or 2023 or '2023' and returns '2023-24'.
    """
    if isinstance(season_or_year, int):
        y = season_or_year
        return f"{y}-{str(y + 1)[-2:]}"
    s = str(season_or_year)
    if "-" in s and len(s) >= 7:
        return s  # already 'YYYY-YY'
    if s.isdigit():
        y = int(s)
        return f"{y}-{str(y + 1)[-2:]}"
    raise ValueError(f"Unrecognized season format: {season_or_year}")

# -------------------------
# Retry helper
# -------------------------
def _with_retries(fn, *, retries=3, base_delay=1.5, jitter=0.75, debug=False):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if debug:
                print(f"[nba_api_utils] attempt {attempt} failed: {e}")
            if attempt < retries:
                delay = base_delay * attempt + random.random() * jitter
                time.sleep(delay)
    raise last_err

# -------------------------
# Team lookup
# -------------------------
def get_team_id_by_full_name(team_full_name: str) -> Optional[int]:
    tlist = static_teams.get_teams()
    for t in tlist:
        if t.get("full_name") == team_full_name:
            return int(t["id"])
    return None

# -------------------------
# Light roster fetch (preferred for populating UI)
# -------------------------
def get_commonteamroster_df(team_id: int, season: str | int,
                            *, timeout=60, retries=3, use_live=True, debug=False) -> pd.DataFrame:
    season_norm = normalize_season(season)
    cache_key = f"commonteamroster_{team_id}_{season_norm}"
    if not use_live:
        cached = _load_cache_df(cache_key)
        if cached is not None:
            return cached

    def _call():
        df = _commonteamroster.CommonTeamRoster(
            team_id=team_id,
            season=season_norm,
            timeout=timeout,
        ).get_data_frames()[0]
        return df

    try:
        df = _with_retries(_call, retries=retries, debug=debug)
        _save_cache_df(cache_key, df)
        return df
    except Exception as e:
        # Fallback to cache if any
        cached = _load_cache_df(cache_key)
        if cached is not None:
            if debug:
                print(f"[nba_api_utils] Using cached roster for {team_id} {season_norm} due to error: {e}")
            return cached
        raise

# -------------------------
# Heavier logs fetch (only when truly needed)
# -------------------------
def get_playergamelogs_df(season: str | int,
                          *, timeout=90, retries=3, use_live=True, debug=False) -> pd.DataFrame:
    season_norm = normalize_season(season)
    cache_key = f"playergamelogs_league_{season_norm}"
    if not use_live:
        cached = _load_cache_df(cache_key)
        if cached is not None:
            return cached

    def _call():
        df = _playergamelogs.PlayerGameLogs(
            season_nullable=season_norm,
            timeout=timeout,
        ).get_data_frames()[0]
        df["SEASON"] = season_norm
        return df

    try:
        df = _with_retries(_call, retries=retries, debug=debug)
        _save_cache_df(cache_key, df)
        return df
    except Exception as e:
        cached = _load_cache_df(cache_key)
        if cached is not None:
            if debug:
                print(f"[nba_api_utils] Using cached logs for {season_norm} due to error: {e}")
            return cached
        raise

# -------------------------
# Champion helper with cache
# -------------------------
def get_champion_team_name(season: str | int, *, timeout=90, retries=3, use_live=True, debug=False) -> Optional[str]:
    """
    Uses playoff games to identify the winner of the final game.
    """
    season_norm = normalize_season(season)
    cache_key = f"champion_team_{season_norm}"

    if not use_live:
        cached = _load_cache_df(cache_key)
        if isinstance(cached, pd.DataFrame) and not cached.empty and "TEAM_NAME" in cached:
            return cached["TEAM_NAME"].iloc[0]

    def _call():
        df = _leaguegamefinder.LeagueGameFinder(
            season_nullable=season_norm,
            season_type_nullable="Playoffs",
            timeout=timeout,
        ).get_data_frames()[0]
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        last_two = df.sort_values("GAME_DATE").iloc[-2:]
        winner_row = last_two[last_two["WL"] == "W"].iloc[0]
        return pd.DataFrame([{"TEAM_NAME": winner_row["TEAM_NAME"]}])

    try:
        winner_df = _with_retries(_call, retries=retries, debug=debug)
        _save_cache_df(cache_key, winner_df)
        return winner_df["TEAM_NAME"].iloc[0]
    except Exception as e:
        cached = _load_cache_df(cache_key)
        if isinstance(cached, pd.DataFrame) and not cached.empty and "TEAM_NAME" in cached:
            if debug:
                print(f"[nba_api_utils] Using cached champion for {season_norm} due to error: {e}")
            return cached["TEAM_NAME"].iloc[0]
        return None
