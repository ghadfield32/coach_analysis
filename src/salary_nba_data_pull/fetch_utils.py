import threading
import time
import random
import logging
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps
from http import HTTPStatus
from typing import Callable
import requests
from nba_api.stats.endpoints import commonallplayers, commonplayerinfo, playercareerstats, leaguestandings, commonteamroster
from requests.exceptions import RequestException
from json.decoder import JSONDecodeError
from joblib import Memory
from unidecode import unidecode
from tenacity import (
    retry, retry_if_exception, wait_random_exponential,
    stop_after_attempt, before_log
)

# --- NEW: Team game logs endpoint detection ---
try:
    # newer nba_api
    from nba_api.stats.endpoints import teamgamelogs as _teamgamelogs_mod
    _HAVE_TEAMGAMELOGS_PLURAL = True
except Exception:
    _HAVE_TEAMGAMELOGS_PLURAL = False
try:
    # older nba_api
    from nba_api.stats.endpoints import teamgamelog as _teamgamelog_mod
    _HAVE_TEAMGAMELOG_SINGULAR = True
except Exception:
    _HAVE_TEAMGAMELOG_SINGULAR = False

REQUESTS_PER_MIN = 8   # ↓ a bit safer for long pulls (NBA suggests ≤10)
_SEM = threading.BoundedSemaphore(REQUESTS_PER_MIN)

# --- cache dir with safe fallback for cloud environments ---
import tempfile
from pathlib import Path

def _pick_cache_dir(preferred: str) -> str:
    """
    Choose a writable cache directory. Fall back to /tmp if needed.
    Prints diagnostics; does not mask errors.
    """
    try_dir = Path(__file__).resolve().parent / preferred
    try:
        try_dir.mkdir(parents=True, exist_ok=True)
        test = try_dir / ".write_test"
        with open(test, "w") as f:
            f.write("ok")
        test.unlink(missing_ok=True)
        print(f"[nba_api cache] using {try_dir}")
        return str(try_dir)
    except Exception as e:
        tmp = Path(tempfile.gettempdir()) / "nba_api_cache"
        tmp.mkdir(parents=True, exist_ok=True)
        print(f"[nba_api cache] WARNING: {try_dir} not writable → falling back to {tmp} ({e})")
        return str(tmp)

cache_dir = _pick_cache_dir("../../data/cache/nba_api")
memory = Memory(cache_dir, verbose=0)

# --- quick environment/network diagnostics ---
def network_env_diagnostics(*, timeout_sec: int = 5) -> dict:
    """
    Smoke test for outbound connectivity and stats.nba.com reachability.
    No filling; returns a dict of raw results.
    """
    out = {"google_204": None, "nba_stats": None, "errors": []}
    try:
        r = requests.get("https://www.google.com/generate_204", timeout=timeout_sec)
        out["google_204"] = getattr(r, "status_code", None)
    except Exception as e:
        out["errors"].append(f"google_204: {type(e).__name__}: {e}")

    try:
        hdr = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://stats.nba.com/",
            "Origin": "https://stats.nba.com",
        }
        r = requests.get("https://stats.nba.com", headers=hdr, timeout=timeout_sec)
        out["nba_stats"] = getattr(r, "status_code", None)
    except Exception as e:
        out["errors"].append(f"stats_nba: {type(e).__name__}: {e}")

    print(f"[net diag] {out}")
    return out


def get_players_for_season_fast(season: str,
                                *,
                                debug: bool = True) -> pd.DataFrame:
    """
    UI helper: return a tiny DataFrame [Player, PlayerID, Team, TeamID] for the season.
    Priority:
      1) local season index (instant)
      2) one roster call via nba_api (guarded, with diagnostics)
    No gamelogs. No filling.
    """
    from salary_nba_data_pull.data_utils import read_season_player_index
    
    idx = read_season_player_index(season, debug=debug)
    if not idx.empty:
        return idx[["Player","PlayerID","Team","TeamID"]].drop_duplicates().reset_index(drop=True)

    # Fallback: make exactly ONE roster call (lightweight) with diagnostics
    diag = network_env_diagnostics(timeout_sec=5)
    if diag.get("nba_stats") not in (200, 301, 302):
        # Cloud likely blocked or unreachable; fail fast with a clear message
        print(f"[players-fast] stats.nba.com not reachable (diag={diag}); returning empty result.")
        return pd.DataFrame(columns=["Player","PlayerID","Team","TeamID"])

    # Otherwise, try the cached / rate-limited commonallplayers path
    roster = fetch_season_players(season, debug=debug)
    rows = []
    for key, meta in roster.items():
        rows.append({
            "Player": key.upper(),
            "PlayerID": meta.get("player_id"),
            "Team": None,          # not known without career row; we don't guess
            "TeamID": meta.get("team_id"),
        })
    out = pd.DataFrame(rows)
    print(f"[players-fast] {season}: built {len(out)} players from roster fallback")
    return out

def _throttle():
    """Global semaphore + sleep to stay under REQUESTS_PER_MIN."""
    _SEM.acquire()
    time.sleep(60 / REQUESTS_PER_MIN)
    _SEM.release()

def _needs_retry(exc: Exception) -> bool:
    """Return True if we should retry."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        code = exc.response.status_code
        if code in (HTTPStatus.TOO_MANY_REQUESTS, HTTPStatus.SERVICE_UNAVAILABLE):
            return True
    return isinstance(exc, (requests.ConnectionError, requests.Timeout))

def _respect_retry_after(resp: requests.Response):
    """Sleep for server‑suggested time if header present."""
    if resp is not None and 'Retry-After' in resp.headers:
        try:
            sleep = int(resp.headers['Retry-After'])
            logging.warning("↺ server asked to wait %ss", sleep)
            time.sleep(sleep)
        except ValueError:
            pass   # header unparsable, ignore

def _make_retry(fn: Callable) -> Callable:
    """Decorator to add tenacity retry with jitter + respect Retry-After."""
    @retry(
        retry=retry_if_exception(_needs_retry),
        wait=wait_random_exponential(multiplier=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )
    @wraps(fn)
    def _wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except requests.HTTPError as exc:
            _respect_retry_after(exc.response)
            raise
    return _wrapper

@memory.cache
@_make_retry
def fetch_with_retry(endpoint, *, timeout=90, debug=False, **kwargs):
    """
    Thread‑safe, rate‑limited, cached NBA‑Stats call with adaptive back‑off.
    """
    _throttle()
    start = time.perf_counter()
    resp = endpoint(timeout=timeout, **kwargs)
    df = resp.get_data_frames()[0]
    if debug:
        logging.debug("✓ %s in %.1fs %s", endpoint.__name__,
                      time.perf_counter() - start, kwargs)
    return df

@memory.cache
def fetch_all_players(season: str, debug: bool = False) -> dict[str, dict]:
    """Return {clean_name: {'player_id':…, 'team_id':…}} for *active* roster."""
    roster_df = fetch_with_retry(
        commonallplayers.CommonAllPlayers,
        season=season,
        is_only_current_season=1,        # <‑‑ key fix
        league_id="00",
        debug=debug,
    )
    players: dict[str, dict] = {}
    if roster_df is not None:
        for _, row in roster_df.iterrows():
            clean = unidecode(row["DISPLAY_FIRST_LAST"]).strip().lower()
            players[clean] = {
                "player_id": int(row["PERSON_ID"]),
                "team_id": int(row["TEAM_ID"]),
            }
    if debug:
        print(f"[fetch_all_players] {len(players)} active players for {season}")
    return players

@memory.cache
def fetch_team_roster(team_id: int, season: str, *, timeout: int = 90, debug: bool = False) -> pd.DataFrame:
    """
    Authoritative team roster for a (team_id, season) using CommonTeamRoster.
    Returns columns at least: PLAYER, PERSON_ID (PLAYER_ID in some versions).
    No filling; empty DataFrame if endpoint returns none.
    """
    try:
        resp = commonteamroster.CommonTeamRoster(team_id=team_id, season=season, timeout=timeout)
        dfs = resp.get_data_frames()
        if not dfs:
            if debug:
                print(f"[fetch_team_roster] empty dfs for team_id={team_id}, season={season}")
            return pd.DataFrame(columns=["PLAYER", "PERSON_ID"])
        df = dfs[0].copy()
        if "PLAYER_ID" in df.columns and "PERSON_ID" not in df.columns:
            df = df.rename(columns={"PLAYER_ID": "PERSON_ID"})
        if debug:
            print(f"[fetch_team_roster] team_id={team_id}, season={season}, rows={len(df)} cols={list(df.columns)}")
            print(df.head(8).to_string(index=False))
        return df
    except Exception as e:
        if debug:
            print(f"[fetch_team_roster][ERROR] team_id={team_id}, season={season}: {e}")
        return pd.DataFrame(columns=["PLAYER", "PERSON_ID"])


@lru_cache(maxsize=None)
def fetch_season_players(season: str, debug: bool = False) -> dict[str, dict]:
    """
    Return {clean_name: {'player_id':…, 'team_id':…}} for *everyone who was
    on a roster at any time during the given season*.
    """
    # call once for the whole database (not "current‑season only")
    df = fetch_with_retry(
        commonallplayers.CommonAllPlayers,
        season=season,
        is_only_current_season=0,         # <-- key change
        league_id="00",
        debug=debug,
    )
    players: dict[str, dict] = {}
    if df is not None:
        yr = int(season[:4])
        # keep rows whose career window encloses this season
        df = df[(df.FROM_YEAR.astype(int) <= yr) & (df.TO_YEAR.astype(int) >= yr)]
        for _, row in df.iterrows():
            clean = unidecode(row["DISPLAY_FIRST_LAST"]).strip().lower()
            players[clean] = {
                "player_id": int(row["PERSON_ID"]),
                "team_id": int(row["TEAM_ID"]),
            }

    if debug:
        print(f"[fetch_season_players] {len(players)} players for {season}")
    return players

@memory.cache
def fetch_player_info(player_id, debug=False):
    return fetch_with_retry(commonplayerinfo.CommonPlayerInfo, player_id=player_id, debug=debug)

@memory.cache
def fetch_career_stats(player_id, debug=False):
    return fetch_with_retry(playercareerstats.PlayerCareerStats, player_id=player_id, debug=debug)

@memory.cache
def fetch_league_standings(season, debug=False):
    return fetch_with_retry(leaguestandings.LeagueStandings, season=season, debug=debug)

def clear_cache():
    """Clear the joblib memory cache."""
    memory.clear()

@memory.cache
def fetch_team_wl_by_season(season: str,
                            season_type: str = "Regular Season",
                            debug: bool = False) -> pd.DataFrame:
    """
    Return per‑team W/L for a season from team game logs.
    Robust to nba_api versions:
      - TeamGameLogs(...).get_data_frames()[0]  (new)
      - TeamGameLog(...).get_data_frames()[0]   (old)
    We do not fill; if logs are empty, we return an empty DataFrame.
    """
    import pandas as pd

    if _HAVE_TEAMGAMELOGS_PLURAL:
        # new endpoint signature (nullable arg names in newer APIs)
        df = fetch_with_retry(
            _teamgamelogs_mod.TeamGameLogs,
            season_nullable=season,
            season_type_nullable=season_type,
            debug=debug,
        )
    elif _HAVE_TEAMGAMELOG_SINGULAR:
        # older endpoint
        df = fetch_with_retry(
            _teamgamelog_mod.TeamGameLog,
            season=season,
            season_type_all_star=season_type,
            debug=debug,
        )
    else:
        if debug:
            print("[fetch_team_wl_by_season] no team game log endpoint available")
        return pd.DataFrame(columns=["TeamID", "Wins", "Losses"])

    if df is None or df.empty:
        if debug:
            print(f"[fetch_team_wl_by_season] empty logs for {season}")
        return pd.DataFrame(columns=["TeamID", "Wins", "Losses"])

    # Normalize column names used across versions
    # Expect at least TEAM_ID and WL fields.
    cols = {c.upper(): c for c in df.columns}
    team_id_col = cols.get("TEAM_ID", None)
    wl_col = cols.get("WL", None)

    if team_id_col is None or wl_col is None:
        if debug:
            print(f"[fetch_team_wl_by_season] required columns missing in logs "
                  f"{list(df.columns)}")
        return pd.DataFrame(columns=["TeamID", "Wins", "Losses"])

    # Count W/L by team
    grp = (df.assign(_W=(df[wl_col] == "W").astype(int),
                     _L=(df[wl_col] == "L").astype(int))
             .groupby(df[team_id_col], dropna=False)[["_W", "_L"]]
             .sum()
             .rename(columns={"_W": "Wins", "_L": "Losses"})
             .reset_index()
             .rename(columns={team_id_col: "TeamID"}))

    if debug:
        tot_w = int(grp["Wins"].sum())
        tot_l = int(grp["Losses"].sum())
        print(f"[fetch_team_wl_by_season] {season} totals: W={tot_w}, L={tot_l}")

    return grp

@memory.cache
def fetch_team_wl_lookup(season: str,
                         season_type: str = "Regular Season",
                         debug: bool = False) -> pd.DataFrame:
    """
    Unified W/L by TeamID for a season.
    Primary: team game logs aggregation (robust across nba_api versions).
    Fallback: LeagueStandings endpoint.
    Returns columns: TeamID, Wins, Losses (one row per TeamID).
    """
    import pandas as pd

    # Primary
    logs = fetch_team_wl_by_season(season, season_type=season_type, debug=debug)
    logs = logs.rename(columns={"Wins": "Wins_logs", "Losses": "Losses_logs"})

    # Fallback (LeagueStandings)
    st = fetch_league_standings(season, debug=debug)
    # nba_api LeagueStandings uses uppercase WINS/LOSSES
    want_cols = {}
    for c in st.columns:
        uc = str(c).upper()
        if uc == "TEAM_ID": want_cols[c] = "TeamID"
        if uc == "WINS":    want_cols[c] = "Wins_stand"
        if uc == "LOSSES":  want_cols[c] = "Losses_stand"
    st = st.rename(columns=want_cols)
    st = st[[c for c in ["TeamID", "Wins_stand", "Losses_stand"] if c in st.columns]].drop_duplicates("TeamID")

    # Outer join both sources on TeamID, then coalesce
    out = pd.merge(logs, st, on="TeamID", how="outer", validate="1:1")
    out["Wins"]   = out["Wins_logs"].combine_first(out["Wins_stand"])
    out["Losses"] = out["Losses_logs"].combine_first(out["Losses_stand"])
    out = out[["TeamID", "Wins", "Losses"]].drop_duplicates("TeamID").reset_index(drop=True)

    if debug:
        miss = int(out["Wins"].isna().sum())
        if miss:
            print(f"[fetch_team_wl_lookup] WARN: {miss} TeamID rows still missing Wins/Losses")
    return out

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
