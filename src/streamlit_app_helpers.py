"""
Fast player list accessors for Streamlit app.
Replaces heavy gamelog calls with lightweight season index lookups.
"""

import pandas as pd
from salary_nba_data_pull.data_utils import read_season_player_index
from salary_nba_data_pull.fetch_utils import network_env_diagnostics, fetch_season_players


def get_players_for_season_fast(season: str,
                                *,
                                debug: bool = True) -> pd.DataFrame:
    """
    UI helper: tiny DataFrame [Player, PlayerID, Team, TeamID] for the season.

    Priority:
      1) local season index (instant)
      2) ONE roster call via nba_api (diagnostics only; Team=None by design)

    No filling; returns empty DataFrame if nothing can be fetched.
    """
    from salary_nba_data_pull.data_utils import read_season_player_index
    from salary_nba_data_pull.fetch_utils import network_env_diagnostics, fetch_season_players
    import pandas as pd

    idx = read_season_player_index(season, debug=debug)
    if not idx.empty:
        out = (idx[["Player","PlayerID","Team","TeamID"]]
               .drop_duplicates()
               .reset_index(drop=True))
        if debug:
            print(f"[players-fast] season={season} source=index rows={len(out)}")
            if "Team" in out.columns:
                print(f"[players-fast] Team sample: {out['Team'].dropna().astype(str).unique()[:12]}")
        return out

    diag = network_env_diagnostics(timeout_sec=5)
    if diag.get("nba_stats") not in (200, 301, 302):
        if debug:
            print(f"[players-fast] stats.nba.com not reachable (diag={diag}); returning empty result.")
        return pd.DataFrame(columns=["Player","PlayerID","Team","TeamID"])

    roster = fetch_season_players(season, debug=debug)
    rows = [{"Player": key.upper(),
             "PlayerID": meta.get("player_id"),
             "Team": None,
             "TeamID": meta.get("team_id")} for key, meta in roster.items()]
    out = pd.DataFrame(rows)
    if debug:
        print(f"[players-fast] season={season} source=roster rows={len(out)}; Team is None; TeamID populated variably.")
    return out


def get_season_list_fast(*, debug: bool = True) -> list[str]:
    """
    Get list of available seasons from the season index directory.
    Fast local lookup, no network calls.
    """
    from pathlib import Path
    from salary_nba_data_pull.settings import DATA_PROCESSED_DIR
    
    index_dir = Path(DATA_PROCESSED_DIR) / "season_index"
    if not index_dir.exists():
        if debug:
            print(f"[season-list] index directory not found at {index_dir}")
        return []
    
    seasons = []
    for parquet_file in index_dir.glob("season=*.parquet"):
        season = parquet_file.stem.replace("season=", "")
        seasons.append(season)
    
    seasons.sort(reverse=True)  # newest first
    if debug:
        print(f"[season-list] found {len(seasons)} seasons: {seasons[:5]}...")
    return seasons


def check_network_connectivity(*, debug: bool = True) -> dict:
    """
    Check if the app can reach external services.
    Returns diagnostic information for troubleshooting.
    """
    return network_env_diagnostics(timeout_sec=5)
