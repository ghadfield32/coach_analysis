import pandas as pd
import numpy as np
import logging
import sqlite3
from datetime import datetime
from functools import lru_cache
from salary_nba_data_pull.fetch_utils import fetch_all_players, fetch_career_stats, fetch_player_info, fetch_league_standings
from salary_nba_data_pull.scrape_utils import scrape_advanced_metrics

# --- CPI lazy‑loader --------------------------------------------------
_CPI_AVAILABLE = False  # toggled at runtime

@lru_cache(maxsize=1)
def _ensure_cpi_ready(debug: bool = False) -> bool:
    """
    Import `cpi` lazily and guarantee its internal SQLite DB is usable.
    Returns True when inflation data are available, False otherwise.
    """
    global _CPI_AVAILABLE
    try:
        import importlib
        cpi = importlib.import_module("cpi")        # late import
        try:
            _ = cpi.models.Series.get_by_id("0000")  # 1‑row sanity query
            _CPI_AVAILABLE = True
            return True
        except sqlite3.OperationalError:
            if debug:
                logging.warning("[CPI] DB invalid – rebuilding from BLS…")
            cpi.update(rebuild=True)                # expensive network call
            _CPI_AVAILABLE = True
            return True
    except ModuleNotFoundError:
        if debug:
            logging.warning("[CPI] package not installed")
    except Exception as e:
        if debug:
            logging.error("[CPI] unexpected CPI failure: %s", e)
    return False
# ---------------------------------------------------------------------

def inflate_value(value: float, year_str: str,
                  *, debug: bool = False, skip_inflation: bool = False) -> float:
    """
    Inflate `value` from the dollars of `year_str` (YYYY or YYYY‑YY) to 2022 USD.
    If CPI data are unavailable or the user opts out, return the original value.
    """
    if skip_inflation or not _ensure_cpi_ready(debug):
        return value
    try:
        import cpi                                       # safe: DB ready
        year = int(year_str[:4])
        if year >= datetime.now().year:
            return value
        return float(cpi.inflate(value, year, to=2022))
    except Exception as e:
        if debug:
            logging.error("[CPI] inflate failed for %s: %s", year_str, e)
        return value
# ---------------------------------------------------------------------

def calculate_percentages(df, debug=False):
    """
    Calculate shooting percentages and other derived statistics.
    """
    if df.empty:
        return df

    # Calculate shooting percentages
    if 'FGA' in df.columns and 'FG' in df.columns:
        df['FG%'] = (df['FG'] / df['FGA'] * 100).round(2)
        df['FG%'] = df['FG%'].replace([np.inf, -np.inf], np.nan)

    if '3PA' in df.columns and '3P' in df.columns:
        df['3P%'] = (df['3P'] / df['3PA'] * 100).round(2)
        df['3P%'] = df['3P%'].replace([np.inf, -np.inf], np.nan)

    if 'FTA' in df.columns and 'FT' in df.columns:
        df['FT%'] = (df['FT'] / df['FTA'] * 100).round(2)
        df['FT%'] = df['FT%'].replace([np.inf, -np.inf], np.nan)

    # Calculate efficiency metrics
    if 'PTS' in df.columns and 'FGA' in df.columns and 'FTA' in df.columns:
        df['TS%'] = (df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA'])) * 100).round(2)
        df['TS%'] = df['TS%'].replace([np.inf, -np.inf], np.nan)

    if 'PTS' in df.columns and 'MP' in df.columns:
        df['PTS_per_36'] = (df['PTS'] / df['MP'] * 36).round(2)
        df['PTS_per_36'] = df['PTS_per_36'].replace([np.inf, -np.inf], np.nan)

    if 'AST' in df.columns and 'MP' in df.columns:
        df['AST_per_36'] = (df['AST'] / df['MP'] * 36).round(2)
        df['AST_per_36'] = df['AST_per_36'].replace([np.inf, -np.inf], np.nan)

    if 'TRB' in df.columns and 'MP' in df.columns:
        df['TRB_per_36'] = (df['TRB'] / df['MP'] * 36).round(2)
        df['TRB_per_36'] = df['TRB_per_36'].replace([np.inf, -np.inf], np.nan)

    if debug:
        print("Percentage calculations completed")

    return df

def process_player_data(player_name: str, season: str,
                        all_players: dict[str, dict], *,
                        debug: bool = False) -> dict | None:
    """
    Build a single‑player dict **including Games Started (GS)** and keep the
    schema aligned with dataset 1.
    """
    meta = all_players.get(player_name.lower().strip())
    if not meta:
        return None

    pid = meta["player_id"]
    info_df   = fetch_player_info(pid, debug=debug)
    career_df = fetch_career_stats(pid, debug=debug)
    if career_df is None or career_df.empty:
        return None

    season_row = career_df.loc[career_df.SEASON_ID.eq(season)]
    if season_row.empty:
        return None
    season_row = season_row.iloc[0]

    data = {
        # ---------- BASIC ------------
        "Player": player_name,
        "Season": season,
        "Team":   season_row["TEAM_ABBREVIATION"],
        "Age":    season_row["PLAYER_AGE"],
        "GP":     season_row["GP"],
        "GS":     season_row.get("GS", 0),        # <-- NEW
        "MP":     season_row["MIN"],
        # ---------- SCORING ----------
        "PTS": season_row["PTS"],
        "FG":  season_row["FGM"],  "FGA": season_row["FGA"],
        "3P":  season_row["FG3M"], "3PA": season_row["FG3A"],
        "FT":  season_row["FTM"],  "FTA": season_row["FTA"],
        # ---------- OTHER ------------
        "TRB": season_row["REB"], "AST": season_row["AST"],
        "STL": season_row["STL"], "BLK": season_row["BLK"],
        "TOV": season_row["TOV"], "PF":  season_row["PF"],
        # NEW  ↩ rename OREB/DREB so downstream sees ORB/DRB
        "ORB": season_row.get("OREB", np.nan),
        "DRB": season_row.get("DREB", np.nan),
    }

    # roster meta
    if info_df is not None and not info_df.empty:
        ir = info_df.iloc[0]
        data["Position"]          = ir.get("POSITION", "")
        data["TeamID"]            = ir.get("TEAM_ID", None)
        data["Years_of_Service"]  = ir.get("SEASON_EXP", None)
    else:
        data["TeamID"] = meta.get("team_id")

    # ---------- Derived shooting splits ----------
    two_att          = data["FGA"] - data["3PA"]
    data["2P"]       = data["FG"] - data["3P"]
    data["2PA"]      = two_att
    data["eFG%"]     = round((data["FG"] + 0.5 * data["3P"]) / data["FGA"] * 100 ,2) if data["FGA"] else None
    data["2P%"]      = round(data["2P"] / two_att * 100 ,2)                           if two_att else None

    # ---------- Advanced metrics ----------
    try:
        data.update(scrape_advanced_metrics(player_name, season, debug=debug))
    except Exception as exc:
        if debug:
            logging.warning("%s advanced scrape failed: %s", player_name, exc)

    return data

def merge_injury_data(player_data: pd.DataFrame,
                      injury_data: pd.DataFrame | None) -> pd.DataFrame:
    """
    Attach four injury‑related columns. If a player has no injuries, leave the fields as NA
    (pd.NA) instead of empty strings so repeated runs compare equal.
    """
    import pandas as pd

    if player_data.empty:
        return player_data

    out = player_data.copy()

    # Ensure columns exist with NA defaults
    defaults = {
        "Injured": False,
        "Injury_Periods": pd.NA,
        "Total_Days_Injured": 0,
        "Injury_Risk": "Low Risk",
    }
    for c, v in defaults.items():
        if c not in out.columns:
            out[c] = v

    if injury_data is None or injury_data.empty:
        # normalize empties just in case
        out["Injury_Periods"] = out["Injury_Periods"].replace("", pd.NA)
        return out

    # Process each player/season
    for idx, row in out.iterrows():
        pname = row["Player"]
        season = row["Season"]

        mask = (injury_data["Season"] == season) & \
               (injury_data["Relinquished"].str.contains(pname, case=False, na=False))
        player_inj = injury_data.loc[mask]

        if player_inj.empty:
            continue  # keep defaults

        periods = []
        total_days = 0
        for _, inj in player_inj.iterrows():
            start = inj["Date"]
            # find the first acquired record after start
            got_back = injury_data[
                (injury_data["Date"] > start) &
                (injury_data["Acquired"].str.contains(pname, case=False, na=False))
            ]
            if not got_back.empty:
                end = got_back.iloc[0]["Date"]
            else:
                end_year = int(season.split("-")[1])
                end = pd.Timestamp(f"{end_year}-06-30")

            total_days += (end - start).days
            periods.append(f"{start:%Y-%m-%d} - {end:%Y-%m-%d}")

        out.at[idx, "Injured"] = True
        out.at[idx, "Injury_Periods"] = "; ".join(periods) if periods else pd.NA
        out.at[idx, "Total_Days_Injured"] = total_days

        if total_days < 10:
            risk = "Low Risk"
        elif total_days <= 20:
            risk = "Moderate Risk"
        else:
            risk = "High Risk"
        out.at[idx, "Injury_Risk"] = risk

    # final normalization
    out["Injury_Periods"] = out["Injury_Periods"].replace("", pd.NA)

    return out

# ──────────────────────────────────────────────────────────────────────────────
# USAGE / LOAD METRICS
# Inspired by Basketball-Reference (USG%), Nylon Calculus (True Usage parts),
# and Thinking Basketball (Offensive Load). See docs in code.
# ──────────────────────────────────────────────────────────────────────────────

USAGE_COMPONENT_COLS = [
    "USG%",               # already scraped but we may recompute if missing
    "Scoring_Usage%",     # (FGA + 0.44*FTA) share of team poss
    "Playmaking_Usage%",  # (AST-created FG poss) share
    "Turnover_Usage%",    # TOV share
    "True_Usage%",        # Scoring + Playmaking + TO
    "Offensive_Load%",    # Thinking Basketball style
    "Player_Poss",        # est. possessions used by player
    "Team_Poss",          # est. team possessions (for join/QA)
]

def _safe_div(a, b):
    return np.where(b == 0, np.nan, a / b)

def add_usage_components(df: pd.DataFrame, *, debug: bool = False) -> pd.DataFrame:
    """
    Compute Scoring‑/Playmaking‑/Turnover usage plus Offensive‑Load.

    The helper now:
      • Renames OREB/DREB → ORB/DRB if needed.
      • Warns – but does not crash – when expected stats are missing.
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    # ── 0. Normalise column spelling ───────────────────────────────
    col_map = {"OREB": "ORB", "DREB": "DRB"}
    out.rename(columns={k: v for k, v in col_map.items() if k in out.columns}, inplace=True)

    # ── 1. Summarise team totals ───────────────────────────────────
    want = ["FGA", "FTA", "TOV", "FG", "ORB", "DRB", "TRB", "MP", "AST"]
    have = [c for c in want if c in out.columns]
    if len(have) < len(want) and debug:
        print(f"[usage] missing cols → {sorted(set(want) - set(have))}")

    grp = out.groupby(["Season", "Team"], dropna=False)
    team_totals = grp[have].sum(min_count=1).rename(columns=lambda c: f"Tm_{c}")
    out = out.merge(team_totals, left_on=["Season", "Team"], right_index=True, how="left")

    # ── 2. Possession estimates ────────────────────────────────────
    out["Team_Poss"]   = out["Tm_FGA"] + 0.44 * out["Tm_FTA"] + out["Tm_TOV"]
    out["Player_Poss"] = out["FGA"]    + 0.44 * out["FTA"]    + out["TOV"]

    share = out["Player_Poss"] / out["Team_Poss"]

    # fill USG% if missing
    if "USG%" not in out.columns or out["USG%"].isna().all():
        out["USG%"] = (share * 100).round(2)

    scor = out["FGA"] + 0.44 * out["FTA"]
    tov  = out["TOV"]
    ast_cre = 0.37 * out["AST"]

    out["Scoring_Usage%"]     = (scor / out["Team_Poss"] * 100).round(2)
    out["Turnover_Usage%"]    = (tov  / out["Team_Poss"] * 100).round(2)
    out["Playmaking_Usage%"]  = (ast_cre / out["Team_Poss"] * 100).round(2)
    out["True_Usage%"]        = (out["Scoring_Usage%"] + out["Turnover_Usage%"] +
                                 out["Playmaking_Usage%"]).round(2)

    tsa       = scor
    creation  = 0.8 * out["AST"]
    non_cre   = 0.2 * out["AST"]
    out["Offensive_Load%"] = ((tsa + creation + non_cre + tov) / out["Team_Poss"] * 100).round(2)

    return out


