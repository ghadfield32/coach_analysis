import pandas as pd
import numpy as np
import logging
import sqlite3
from datetime import datetime
from functools import lru_cache
from salary_nba_data_pull.fetch_utils import fetch_all_players, fetch_career_stats, fetch_player_info, fetch_league_standings, fetch_season_players
from salary_nba_data_pull.scrape_utils import scrape_advanced_metrics
from salary_nba_data_pull.name_utils import normalize_name

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
    Calculate shooting percentages and other derived statistics,
    adding indicators and imputing zeros for zero‐denominator cases.
    
    Enhanced Features:
    - Indicator columns (3PA_zero, FTA_zero) flag zero attempts
    - Zero imputation for undefined percentages (no attempts → 0% success)
    - Debug counts for zero-denominator cases
    - ML-ready numeric dataset with preserved semantic meaning
    """
    if df.empty:
        return df

    # 1️⃣ Compute FG% (unchanged - no zero denominator issues)
    if 'FGA' in df.columns and 'FG' in df.columns:
        df['FG%'] = (df['FG'] / df['FGA'] * 100).round(2)
        df['FG%'] = df['FG%'].replace([np.inf, -np.inf], np.nan)

    # 2️⃣ Compute 3P% with debug, indicator, and zero fill
    if '3PA' in df.columns and '3P' in df.columns:
        # Debug: count zero-attempts
        zero_3pa = (df['3PA'] == 0).sum()
        if debug:
            print(f"[calculate_percentages] 3PA==0 count: {zero_3pa}")
        
        # Indicator for zero attempts (preserves information)
        df['3PA_zero'] = df['3PA'] == 0
        
        # Raw percentage calculation (NaN where 3PA==0)
        df['3P%'] = (df['3P'] / df['3PA'] * 100).round(2)
        df['3P%'] = df['3P%'].replace([np.inf, -np.inf], np.nan)
        
        # Impute zeros for undefined cases (no attempts → 0% success)
        df.loc[df['3PA_zero'], '3P%'] = 0.0

    # 3️⃣ Compute FT% with debug, indicator, and zero fill
    if 'FTA' in df.columns and 'FT' in df.columns:
        zero_fta = (df['FTA'] == 0).sum()
        if debug:
            print(f"[calculate_percentages] FTA==0 count: {zero_fta}")
        
        # Indicator for zero attempts
        df['FTA_zero'] = df['FTA'] == 0
        
        # Raw percentage calculation (NaN where FTA==0)
        df['FT%'] = (df['FT'] / df['FTA'] * 100).round(2)
        df['FT%'] = df['FT%'].replace([np.inf, -np.inf], np.nan)
        
        # Impute zeros for undefined cases (no attempts → 0% success)
        df.loc[df['FTA_zero'], 'FT%'] = 0.0

    # 4️⃣ Calculate efficiency metrics (unchanged)
    if 'PTS' in df.columns and 'FGA' in df.columns and 'FTA' in df.columns:
        df['TS%'] = (df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA'])) * 100).round(2)
        df['TS%'] = df['TS%'].replace([np.inf, -np.inf], np.nan)

    # 5️⃣ Per‐36 min rates (unchanged)
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
        print("Percentage calculations completed with zero-denominator handling")
    return df

def process_player_data(player_name: str, season: str,
                        all_players: dict[str, dict], *,
                        debug: bool = False) -> dict | None:
    """
    Build a single‑player dict for a given season with a concrete Team/TeamID.
    For traded players:
      • Prefer a non‑TOT row for that season.
      • Pick the row with max GP (tie‑break by MIN).
    This avoids ambiguous team context that breaks W/L joins.

    Returns:
        dict with uppercased display names and PlayerID included.
    """
    import numpy as np

    meta = all_players.get(player_name.lower().strip())
    if not meta:
        return None

    pid = meta["player_id"]
    info_df   = fetch_player_info(pid, debug=debug)
    career_df = fetch_career_stats(pid, debug=debug)
    if career_df is None or career_df.empty:
        return None

    # rows for the requested season (may include multiple teams + a total row)
    srows = career_df.loc[career_df.SEASON_ID.eq(season)].copy()
    if srows.empty:
        return None

    # Prefer concrete team rows over season "TOT/2TM/3TM" rows
    def _is_total_label(x: str) -> bool:
        x = str(x).upper()
        return x in {"TOT", "2TM", "3TM", "4TM"}  # BBR uses 2TM/3TM; NBA may have "TOT"
    non_tot = srows[~srows["TEAM_ABBREVIATION"].map(_is_total_label)]

    pick_from = non_tot if not non_tot.empty else srows
    # pick the most representative stint: max GP, then MIN
    season_row = (pick_from.sort_values(["GP", "MIN"], ascending=False)
                           .iloc[0])

    # Build the record; enforce uppercase for display names
    data = {
        "Player": player_name.upper(),
        "Season": season,
        "Team":   str(season_row["TEAM_ABBREVIATION"]).upper(),
        "Age":    season_row["PLAYER_AGE"],
        "GP":     season_row["GP"],
        "GS":     season_row.get("GS", 0),
        "MP":     season_row["MIN"],

        "PTS": season_row["PTS"],
        "FG":  season_row["FGM"],  "FGA": season_row["FGA"],
        "3P":  season_row["FG3M"], "3PA": season_row["FG3A"],
        "FT":  season_row["FTM"],  "FTA": season_row["FTA"],

        "TRB": season_row["REB"], "AST": season_row["AST"],
        "STL": season_row["STL"], "BLK": season_row["BLK"],
        "TOV": season_row["TOV"], "PF":  season_row["PF"],

        "ORB": season_row.get("OREB", np.nan),
        "DRB": season_row.get("DREB", np.nan),
    }

    # Include the PlayerID explicitly
    data["PlayerID"] = pid

    # TeamID from the chosen season row whenever possible
    data["TeamID"] = season_row.get("TEAM_ID", np.nan)

    # roster meta (position, experience)
    if info_df is not None and not info_df.empty:
        ir = info_df.iloc[0]
        data["Position"]          = ir.get("POSITION", "")
        data["Years_of_Service"]  = ir.get("SEASON_EXP", None)

    # Derived splits (leave denominator=0 as NaN, do not fill)
    two_att     = data["FGA"] - data["3PA"]
    data["2P"]  = data["FG"] - data["3P"]
    data["2PA"] = two_att
    data["eFG%"] = round((data["FG"] + 0.5 * data["3P"]) / data["FGA"] * 100, 2) if data["FGA"] else np.nan
    data["2P%"]  = round(data["2P"] / two_att * 100, 2)                           if two_att else np.nan

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

# --- NEW: Advanced metrics audit helper ---
def report_advanced_mismatches(player_names: list[str], season: str, *, topk: int = 3):
    """
    Prints players we couldn't match in the BBR advanced table and closest suggestions.
    No filling - just diagnostics.
    """
    import difflib
    from salary_nba_data_pull.scrape_utils import _season_advanced_df
    
    # Prefer the shared normalizer from nba_utils; fall back to local
    try:
        from api.src.airflow_project.utils.nba_utils import normalize_name as _norm
    except Exception:
        from salary_nba_data_pull.scrape_utils import _normalise_name as _norm

    df = _season_advanced_df(season)
    # Ensure the season table uses the same normalizer
    if "player_key" not in df.columns or df["player_key"].isna().all():
        df = df.copy()
        df["player_key"] = df["Player"].map(_norm)

    keys = set(df["player_key"].dropna())
    all_keys = list(keys)
    misses = []
    
    for raw in player_names:
        q = _norm(raw)
        if q not in keys:
            suggestions = difflib.get_close_matches(q, all_keys, n=topk, cutoff=0.75)
            print(f"[adv-miss] {raw}  → key='{q}'  suggestions={suggestions}")
            misses.append(raw)
    
    print(f"[adv-miss] total misses: {len(misses)}/{len(player_names)}")
    return misses

def attach_wins_losses_using_logs(df_season: pd.DataFrame,
                                  season: str,
                                  logs_wl: pd.DataFrame,
                                  *,
                                  debug: bool = False) -> pd.DataFrame:
    """
    Left-merge W/L by TeamID using precomputed team-game-log totals.
    No filling. If TeamID is missing or ambiguous (e.g., TOT), W/L stays NaN.
    """
    if df_season.empty or logs_wl.empty:
        return df_season

    out = df_season.merge(
        logs_wl.drop_duplicates("TeamID"),
        on="TeamID", how="left", validate="m:1"
    )
    if debug:
        null_rate = out["Wins"].isna().mean() * 100
        print(f"[attach_wins_losses_using_logs] {season} W/L null% = {null_rate:.2f}")
    return out


def diagnose_wl_nulls(df_after_merge: pd.DataFrame,
                      season: str,
                      *,
                      debug: bool = True) -> pd.DataFrame:
    """
    Attribute W/L nulls to concrete reasons:
      - TeamID missing
      - team label equals 'TOT' for multi-team season rows
      - TeamID present but no match in W/L lookup
      - Player has 0 GP (edge case)
    Returns a small DataFrame with reason counts/samples.
    """
    import pandas as pd
    if df_after_merge.empty:
        return pd.DataFrame()

    mask_null = df_after_merge["Wins"].isna()
    sub = df_after_merge.loc[mask_null].copy()

    reasons = []
    if "Team" in sub.columns:
        reasons.append(("TOT team label", sub["Team"].str.upper().eq("TOT")))
    reasons.append(("TeamID missing", sub["TeamID"].isna()))
    reasons.append(("Zero GP", sub.get("GP", pd.Series(index=sub.index)).fillna(0).eq(0)))
    # anything else falls into "No W/L match for TeamID"
    base_mask = pd.Series(False, index=sub.index)
    for _, m in reasons:
        base_mask |= m.fillna(False)
    reasons.append(("No W/L match for TeamID", ~base_mask))

    rows = []
    for label, m in reasons:
        cnt = int(m.sum())
        ex = sub.loc[m, ["Player","Team","TeamID"]].head(5).to_dict("records") if cnt else []
        rows.append({"season": season, "reason": label, "count": cnt, "examples": ex})

    diag = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    if debug:
        print("[diagnose_wl_nulls]")
        print(diag)
    return diag


def diagnose_injury_nulls(df: pd.DataFrame,
                          injury_df: pd.DataFrame | None,
                          *,
                          debug: bool = True) -> pd.DataFrame:
    """
    Break down Injury_Periods nulls by season:
      - season beyond injury source coverage
      - player has no injury rows in covered season (legit NA)
    """
    import pandas as pd
    if df.empty:
        return pd.DataFrame()

    by_season = (df.groupby("Season")
                   .agg(total=("Player","count"),
                        nulls=("Injury_Periods", lambda s: int(s.isna().sum())))
                   .assign(null_pct=lambda d: 100*d["nulls"]/d["total"])
                   .reset_index())

    if injury_df is not None and not injury_df.empty:
        covered = set(injury_df["Season"].dropna().unique())
        by_season["in_coverage"] = by_season["Season"].isin(covered)
    else:
        by_season["in_coverage"] = False

    if debug:
        print("[diagnose_injury_nulls]")
        print(by_season)
    return by_season


def audit_min_date_alignment(source_map: dict[str, tuple[pd.DataFrame, list[str]]],
                             *,
                             debug: bool = True) -> pd.DataFrame:
    """
    For each source, compute the earliest season where *all listed columns*
    are non‑NA for at least one row.
    `source_map[name] = (df, ["colA","colB",...])`
    """
    import pandas as pd
    rows = []
    for name, (df, cols) in source_map.items():
        if df is None or df.empty:
            rows.append({"source": name, "min_non_na_season": None, "cols": cols})
            continue
        # seasons with any non-NA across the requested columns
        ok = (df[cols].notna().any(axis=1))
        seasons = pd.Series(df["Season"][ok].dropna().unique())
        min_seas = seasons.sort_values().iloc[0] if not seasons.empty else None
        rows.append({"source": name, "min_non_na_season": min_seas, "cols": cols})
    rep = pd.DataFrame(rows)
    if debug:
        print("[audit_min_date_alignment]")
        print(rep)
    return rep


def enhanced_normalize_name(name: str) -> str:
    """
    Enhanced name normalization that handles common edge cases.
    
    Handles:
    - "Jr.", "Sr.", "II", "III", "IV" suffixes
    - "A.J." vs "AJ" abbreviations
    - "G.G." vs "GG" abbreviations
    - Accented characters and special characters
    - Extra spaces and punctuation
    """
    if not name or pd.isna(name):
        return ""
    
    # Convert to string and lowercase
    name = str(name).lower().strip()
    
    # Handle common abbreviations
    name = name.replace("a.j.", "aj")
    name = name.replace("g.g.", "gg")
    name = name.replace("j.j.", "jj")
    name = name.replace("t.j.", "tj")
    name = name.replace("d.j.", "dj")
    name = name.replace("k.j.", "kj")
    name = name.replace("m.j.", "mj")
    name = name.replace("p.j.", "pj")
    name = name.replace("r.j.", "rj")
    name = name.replace("s.j.", "sj")
    name = name.replace("v.j.", "vj")
    name = name.replace("w.j.", "wj")
    
    # Remove common suffixes that cause matching issues
    suffixes_to_remove = [" jr.", " sr.", " ii", " iii", " iv", " v"]
    for suffix in suffixes_to_remove:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    
    # Handle accented characters
    import unicodedata
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))
    
    # Remove extra spaces and punctuation
    import re
    name = re.sub(r'[^\w\s]', '', name)  # Remove punctuation except spaces
    name = re.sub(r'\s+', ' ', name)     # Normalize spaces
    name = name.strip()
    
    return name

def diagnose_advanced_nulls(df: pd.DataFrame,
                            season: str,
                            *,
                            max_print: int = 5,
                            tail_print: int = 0,
                            show_all: bool = False,
                            debug: bool = True,
                            prefer_left: set[str] = {"TS%", "USG%"},
                           ) -> dict:
    """
    Summarise advanced-stat availability for a season based on *BBR-only* columns.
    We exclude columns we intentionally keep from the left (e.g., TS%, USG%)
    so that unmatched BBR rows are not masked.

    Parameters:
        max_print: int - how many head samples to show (default: 5)
        tail_print: int - how many tail samples to show (default: 0)
        show_all: bool - if True, return and print ALL unmatched players
        debug: bool - whether to print diagnostics
        prefer_left: set - columns to prefer from left side (TS%, USG%)

    Returns a dict with:
      - season
      - players
      - adv_all_na_rows: count of rows where *all attached-from-BBR* cols are NA
      - per_col_nulls: per-column NA counts for those BBR-only cols
      - sample_head: first max_print unmatched players
      - sample_tail: last tail_print unmatched players
      - sample_all: all unmatched players if show_all=True

    No filling, no mutation.
    """
    from salary_nba_data_pull.scrape_utils import ADV_METRIC_COLS

    if df.empty:
        out = {
            "season": season,
            "players": 0,
            "adv_all_na_rows": 0,
            "per_col_nulls": {},
            "sample_head": [],
            "sample_tail": [],
            "sample_all": []
        }
        if debug: print("[diagnose_advanced_nulls]", out)
        return out

    # Focus only on columns we expect to be attached from BBR
    bbr_only = [c for c in ADV_METRIC_COLS if c in df.columns and c not in prefer_left]
    if not bbr_only:
        out = {
            "season": season,
            "players": len(df),
            "adv_all_na_rows": 0,
            "per_col_nulls": {},
            "sample_head": [],
            "sample_tail": [],
            "sample_all": []
        }
        if debug:
            print(f"[diagnose_advanced_nulls] {season}: no BBR-only adv cols present")
        return out

    # Mask rows where *all* BBR-only cols are NA
    all_na_mask = df[bbr_only].isna().all(axis=1)
    adv_all_na_rows = int(all_na_mask.sum())

    # Extract the unmatched Player names
    unmatched_players = df.loc[all_na_mask, "Player"]
    sample_head = unmatched_players.head(max_print).tolist()
    sample_tail = unmatched_players.tail(tail_print).tolist() if tail_print > 0 else []
    sample_all = unmatched_players.tolist() if show_all else []

    # Per-column null counts
    per_col_nulls = df[bbr_only].isna().sum().sort_values(ascending=False).to_dict()

    out = {
        "season": season,
        "players": len(df),
        "adv_all_na_rows": adv_all_na_rows,
        "per_col_nulls": per_col_nulls,
        "sample_head": sample_head,
        "sample_tail": sample_tail,
        "sample_all": sample_all,
    }

    if debug:
        print(f"[diagnose_advanced_nulls] {season}: rows with ALL BBR-only adv cols NA = "
              f"{adv_all_na_rows}/{len(df)}")
        if adv_all_na_rows:
            print(f"  head sample ({max_print}): {sample_head}")
            if tail_print > 0:
                print(f"  tail sample ({tail_print}): {sample_tail}")
            if show_all:
                print(f"  all unmatched ({len(sample_all)}): {sample_all}")
        print(f"  per-col nulls (BBR-only): {per_col_nulls}")

    return out


def attach_wins_losses(df_season: pd.DataFrame,
                       season: str,
                       *,
                       debug: bool = False) -> pd.DataFrame:
    """
    Left-merge W/L using unified lookup (game logs + standings).
    Emits a reason breakdown for any residual nulls.
    """
    if df_season.empty:
        return df_season

    from salary_nba_data_pull.fetch_utils import fetch_team_wl_lookup
    wl = fetch_team_wl_lookup(season, season_type="Regular Season", debug=debug)

    out = df_season.merge(wl.drop_duplicates("TeamID"),
                          on="TeamID", how="left", validate="m:1")
    if debug:
        null_rate = out["Wins"].isna().mean() * 100
        print(f"[attach_wins_losses] {season} W/L null% = {null_rate:.2f}")
        if null_rate > 0:
            _ = diagnose_wl_nulls(out, season, debug=True)
    return out


# ──────────────────────────────────────────────────────────────────────────────
#  Consolidate _x / _y duplicate columns
# ──────────────────────────────────────────────────────────────────────────────
def _choose_preferred_column(df: pd.DataFrame, col_a: str, col_b: str) -> str:
    """
    Return the column name (col_a or col_b) that should survive a
    consolidate-duplicates decision.

    Heuristic:
    1. Keep the one with *fewer NaNs*.
    2. If tied, keep the one that is *not all-NaN*.
    3. If still tied (both fully populated or both empty), keep the one that
       comes first alphabetically (stable with previous behaviour).

    The caller is responsible for ensuring both columns exist in *df*.
    """
    na_a = df[col_a].isna().sum()
    na_b = df[col_b].isna().sum()

    if na_a < na_b:
        return col_a
    if na_b < na_a:
        return col_b

    # tie – favour the column that isn't entirely NaN
    if df[col_a].notna().any() and df[col_b].isna().all():
        return col_a
    if df[col_b].notna().any() and df[col_a].isna().all():
        return col_b

    # final tie-break – deterministic old rule
    return min(col_a, col_b)



def consolidate_duplicate_columns(df: pd.DataFrame,
                                  *,
                                  preferred: str | None = None,
                                  debug: bool = False) -> pd.DataFrame:
    """
    Collapse *_x / *_y duplicates **and** guarantee no duplicate labels remain.
    Additionally, assert that every column in CRITICAL_ID_COLS still exists.
    """
    from salary_nba_data_pull.data_utils import CRITICAL_ID_COLS
    from salary_nba_data_pull.main import _almost_equal_numeric
    
    out = df.copy()
    suff_pairs: dict[str, list[str]] = {}

    # ── Phase 1 – detect duplicate suffix pairs ───────────────────────────
    for col in out.columns:
        if col.endswith(("_x", "_y")):
            base = col[:-2]
            suff_pairs.setdefault(base, []).append(col)

    # ── Phase 2 – resolve each pair/group ────────────────────────────────
    for base, cols in suff_pairs.items():
        cols = sorted(cols)                                  # deterministic
        winner: str

        if len(cols) == 1:                                   # only _x **or** _y
            winner = cols[0]
        else:                                                # both present
            if preferred in {"_x", "_y"}:                    # explicit hint
                pref_col = f"{base}{preferred}"
                if pref_col in cols:
                    winner = pref_col
                else:
                    winner = _choose_preferred_column(out, *cols)
            else:
                winner = _choose_preferred_column(out, *cols)

            # Show mismatches that matter
            other = [c for c in cols if c != winner][0]
            unequal = ~_almost_equal_numeric(out[winner], out[other])
            if debug and unequal.any():
                nbad = int(unequal.sum())
                print(f"[consolidate] '{base}': kept '{winner}' "
                      f"(better NaN profile) – {nbad}/{len(out)} rows differed")

        # finally: rename winner → base, drop the rest
        out.rename(columns={winner: base}, inplace=True)
        drop_cols = [c for c in cols if c != winner]
        out.drop(columns=drop_cols, inplace=True)

    # ── Phase 3 – guarantee column-label uniqueness ──────────────────────
    dup_labels = out.columns[out.columns.duplicated()].unique()
    if dup_labels.size:
        if debug:
            print(f"[consolidate] WARNING: removing duplicate labels "
                  f"{dup_labels.tolist()}")
        out = out.loc[:, ~out.columns.duplicated()]

    # -------------------------------------------------------------
    # 🔒 Sanity – critical IDs must survive
    # -------------------------------------------------------------
    missing = [c for c in CRITICAL_ID_COLS if c not in out.columns]
    if missing:
        raise RuntimeError(
            f"[consolidate_duplicate_columns] lost critical columns: {missing}"
        )

    return out





def merge_advanced_metrics(df_season: pd.DataFrame,
                           season: str,
                           *,
                           debug: bool = False,
                           name_overrides: dict[str, str] | None = None) -> pd.DataFrame:
    """
    Attach Basketball-Reference advanced metrics for one season.

    • Keeps caller’s TS% / USG% values (NBA API derived) **without creating
      duplicate *_x / *_y columns**.                    ← CHG
    • Drops BBR “Team”, “MP”, “TS%”, “USG%” **before** merging. ← NEW
    • Emits match diagnostics but never fills data.
    """
    if df_season.empty:
        return df_season

    from difflib import get_close_matches
    from salary_nba_data_pull.scrape_utils import _season_advanced_df, ADV_METRIC_COLS
    from salary_nba_data_pull.name_utils     import normalize_name

    adv = _season_advanced_df(season)
    if adv is None or adv.empty:
        if debug:
            print(f"[merge_advanced_metrics] no advanced table for {season}")
        return df_season

    # ── Build player keys ─────────────────────────────────────────────
    adv = adv.copy()
    adv["player_key"] = adv["Player"].map(normalize_name)

    df  = df_season.copy()
    df["player_key"] = df["Player"].map(normalize_name)

    # ── Decide which columns to attach ───────────────────────────────
    prefer_left = {"TS%", "USG%"}                 # we already calculated these
    adv_cols_available = [c for c in ADV_METRIC_COLS if c in adv.columns]
    attach_cols = [c for c in adv_cols_available if c not in prefer_left]

    # ── Prepare a one-row-per-key BBR slice ──────────────────────────
    team_col = next((c for c in ["Team", "Tm", "TEAM"] if c in adv.columns), None)

    pick_cols = ["player_key"] + attach_cols
    if "MP" in adv.columns:               # needed only for sorting, drop later
        pick_cols.append("MP")
    if team_col:
        pick_cols.append(team_col)

    adv_small = adv.loc[:, pick_cols].copy()

    # Pick 'TOT' first (if present), then max MP
    def _is_tot(x: str) -> bool:
        return str(x).upper() in {"TOT", "2TM", "3TM", "4TM"}

    sort_cols, ascending = ["player_key"], [True]
    if team_col:
        adv_small["_is_tot"] = adv_small[team_col].map(_is_tot).astype(int)
        sort_cols += ["_is_tot"]; ascending += [False]
    if "MP" in adv_small.columns:
        sort_cols += ["MP"]; ascending += [False]

    adv_small = adv_small.sort_values(sort_cols, ascending=ascending)\
                         .drop_duplicates("player_key", keep="first")

    # Drop helper cols so they can't collide in the merge            ← NEW
    adv_small = adv_small.drop(columns=[c for c in ["MP", team_col, "_is_tot"] if c in adv_small.columns])

    # ── Merge (no suffixes needed now) ───────────────────────────────
    merged = df.merge(adv_small, on="player_key", how="left", validate="m:1")

    # ── Diagnostics ─────────────────────────────────────────────────
    if debug:
        added_cols = [c for c in attach_cols if c in merged.columns]
        print(f"[merge_advanced_metrics] {season}: attached {len(added_cols)} cols – {added_cols[:10]}…")

        # Success rate
        if attach_cols:
            all_na = merged[attach_cols].isna().all(axis=1)
            matched = (~all_na).sum()
            print(f"[merge_advanced_metrics] {season}: matched {matched}/{len(merged)} "
                  f"players ({matched/len(merged)*100:.1f} %)")

            if all_na.any():
                sample = merged.loc[all_na, ["Player"]].head(5)["Player"].tolist()
                print(f"  unmatched sample: {sample}")

    # ── Display normalization: uppercase the Player and Team for consistent downstream visibility
    if "Player" in merged.columns:
        merged["Player"] = merged["Player"].str.upper()
    if "Team" in merged.columns:
        merged["Team"] = merged["Team"].astype(str).str.upper()

    return merged


def guard_advanced_null_regress(df_new: pd.DataFrame,
                                season: str,
                                *,
                                base_dir: str | None = None,
                                debug: bool = True) -> None:
    """
    Season-aware guard: compare the count of rows whose *all* advanced metrics
    are NA in the *new* season vs the *previous* parquet for the SAME season.

    This is READ-ONLY and prints diagnostics. It does not mutate data or fill.
    """
    from pathlib import Path
    from salary_nba_data_pull.scrape_utils import ADV_METRIC_COLS

    if df_new.empty:
        if debug:
            print(f"[guard_advanced_null_regress] {season}: empty new df")
        return

    # Only check columns that actually exist after merge
    adv_cols_present = [c for c in ADV_METRIC_COLS if c in df_new.columns]
    if not adv_cols_present:
        if debug:
            print(f"[guard_advanced_null_regress] {season}: no advanced cols present to check")
        return

    # New all-advanced-NA count (per row)
    na_mask_new = df_new[adv_cols_present].isna().all(axis=1)
    new_all_na = int(na_mask_new.sum())
    total = len(df_new)

    # Load previous parquet for the same season
    root = Path(base_dir) if base_dir else Path(__file__).resolve().parents[3] / "data" / "new_processed"
    prev_path = root / f"season={season}" / "part.parquet"

    prev_all_na = None
    if prev_path.exists():
        try:
            prev = pd.read_parquet(prev_path)
            prev_cols = [c for c in adv_cols_present if c in prev.columns]
            if prev_cols:
                na_mask_prev = prev[prev_cols].isna().all(axis=1)
                prev_all_na = int(na_mask_prev.sum())
        except Exception as exc:
            if debug:
                print(f"[guard_advanced_null_regress] {season}: failed to read previous parquet -> {exc!s}")

    if debug:
        print(f"[guard_advanced_null_regress] {season}: all-advanced-NA rows: "
              f"new={new_all_na}/{total}"
              + (f"  prev={prev_all_na}" if prev_all_na is not None else "  prev=<none>"))

    # This is a guardrail/print only. We DO NOT fail the run or fill values.

def report_advanced_join_issues(df_after_merge: pd.DataFrame,
                                season: str,
                                *,
                                topk: int = 3,
                                max_rows: int = 25,
                                debug: bool = True) -> pd.DataFrame:
    """
    Diagnostic only: list rows where *all BBR-only* advanced metrics are NA,
    and show how keys differ between normalizers.

    Columns returned:
      Player, Team, player_key_left,
      adv_key_enh (enhanced_normalize_name on BBR 'Player'),
      adv_key_bbr (legacy _normalise_name (Unidecode) on BBR 'Player'),
      close_matches_enh, close_matches_bbr

    No mutation. Safe to run anytime after merge_advanced_metrics.
    """
    import difflib
    from salary_nba_data_pull.scrape_utils import _season_advanced_df, ADV_METRIC_COLS
    # Normalizers
    try:
        # the same enhanced used on the left df
        norm_left = enhanced_normalize_name
    except NameError:
        from salary_nba_data_pull.process_utils import enhanced_normalize_name as norm_left
    try:
        from salary_nba_data_pull.scrape_utils import _normalise_name as norm_bbr_legacy
    except Exception:
        norm_bbr_legacy = norm_left  # fallback

    if df_after_merge.empty:
        if debug: print("[report_advanced_join_issues] empty frame")
        return pd.DataFrame()

    # Identify rows where all BBR-only cols are NA
    bbr_only = [c for c in ADV_METRIC_COLS if c in df_after_merge.columns and c not in {"TS%", "USG%"}]
    if not bbr_only:
        if debug: print("[report_advanced_join_issues] no BBR-only columns present")
        return pd.DataFrame()

    mask = df_after_merge[bbr_only].isna().all(axis=1)
    left_unmatched = df_after_merge.loc[mask, ["Player", "Team"]].copy()
    if left_unmatched.empty:
        if debug: print("[report_advanced_join_issues] no unmatched rows on BBR-only cols")
        return pd.DataFrame()

    # Build left keys
    left_unmatched["player_key_left"] = left_unmatched["Player"].map(norm_left)

    # Load the season advanced table
    adv = _season_advanced_df(season)
    adv = adv[["Player"]].drop_duplicates()
    adv["adv_key_enh"] = adv["Player"].map(norm_left)
    adv["adv_key_bbr"] = adv["Player"].map(norm_bbr_legacy)

    keys_enh = adv["adv_key_enh"].dropna().unique().tolist()
    keys_bbr = adv["adv_key_bbr"].dropna().unique().tolist()

    def _cmatch(k, pool):
        return difflib.get_close_matches(k, pool, n=topk, cutoff=0.80)

    out_rows = []
    for _, r in left_unmatched.head(max_rows).iterrows():
        lk = r["player_key_left"]
        out_rows.append({
            "Player": r["Player"],
            "Team": r.get("Team"),
            "player_key_left": lk,
            "close_matches_enh": _cmatch(lk, keys_enh),
            "close_matches_bbr": _cmatch(lk, keys_bbr),
        })
    out = pd.DataFrame(out_rows)
    if debug:
        print("[report_advanced_join_issues] sample unmatched with suggestions:")
        print(out.to_string(index=False))
    return out

def investigate_unmatched_players(df_after_merge: pd.DataFrame,
                                 season: str,
                                 *,
                                 debug: bool = True) -> pd.DataFrame:
    """
    Investigate players that exist in NBA API but not in BBR advanced stats.
    
    This helps identify legitimate data source discrepancies vs encoding issues.
    
    Args:
        df_after_merge: DataFrame after merge_advanced_metrics
        season: Season string
        debug: Whether to print diagnostics
        
    Returns:
        DataFrame with investigation results
    """
    from salary_nba_data_pull.scrape_utils import ADV_METRIC_COLS
    
    if df_after_merge.empty:
        return pd.DataFrame()
    
    # Find rows where all BBR-only advanced metrics are NA
    bbr_only = [c for c in ADV_METRIC_COLS if c in df_after_merge.columns and c not in {"TS%", "USG%"}]
    if not bbr_only:
        return pd.DataFrame()
    
    unmatched_mask = df_after_merge[bbr_only].isna().all(axis=1)
    unmatched_df = df_after_merge.loc[unmatched_mask].copy()
    
    if unmatched_df.empty:
        if debug:
            print(f"[investigate_unmatched_players] {season}: No unmatched players found")
        return pd.DataFrame()
    
    # Add investigation columns
    unmatched_df['MP_threshold'] = unmatched_df['MP'] < 200  # Common BBR threshold
    unmatched_df['GP_threshold'] = unmatched_df['GP'] < 10   # Common BBR threshold
    
    # Load BBR data to check if player exists
    from src.salary_nba_data_pull.scrape_utils import _season_advanced_df
    from src.salary_nba_data_pull.name_utils import normalize_name
    bbr_df = _season_advanced_df(season)
    
    investigation_results = []
    for _, row in unmatched_df.iterrows():
        player_name = row['Player']
        player_key = normalize_name(player_name)
        
        # Check if player exists in BBR
        bbr_match = bbr_df[bbr_df['player_key'] == player_key]
        exists_in_bbr = len(bbr_match) > 0
        
        # Check for similar names
        similar_names = bbr_df[bbr_df['Player'].str.contains(player_name.split()[-1], case=False, na=False)]
        
        result = {
            'Player': player_name,
            'Team': row.get('Team', ''),
            'GP': row.get('GP', 0),
            'MP': row.get('MP', 0),
            'exists_in_bbr': exists_in_bbr,
            'low_mp': row.get('MP', 0) < 200,
            'low_gp': row.get('GP', 0) < 10,
            'similar_names_count': len(similar_names),
            'similar_names': similar_names['Player'].tolist()[:3] if len(similar_names) > 0 else []
        }
        investigation_results.append(result)
    
    results_df = pd.DataFrame(investigation_results)
    
    if debug:
        print(f"[investigate_unmatched_players] {season}: Found {len(results_df)} unmatched players")
        for _, row in results_df.iterrows():
            print(f"  {row['Player']} ({row['Team']}) - {row['GP']} GP, {row['MP']} MP")
            print(f"    Exists in BBR: {row['exists_in_bbr']}")
            print(f"    Low MP: {row['low_mp']}, Low GP: {row['low_gp']}")
            if row['similar_names_count'] > 0:
                print(f"    Similar names: {row['similar_names']}")
            print()
    
    return results_df


# ────────────────────────────────────────────────────────────────
# NEW utility – guarantee PlayerID / TeamID are present
# ────────────────────────────────────────────────────────────────
def ensure_player_ids(df: pd.DataFrame, season: str,
                      *, debug: bool = False) -> pd.DataFrame:
    """
    For legacy parquet partitions that pre-date the PlayerID field.

    • Uses `fetch_season_players()` once per season (cached)  
    • Matches on the same `normalize_name()` key the rest of the pipeline uses  
    • Fills **only the null rows** – never overwrites an existing ID  
    • Also fills `TeamID` when missing and unambiguous

    Returns a *copy* so the caller keeps purity.
    """
    if df.empty:
        return df

    if "PlayerID" in df.columns and df["PlayerID"].notna().all():
        # nothing to do – fast path
        return df

    roster = fetch_season_players(season, debug=debug)      # cached
    if debug:
        print(f"[ensure_player_ids] back-filling IDs for {season} "
              f"(roster size {len(roster)})")

    out = df.copy()
    # Build key once – works whether the column existed or not
    out["__key"] = out["Player"].map(normalize_name)

    # Create columns if they were totally missing
    if "PlayerID" not in out.columns:
        out["PlayerID"] = pd.NA
    if "TeamID" not in out.columns:
        out["TeamID"] = pd.NA

    for k, meta in roster.items():
        mask = (out["__key"] == k) & (out["PlayerID"].isna())
        if mask.any():
            out.loc[mask, "PlayerID"] = meta["player_id"]
            # Fill TeamID only when unambiguous (1 franchise per season key)
            out.loc[mask & out["TeamID"].isna(), "TeamID"] = meta["team_id"]

    out.drop(columns="__key", inplace=True)

    # Final sanity
    if debug:
        miss = int(out["PlayerID"].isna().sum())
        if miss:
            print(f"[ensure_player_ids] WARNING: {miss} rows still lack PlayerID")

    return out

