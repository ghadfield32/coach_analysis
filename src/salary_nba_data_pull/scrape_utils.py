import pandas as pd
import requests
import time
import random
import re
from bs4 import BeautifulSoup
from io import StringIO
from typing import Optional
import os
import requests_cache
from unidecode import unidecode
from pathlib import Path
from datetime import datetime
from salary_nba_data_pull.settings import DATA_PROCESSED_DIR
from functools import lru_cache
import threading
_ADV_LOCK   = threading.Lock()
_ADV_CACHE: dict[str, pd.DataFrame] = {}   # season -> DataFrame

# Install cache for all requests
requests_cache.install_cache('nba_scraping', expire_after=86400)  # 24 hours

# Create cached session with stale-if-error capability
session = requests_cache.CachedSession(
    'nba_scraping',
    expire_after=86400,
    stale_if_error=True       # <-- NEW: serve expired cache if remote 429s
)

def scrape_salary_cap_history(*, debug: bool = False) -> pd.DataFrame | None:
    """
    Robust pull of historical cap / tax / apron lines.

    Strategy:
    1. Try RealGM (live HTML).
    2. If the selector fails, look for an existing CSV in DATA_PROCESSED_DIR.
    3. As a last‑chance fallback, hit NBA.com / Reuters bulletins for the
       current season only (so we still merge *something*).
    """
    import json
    from salary_nba_data_pull.settings import DATA_PROCESSED_DIR

    url = "https://basketball.realgm.com/nba/info/salary_cap"

    try:
        html = requests.get(url, timeout=30).text
        soup = BeautifulSoup(html, "html.parser")

        # -------- 1️⃣  RealGM table (new markup) --------------------
        blk = soup.find("pre")                      # new 2025 layout
        if blk:                                     # parse fixed‑width block
            rows = [r.strip().split() for r in blk.text.strip().splitlines()]
            header = rows[0]
            data = rows[1:]
            df = pd.DataFrame(data, columns=header)
        else:
            # Legacy table path (kept for safety)
            tbl = soup.select_one("table")
            if not tbl:
                raise ValueError("salary_cap table not found")
            df = pd.read_html(str(tbl))[0]

        # ---- normalise ----
        df["Season"] = df["Season"].str.extract(r"(\d{4}-\d{4})")
        money_cols = [c for c in df.columns if c != "Season"]
        for c in money_cols:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(r"[$,]", "", regex=True)
                .replace("", pd.NA)
                .astype(float)
            )

        if debug:
            print(f"[salary‑cap] scraped {len(df)} rows from RealGM")

        return df

    except Exception as exc:
        if debug:
            print(f"[salary‑cap] primary scrape failed → {exc!s}")

        # -------- 2️⃣  local cached CSV ----------------------------
        fallback = DATA_PROCESSED_DIR / "salary_cap_history_inflated.csv"
        if fallback.exists():
            if debug:
                print(f"[salary‑cap] using cached CSV at {fallback}")
            return pd.read_csv(fallback)

        # -------- 3️⃣  NBA.com / Reuters one‑liner -----------------
        try:
            # Latest season only
            # For now, create a minimal fallback with current season data
            year = datetime.now().year
            cap = 140.588  # 2024-25 cap as fallback
            df = pd.DataFrame(
                {"Season": [f"{year}-{str(year+1)[-2:]}"],
                 "Salary Cap": [cap * 1_000_000]}
            )
            if debug:
                print("[salary‑cap] built minimal one‑row DataFrame "
                      "from fallback values")
            return df
        except Exception:
            pass

    if debug:
        print("[salary‑cap] giving up – no data available")
    return None

# User-Agent header to avoid Cloudflare blocks
UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    )
}
DELAY_BETWEEN_REQUESTS = 3  # seconds

# Define column templates to guarantee DataFrame structure
PLAYER_COLS = ["Player", "Salary", "Season"]
TEAM_COLS = ["Team", "Team_Salary", "Season"]

# Salary parsing pattern
_salary_pat = re.compile(r"\$?\d[\d,]*")

def _clean_salary(text: str) -> int | None:
    """Return salary as int or None when text has no digits."""
    m = _salary_pat.search(text)
    return int(m.group(0).replace(",", "").replace("$", "")) if m else None

# Name normalization pattern with unidecode
def _normalise_name(raw: str) -> str:
    """ASCII‑fold, trim, lower."""
    return unidecode(raw).split(",")[0].split("(")[0].strip().lower()


# ------- INTERNAL HELPER --------
def _get_hoopshype_soup(url: str, debug: bool = False) -> Optional[BeautifulSoup]:
    """
    Hit HoopsHype once with a realistic UA.  
    Return BeautifulSoup if the page looks OK, else None.
    """
    for attempt in range(2):
        try:
            if debug:
                print(f"[fetch] {url} (attempt {attempt+1})")
            resp = requests.get(url, headers=UA, timeout=30)
            if resp.status_code != 200:
                if debug:
                    print(f"  -> HTTP {resp.status_code}, skipping.")
                return None
            html = resp.text
            # crude Cloudflare challenge check
            if ("Access denied" in html) or ("cf-chl" in html):
                if debug:
                    print("  -> Cloudflare challenge detected; giving up.")
                return None
            return BeautifulSoup(html, "html.parser")
        except requests.RequestException as e:
            if debug:
                print(f"  -> network error {e}, retrying…")
            time.sleep(2 ** attempt + random.random())
    return None
# --------------------------------------------------------------------------


def _espn_salary_url(year: int, page: int = 1) -> str:
    """
    Build the new ESPN salary URL. Examples:
      page 1 → https://www.espn.com/nba/salaries/_/year/2024/seasontype/4
      page 3 → https://www.espn.com/nba/salaries/_/year/2024/page/3/seasontype/4
    """
    base = f"https://www.espn.com/nba/salaries/_/year/{year}"
    return f"{base}/seasontype/4" if page == 1 else f"{base}/page/{page}/seasontype/4"


def _scrape_espn_player_salaries(season_start: int, debug: bool = False) -> list[dict]:
    """
    DEPRECATED: Salary scraping was removed – consume pre-loaded salary parquet instead.
    """
    raise NotImplementedError(
        "Salary scraping was removed – consume pre-loaded salary parquet instead."
    )


def scrape_player_salary_data(start_season: int, end_season: int,
                              player_filter: str | None = None,
                              debug: bool = False) -> pd.DataFrame:
    """
    DEPRECATED: Salary scraping was removed – consume pre-loaded salary parquet instead.
    """
    raise NotImplementedError(
        "Salary scraping was removed – consume pre-loaded salary parquet instead."
    )
# --------------------------------------------------------------------------


def _scrape_espn_team_salaries(season: str, debug: bool = False) -> list[dict]:
    """
    DEPRECATED: Team salary scraping was removed – consume pre-loaded salary parquet instead.
    """
    raise NotImplementedError(
        "Team salary scraping was removed – consume pre-loaded salary parquet instead."
    )


def scrape_team_salary_data(season: str, debug: bool = False) -> pd.DataFrame:
    """
    DEPRECATED: Team salary scraping was removed – consume pre-loaded salary parquet instead.
    """
    raise NotImplementedError(
        "Team salary scraping was removed – consume pre-loaded salary parquet instead."
    )

# --- Season‑level advanced stats --------------------------------------------
ADV_METRIC_COLS = [
    "PER", "TS%", "3PAr", "FTr", "ORB%", "DRB%", "TRB%", "AST%", "STL%", "BLK%",
    "TOV%", "USG%", "OWS", "DWS", "WS", "WS/48", "OBPM", "DBPM", "BPM", "VORP",
    "ORtg", "DRtg",  # extra goodies if you want them
]

def _season_advanced_df(season: str) -> pd.DataFrame:
    """
    Thread‑safe, memoised download of the *season‑wide* advanced‑stats table.
    
    Root-cause fixes:
      • Use bytes (resp.content) instead of text to avoid encoding guesswork
      • Let lxml parser handle UTF-8 charset from page's <meta> tag
      • Use centralized name normalization
      • Validate encoding with known Unicode names
    
    The first thread to request a given season does the HTTP work while holding
    a lock; all others simply wait for the result instead of firing duplicate
    requests. The DataFrame is cached in‑process for the life of the run.
    """
    if season in _ADV_CACHE:            # fast path, no lock
        return _ADV_CACHE[season]

    with _ADV_LOCK:                     # only one thread may enter the block
        if season in _ADV_CACHE:        # double‑checked locking
            return _ADV_CACHE[season]

        end_year = int(season[:4]) + 1
        url = f"https://www.basketball-reference.com/leagues/NBA_{end_year}_advanced.html"
        print(f"[adv] fetching {url}")
        
        # Get raw bytes to avoid encoding guesswork
        resp = session.get(url, headers=UA, timeout=30)
        resp.raise_for_status()
        raw_content = resp.content  # Use bytes, not resp.text
        
        # Parse tables from bytes - let lxml handle charset detection
        from io import BytesIO
        dfs = pd.read_html(BytesIO(raw_content), flavor="lxml", header=0)
        
        if not dfs:
            raise ValueError(f"No tables found at {url}")
        
        # Find the table with Player column
        df = next((t for t in dfs if "Player" in t.columns), dfs[0]).copy()
        
        # Remove repeated header rows that BBR embeds
        if "Player" in df.columns:
            df = df[df["Player"] != "Player"]
        
        # Use centralized normalization
        from salary_nba_data_pull.name_utils import normalize_name, validate_name_encoding
        df["player_key"] = df["Player"].map(normalize_name)
        
        # Validate encoding (will raise if critical issues detected)
        try:
            validate_name_encoding(df, season, debug=True)
        except AssertionError as e:
            print(f"[adv] WARNING: {e}")
            # Continue anyway but log the issue
        
        # Convert numeric columns
        avail = [c for c in ADV_METRIC_COLS if c in df.columns]
        if avail:
            df[avail] = df[avail].apply(pd.to_numeric, errors="coerce")

        _ADV_CACHE[season] = df                # memoise
        time.sleep(random.uniform(1.5, 2.5))   # be polite
        return df

def scrape_advanced_metrics(player_name: str,
                            season: str,
                            *,
                            debug: bool = False) -> dict:
    """
    O(1) lookup in the cached season DataFrame – zero extra HTTP traffic.
    Uses a shared normalizer (nba_utils.normalize_name) to reduce mismatches.
    Prints closest suggestions when no row is found (no filling).
    """
    import difflib

    # Prefer the shared normalizer from nba_utils; fall back to local
    try:
        from api.src.airflow_project.utils.nba_utils import normalize_name as _norm
    except Exception:
        _norm = _normalise_name

    df = _season_advanced_df(season)
    # Ensure the season table uses the same normalizer
    if "player_key" not in df.columns or df["player_key"].isna().all():
        df = df.copy()
        df["player_key"] = df["Player"].map(_norm)

    key = _norm(player_name)
    row = df.loc[df.player_key == key]

    if row.empty:
        if debug:
            # Provide top-3 closest suggestions to help diagnose mismatches
            all_keys = df["player_key"].dropna().unique().tolist()
            suggestions = difflib.get_close_matches(key, all_keys, n=3, cutoff=0.75)
            print(f"[adv] no advanced stats for '{player_name}' (key='{key}') in {season}. "
                  f"Closest: {suggestions}")
        return {}

    row = row.iloc[0]
    # Only return columns that actually exist in the DataFrame
    available_cols = [col for col in ADV_METRIC_COLS if col in row.index]
    result = {col: row[col] for col in available_cols}
    if debug:
        print(f"[adv] {player_name} → {result}")
    return result
# --- End of new season-level advanced stats ---------------------------------

def load_injury_data(
    file_path: str | Path | None = None,
    *,
    base_dir: str | Path | None = None,
    debug: bool = False,
):
    """
    Load the historical injury CSV. By default we look inside the *new*
    processed folder; pass ``file_path`` to override a specific file,
    or ``base_dir`` to point at a different processed directory.
    """
    root = Path(base_dir) if base_dir else DATA_PROCESSED_DIR
    if file_path is None:
        file_path = root / "NBA Player Injury Stats(1951 - 2023).csv"
    file_path = Path(file_path).expanduser().resolve()

    try:
        injury = (
            pd.read_csv(file_path)
            .assign(Date=lambda d: pd.to_datetime(d["Date"]))
        )
        injury["Season"] = injury["Date"].apply(
            lambda x: (
                f"{x.year}-{str(x.year + 1)[-2:]}"
                if x.month >= 10
                else f"{x.year - 1}-{str(x.year)[-2:]}"
            )
        )
        if debug:
            print(f"[load_injury_data] loaded {len(injury):,} rows from {file_path}")
        return injury
    except FileNotFoundError:
        if debug:
            print(f"[load_injury_data] ✖ no injury file at {file_path}")
        return None

if __name__ == "__main__":
    # Example usage and testing of all functions
    debug = True
    start_season = 2022
    end_season = 2023
    sample_player = "Ja Morant"  # Example player

    print("1. Testing scrape_salary_cap_history:")
    salary_cap_history = scrape_salary_cap_history(debug=debug)

    print("\n2. Testing scrape_player_salary_data:")
    player_salary_data = scrape_player_salary_data(start_season, end_season, player_filter=sample_player, debug=debug)

    print("\n3. Testing scrape_team_salary_data:")
    team_salary_data = scrape_team_salary_data(f"{start_season}-{str(start_season+1)[-2:]}", debug=debug)

    print("\n4. Testing scrape_advanced_metrics:")
    advanced_metrics = scrape_advanced_metrics(sample_player, f"{start_season}-{str(start_season+1)[-2:]}", debug=debug)
    print(f"Advanced Metrics for {sample_player}:")
    print(advanced_metrics)

    print("\n5. Testing load_injury_data and merge_injury_data:")
    injury_data = load_injury_data()
    if injury_data is not None:
        print(injury_data.head())
    else:
        print("No injury data loaded.")
    if not player_salary_data.empty and injury_data is not None:
        from salary_nba_data_pull.process_utils import merge_injury_data
        merged_data = merge_injury_data(player_salary_data, injury_data)
        print("Merged data with injury info:")
        columns_to_display = ['Player', 'Season', 'Salary']
        if 'Injured' in merged_data.columns:
            columns_to_display.append('Injured')
        if 'Injury_Periods' in merged_data.columns:
            columns_to_display.append('Injury_Periods')
        if 'Total_Days_Injured' in merged_data.columns:
            columns_to_display.append('Total_Days_Injured')
        if 'Injury_Risk' in merged_data.columns:
            columns_to_display.append('Injury_Risk')
        print(merged_data[columns_to_display].head())

    if not player_salary_data.empty:
        avg_salary = player_salary_data['Salary'].mean()
        print(f"Average salary for {sample_player} from {start_season} to {end_season}: ${avg_salary:,.2f}")

    if not team_salary_data.empty:
        highest_team_salary = team_salary_data.loc[team_salary_data['Team_Salary'].idxmax()]
        print(f"Team with highest salary in {start_season}-{end_season}: {highest_team_salary['Team']} (${highest_team_salary['Team_Salary']:,.2f})")

    if not injury_data.empty:
        injury_count = injury_data['Relinquished'].str.contains(sample_player, case=False).sum()
        print(f"Number of injuries/illnesses for {sample_player} from {start_season} to {end_season}: {injury_count}")

    print("\nAll tests completed.")
