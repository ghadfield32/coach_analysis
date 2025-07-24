
import pandas as pd
import numpy as np
from pathlib import Path
from salary_nba_data_pull.process_utils import (
    inflate_value
)
from salary_nba_data_pull.quality import (
    ExpectedSchema, audit_dataframe, write_audit_reports
)
from salary_nba_data_pull.settings import DATA_PROCESSED_DIR

PRESERVE_EVEN_IF_ALL_NA = {
    "3P%", "Injured", "Injury_Periods", "Total_Days_Injured", "Injury_Risk"
}

# --- NEW helper ------------------------------------------------------
def load_salary_cap_parquet(path: str | Path, *, debug: bool = False) -> pd.DataFrame:
    """
    Load the pre‑inflated salary‑cap parquet file; fall back to CSV loader
    if the parquet is not found.
    """
    path = Path(path).expanduser().with_suffix(".parquet")
    if path.exists():
        if debug:
            print(f"[salary-cap] loading Parquet: {path}")
        return pd.read_parquet(path)
    # fallback to old CSV helper for legacy compatibility
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return load_salary_cap_csv(csv_path, debug=debug)
    raise FileNotFoundError(f"No salary‑cap parquet or CSV found at {path}")

def load_salary_cap_csv(path: str | Path, *, debug: bool = False) -> pd.DataFrame:
    """
    Load the preprocessed salary cap CSV (inflated) instead of scraping.
    We DO NOT fill or coerce silently – if a required column is missing,
    we log it and let the caller decide.
    """
    path = Path(path).expanduser().resolve()
    if debug:
        print(f"[salary-cap] loading local file: {path}")
    df = pd.read_csv(path)
    if debug:
        print(f"[salary-cap] rows={len(df)}, cols={df.columns.tolist()}")
    return df

def clean_dataframe(df):
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Remove columns with all NaN values **except** ones we want to keep
    all_na = df.columns[df.isna().all()]
    to_drop = [c for c in all_na if c not in PRESERVE_EVEN_IF_ALL_NA]
    df = df.drop(columns=to_drop)

    # Remove rows with all NaN values
    df = df.dropna(axis=0, how='all')

    # Ensure only one 'Season' column exists
    season_columns = [col for col in df.columns if 'Season' in col]
    if len(season_columns) > 1:
        df = df.rename(columns={season_columns[0]: 'Season'})
        for col in season_columns[1:]:
            df = df.drop(columns=[col])

    # Remove '3PAr' and 'FTr' columns
    columns_to_remove = ['3PAr', 'FTr']
    df = df.drop(columns=columns_to_remove, errors='ignore')

    # Round numeric columns to 2 decimal places
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(2)

    return df

def merge_salary_cap_data(player_data: pd.DataFrame,
                          salary_cap_data: pd.DataFrame,
                          *,
                          debug: bool = False) -> pd.DataFrame:
    """
    Left-merge cap data by season-year. Preserve all cap columns even if all NaN.
    """
    if player_data.empty or salary_cap_data.empty:
        if debug:
            print("[merge_salary_cap_data] one side empty -> returning player_data unchanged")
        return player_data

    # Make sure we don't mutate originals
    p = player_data.copy()
    cap = salary_cap_data.copy()

    # Extract year
    p["Season_Year"]   = p["Season"].str[:4].astype(int)
    cap["Season_Year"] = cap["Season"].str[:4].astype(int)

    # Inflate cap if not present
    if "Salary_Cap_Inflated" not in cap.columns:
        if debug:
            print("[merge_salary_cap_data] computing Salary_Cap_Inflated")
        cap["Salary_Cap_Inflated"] = cap.apply(
            lambda r: inflate_value(r.get("Salary Cap", np.nan), r.get("Season", "")),
            axis=1
        )

    # Merge
    merged = pd.merge(p, cap, on="Season_Year", how="left", suffixes=("", "_cap"))

    # Figure out which columns came from cap
    cap_cols = [c for c in cap.columns if c not in {"Season_Year"}]

    # For each cap col, if we created a *_cap twin, consolidate
    for col in cap_cols:
        src = f"{col}_cap"
        if src in merged.columns:
            merged[col] = merged[col].where(~merged[col].isna(), merged[src])
            merged.drop(columns=[src], inplace=True)

    # Cleanup
    merged.drop(columns=["Season_Year"], inplace=True)

    # Protect salary-cap columns from being dropped in clean_dataframe
    global PRESERVE_EVEN_IF_ALL_NA
    PRESERVE_EVEN_IF_ALL_NA = PRESERVE_EVEN_IF_ALL_NA.union(set(cap_cols))

    merged = clean_dataframe(merged)

    if debug:
        miss = [c for c in cap_cols if c not in merged.columns]
        if miss:
            print(f"[merge_salary_cap_data] WARNING missing cap cols after merge: {miss}")

    return merged

def load_external_salary_data(season: str,
                              root: Path | str = DATA_PROCESSED_DIR / "salary_external",
                              *, debug: bool = False) -> pd.DataFrame:
    """
    Read player‑salary parquet pre‑dropped by an upstream job.
    Expected path:  {root}/season={YYYY-YY}/part.parquet
    """
    path = Path(root) / f"season={season}/part.parquet"
    if not path.exists():
        if debug:
            print(f"[salary‑ext] no salary file at {path}")
        return pd.DataFrame(columns=["Player", "Salary", "Season"])
    if debug:
        print(f"[salary‑ext] loading {path}")
    return pd.read_parquet(path)

def validate_data(df: pd.DataFrame,
                  *,
                  name: str = "player_dataset",
                  save_reports: bool = True) -> pd.DataFrame:
    """
    Same validation, but salary columns are now OPTIONAL.
    """
    schema = ExpectedSchema(
        expected_cols=df.columns,
        required_cols=["Season", "Player", "Team"],   # ‼ Salary removed
        dtypes={
            "Season": "object",
            "Player": "object",
        },
        # Salary & Team_Salary dropped from non‑neg / non‑constant
        non_negative_cols=["GP", "MP", "PTS", "TRB", "AST"],
        non_constant_cols=["PTS"],
        unique_key=["Season", "Player"]
    )

    reports = audit_dataframe(df, schema, name=name)

    if save_reports:
        out_dir = DATA_PROCESSED_DIR / "audits"
        write_audit_reports(reports, out_dir, prefix=name)

    # Print a one-liner summary (optional)
    missing_req = reports["cols_overview"].query("missing_required == True")
    if not missing_req.empty:
        print(f"[validate_data] Missing required columns: {missing_req['column'].tolist()}")

    return df
