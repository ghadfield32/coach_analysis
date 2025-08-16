"""
Notebook/REPL helper utilities for salary_nba_data_pull.

Goals
-----
• Work no matter where the notebook is opened (absolute paths).
• Avoid NameError on __file__.
• Keep hot‑reload for iterative dev.
• Forward arbitrary args to main() so we can test all scenarios.
• Support NaN filtering with configurable thresholds.

Use:
>>> import salary_nba_data_pull.notebook_helper as nb
>>> nb.quick_pull(2024, workers=12, debug=True)
>>> nb.quick_pull(2024, nan_filter=True, nan_filter_percentage=0.02)
"""

from __future__ import annotations
import sys, importlib, inspect, os
from pathlib import Path
import requests_cache
from typing import Iterable
import pandas as pd

def _find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root by looking for pyproject.toml or .git."""
    markers = {"pyproject.toml", ".git"}
    here = (start or Path.cwd()).resolve()
    for p in [here] + list(here.parents):
        if any((p / m).exists() for m in markers):
            return p
    return here

# Ensure project root & src are on sys.path (defensive)
ROOT = _find_repo_root()
SRC  = ROOT / "src"
for p in (ROOT, SRC):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Sanity print (can be silenced)
if __name__ == "__main__" or "JPY_PARENT_PID" in os.environ:
    print(f"[notebook_helper] sys.path[0:3]={sys.path[:3]}")

# Import after path fix
try:
    from salary_nba_data_pull import main as nba_main
    from salary_nba_data_pull.settings import DATA_PROCESSED_DIR
    from salary_nba_data_pull.fetch_utils import clear_cache as _cc
    print("✅ salary_nba_data_pull imported successfully")
except ImportError as e:
    print(f"❌ Failed to import salary_nba_data_pull: {e}")
    print(f"   ROOT={ROOT}")
    print(f"   SRC={SRC}")
    print(f"   sys.path[0:3]={sys.path[:3]}")
    raise
    
    
def _reload():
    """Reload the main module so code edits are picked up."""
    importlib.reload(nba_main)

def quick_pull(season: int, **kwargs):
    """
    Pull data for a single season with optional NaN filtering.
    
    Args:
        season: Year to pull (e.g., 2024 for 2024-25 season)
        **kwargs: Additional arguments passed to main()
        
    NaN Filtering:
        nan_filter: If True, use threshold-aware NaN filtering (default: False)
        nan_filter_percentage: Threshold for low-missing columns (default: 0.01 = 1%)
        
    Examples:
        >>> quick_pull(2024, debug=True)  # Legacy behavior
        >>> quick_pull(2024, nan_filter=True, nan_filter_percentage=0.02)  # 2% threshold
    """
    _reload()
    # Explicitly support nan_filter and its threshold:
    nan_filter = kwargs.pop("nan_filter", False)
    nan_filter_percentage = kwargs.pop("nan_filter_percentage", 0.01)
    print(f"[quick_pull] season={season}  nan_filter={nan_filter} "
          f"nan_filter_percentage={nan_filter_percentage}  other_kwargs={kwargs}")
    nba_main.main(
        start_year=season,
        end_year=season,
        nan_filter=nan_filter,
        nan_filter_percentage=nan_filter_percentage,
        **kwargs
    )

def historical_pull(start_year: int, end_year: int, **kwargs):
    """
    Pull data for multiple seasons with optional NaN filtering.
    
    Args:
        start_year: First year to pull (inclusive)
        end_year: Last year to pull (inclusive)
        **kwargs: Additional arguments passed to main()
        
    NaN Filtering:
        nan_filter: If True, use threshold-aware NaN filtering (default: False)
        nan_filter_percentage: Threshold for low-missing columns (default: 0.01 = 1%)
        
    Examples:
        >>> historical_pull(2022, 2024, debug=True)  # Legacy behavior
        >>> historical_pull(2022, 2024, nan_filter=True, nan_filter_percentage=0.02)  # 2% threshold
    """
    _reload()
    # Explicitly support nan_filter and its threshold:
    nan_filter = kwargs.pop("nan_filter", False)
    nan_filter_percentage = kwargs.pop("nan_filter_percentage", 0.01)
    print(f"[historical_pull] {start_year}-{end_year}  nan_filter={nan_filter} "
          f"nan_filter_percentage={nan_filter_percentage}  other_kwargs={kwargs}")
    nba_main.main(
        start_year=start_year,
        end_year=end_year,
        nan_filter=nan_filter,
        nan_filter_percentage=nan_filter_percentage,
        **kwargs
    )

def check_existing_data(base: Path | str | None = None) -> list[str]:
    base = Path(base) if base else DATA_PROCESSED_DIR
    seasons = sorted(d.name.split("=", 1)[-1] for d in base.glob("season=*") if d.is_dir())
    print(f"[check_existing_data] found {len(seasons)} seasons in {base}")
    return seasons

def load_parquet_data(season: str | None = None, *, base: Path | str | None = None):
    import pandas as pd
    base = Path(base) if base else DATA_PROCESSED_DIR
    files = list(base.glob(f"season={season}/part.parquet")) if season else list(base.glob("season=*/part.parquet"))
    if not files:
        print("[load_parquet_data] No parquet files found.")
        return pd.DataFrame()
    print(f"[load_parquet_data] loading {len(files)} files from {base}")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)

def clear_all_caches():
    requests_cache.clear()
    _cc()
    print("✅ caches cleared")

def print_args():
    sig = inspect.signature(nba_main.main)
    for name, param in sig.parameters.items():
        print(f"{name:<15} default={param.default!r}  kind={param.kind}")

def query_data(sql: str, db: str | None = None):
    """
    Run arbitrary SQL against the DuckDB lake. Example:
        query_data("SELECT COUNT(*) FROM parquet_scan('data/new_processed/season=*/part.parquet')")
    """
    import duckdb, pandas as pd
    db = db or (DATA_PROCESSED_DIR.parent / "nba_stats.duckdb")
    with duckdb.connect(str(db), read_only=True) as con:
        return con.execute(sql).fetchdf()


# ── NEW VALIDATORS ──────────────────────────────────────────────────────────

def validate_season_coverage(df: pd.DataFrame,
                             expected_seasons: list[str]) -> None:
    """
    Check that df['Season'] covers exactly the expected seasons.
    Prints missing and extra seasons.
    """
    if "Season" not in df.columns:
        print("[validate_season_coverage] ERROR: no 'Season' column")
        return

    actual = sorted(df["Season"].dropna().unique().tolist())
    missing = [s for s in expected_seasons if s not in actual]
    extra   = [s for s in actual if s not in expected_seasons]

    print(f"[validate_season_coverage] expected: {expected_seasons}")
    print(f"[validate_season_coverage] actual:   {actual}")
    if missing:
        print(f"[validate_season_coverage] MISSING seasons: {missing}")
    if extra:
        print(f"[validate_season_coverage] EXTRA seasons:   {extra}")
    if not missing and not extra:
        print("[validate_season_coverage] ✅ coverage OK")

def report_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise null counts & percentages for each column in df.
    Returns a DataFrame with columns: column, null_count, total_rows, null_pct.
    """
    total = len(df)
    stats = []
    for col in df.columns:
        nulls = int(df[col].isna().sum())
        pct   = 100 * nulls / total if total else 0
        stats.append({
            "column": col,
            "null_count": nulls,
            "total_rows": total,
            "null_pct": round(pct, 2)
        })
    report = pd.DataFrame(stats).sort_values("null_pct", ascending=False)
    print("[report_nulls]")
    print(report.to_string(index=False))
    return report



if __name__ == "__main__":
    print_args()
    # quick_pull(2023, workers=4, debug=True)

    historical_pull(2023, 2024,        # multi‑season, 2012, 2024,
                    workers=6,
                    min_avg_minutes=10,
                    min_shot_attempts=50,
                    overwrite=True,
                    debug=True)
    check_existing_data()              # see which seasons are cached
    df = load_parquet_data("2023-24")  # inspect a single season

    # Suppose you want exactly that one season:
    validate_season_coverage(df, ["2023-24"])
    # Check nulls:
    null_report = report_nulls(df)
    # Examine the top 5 columns by null_pct
    null_report.head()
