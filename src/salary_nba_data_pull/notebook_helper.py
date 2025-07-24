"""
Notebook/REPL helper utilities for salary_nba_data_pull.

Goals
-----
• Work no matter where the notebook is opened (absolute paths).
• Avoid NameError on __file__.
• Keep hot‑reload for iterative dev.
• Forward arbitrary args to main() so we can test all scenarios.

Use:
>>> import salary_nba_data_pull.notebook_helper as nb
>>> nb.quick_pull(2024, workers=12, debug=True)
"""

from __future__ import annotations
import sys, importlib, inspect, os
from pathlib import Path
import requests_cache
from typing import Iterable

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
    _reload()
    print(f"[quick_pull] season={season}, kwargs={kwargs}")
    nba_main.main(start_year=season, end_year=season, **kwargs)

def historical_pull(start_year: int, end_year: int, **kwargs):
    _reload()
    print(f"[historical_pull] {start_year}-{end_year}, kwargs={kwargs}")
    nba_main.main(start_year=start_year, end_year=end_year, **kwargs)

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


if __name__ == "__main__":
    print_args()
    # quick_pull(2023, workers=4, debug=True)



    historical_pull(2012, 2024,        # multi‑season, 2012, 2024,
                    workers=6,
                    min_avg_minutes=10,
                    overwrite=True,
                    debug=True)
    check_existing_data()              # see which seasons are cached
    df = load_parquet_data("2023-24")  # inspect a single season
