from __future__ import annotations
import argparse
import pandas as pd
import logging
import time
import glob
import os
import hashlib
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
import requests_cache
from salary_nba_data_pull.fetch_utils import fetch_all_players, fetch_season_players, fetch_league_standings
from salary_nba_data_pull.process_utils import (
    process_player_data,
    inflate_value,
    calculate_percentages,
    _ensure_cpi_ready,
    add_usage_components,
    consolidate_duplicate_columns,
    ensure_player_ids,
)

    # Removed advanced metrics scraping imports to eliminate nulls
from salary_nba_data_pull.data_utils import (
    clean_dataframe,
    validate_data,
)
from salary_nba_data_pull.settings import DATA_PROCESSED_DIR

# Enable requests-cache for all HTTP traffic
requests_cache.install_cache("nba_pull", backend="sqlite", allowable_codes=(200,))

# CPI self-test - logs a warning once per run if CPI is unavailable
_ensure_cpi_ready(debug=False)

# Default number of worker threads
DEFAULT_WORKERS = 8                # tweak ≤ CPU cores

def _almost_equal_numeric(a: pd.Series, b: pd.Series, atol=1e-6, rtol=1e-9):
    # Handle NA values first
    mask = a.isna() & b.isna()
    
    # For non-NA values, compare them
    both_numeric = pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)
    if not both_numeric:
        # For non-numeric columns, use pandas equals but handle NA carefully
        non_na_mask = ~(a.isna() | b.isna())
        eq_result = pd.Series(False, index=a.index)
        if non_na_mask.any():
            eq_result[non_na_mask] = a[non_na_mask].eq(b[non_na_mask])
        return eq_result | mask
    else:
        # For numeric columns, use numpy isclose
        non_na_mask = ~(a.isna() | b.isna())
        diff_ok = pd.Series(False, index=a.index)
        if non_na_mask.any():
            diff_ok[non_na_mask] = np.isclose(
                a[non_na_mask].astype(float), 
                b[non_na_mask].astype(float), 
                atol=atol, rtol=rtol
            )
        return diff_ok | mask

# helper 1 ─ column drift
def _columns_diff(old_df: pd.DataFrame, new_df: pd.DataFrame):
    added   = sorted(set(new_df.columns) - set(old_df.columns))
    removed = sorted(set(old_df.columns) - set(new_df.columns))
    return added, removed

# helper 2 ─ mean smoke‑test
def _mean_diff(old_df: pd.DataFrame, new_df: pd.DataFrame,
               tol_pct: float = 0.001) -> pd.DataFrame:
    common = old_df.select_dtypes("number").columns.intersection(
             new_df.select_dtypes("number").columns)
    rows = []
    for c in common:
        o, n = old_df[c].mean(skipna=True), new_df[c].mean(skipna=True)
        if pd.isna(o) or pd.isna(n):
            continue
        rel = abs(n - o) / (abs(o) + 1e-12)
        if rel > tol_pct:
            rows.append({"column": c, "old_mean": o, "new_mean": n, "rel_diff": rel})
    return pd.DataFrame(rows)

def _diff_report(old_df, new_df, key_cols=("Season","Player"),
                 numeric_atol=1e-6, numeric_rtol=1e-9, max_print=10):
    cols_add, cols_rem = _columns_diff(old_df, new_df)
    mean_diffs = _mean_diff(old_df, new_df)

    # value‑level diff (original logic)
    common = [c for c in new_df.columns if c in old_df.columns]
    old, new = old_df[common], new_df[common]

    # Handle case where dataframes have different lengths
    if len(old) != len(new):
        # If lengths differ, we can't do row-by-row comparison
        diffs = []
    else:
        if all(k in common for k in key_cols):
            old = old.sort_values(list(key_cols)).reset_index(drop=True)
            new = new.sort_values(list(key_cols)).reset_index(drop=True)
        else:
            key_cols = ("__row__",)
            old["__row__"] = new["__row__"] = range(len(old))

        diffs = []
        for col in common:
            equal = _almost_equal_numeric(old[col], new[col],
                                          atol=numeric_atol, rtol=numeric_rtol)
            for i in np.where(~equal)[0]:
                if i < len(old) and i < len(new):  # Safety check
                    row_keys = {k: new.iloc[i][k] for k in key_cols}
                    diffs.append({**row_keys, "column": col,
                                  "old": old.iloc[i][col], "new": new.iloc[i][col]})

    is_equal = (not diffs) and (not cols_add) and (not cols_rem) and mean_diffs.empty
    summary = (f"cells:{len(diffs)}  col+:{len(cols_add)}  col-:{len(cols_rem)}  "
               f"meanΔ:{len(mean_diffs)}")
    return is_equal, summary, pd.DataFrame(diffs), cols_add, cols_rem, mean_diffs

def _file_md5(path: str, chunk: int = 1 << 20) -> str:
    """Return md5 hexdigest for *path* streaming in 1 MiB chunks."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()

def _season_partition_identical(season: str,
                                base_dir: Path | str,
                                new_df: pd.DataFrame) -> bool:
    """
    Return True if on-disk parquet for `season` is byte-wise equivalent (after
    canonical sort & column alignment) to `new_df`.
    """
    ckpt = Path(base_dir) / f"season={season}" / "part.parquet"
    if not ckpt.exists():
        return False

    try:
        old_df = pd.read_parquet(ckpt)
    except Exception as exc:
        logging.warning("[identical] failed to read %s → %s", ckpt, exc)
        return False

    # STEP B1: align columns and sort only by stable key
    cols = sorted(set(old_df.columns) | set(new_df.columns))
    key = ["Season","Player"]

    old_cmp = (old_df.reindex(columns=cols)
                     .sort_values(key)
                     .reset_index(drop=True))
    new_cmp = (new_df.reindex(columns=cols)
                     .sort_values(key)
                     .reset_index(drop=True))

    return old_cmp.equals(new_cmp)   # NaNs treated equal if aligned

def _season_partition_exists(season, base_dir):
    """Check if a season partition already exists in Parquet format."""
    return os.path.exists(os.path.join(base_dir, f"season={season}"))

def _player_task(args):
    """Wrapper for ThreadPoolExecutor."""
    (player_name, season, salary, all_players, debug) = args
    stats = process_player_data(player_name, season, all_players, debug=debug)
    if stats:
        stats['Salary'] = salary
    return stats



import pandas as pd
import logging, textwrap

CORE_COLS = ("FGA", "FTA", "MP", "PTS")

def debug_checkpoint(df: pd.DataFrame,
                     label: str,
                     *,
                     core_cols: tuple[str, ...] = CORE_COLS,
                     head: int = 0) -> None:
    """
    Print a compact overview of the DataFrame at a pipeline milestone.

    • Always shows #rows, #cols.
    • Warns if any `core_cols` are missing.
    • Optionally prints `df.head(head)` for a quick sanity scan.
    """
    msg = f"[chk:{label}] rows={len(df):,}  cols={len(df.columns):,}"
    missing = [c for c in core_cols if c not in df.columns]
    if missing:
        msg += f"  ❌ MISSING: {missing}"
    logging.debug(msg)
    print(msg)                     # visible even without logging configured
    if head > 0:
        print(textwrap.indent(df.head(head).to_string(index=False), "    "))

# ----------------------------------------------------------------------
def update_data(existing_data,
                start_year: int,
                end_year: int,
                *,
                player_filter: str = "all",
                min_avg_minutes: float | None = None,    # NEW: filter on avg minutes
                min_shot_attempts: int | None = None,    # NEW: filter on shot attempts
                nan_filter: bool = False,                 # NEW: enable threshold-aware NaN filtering
                nan_filter_percentage: float = 0.01,      # NEW: threshold for low-missing columns
                debug: bool = False,
                small_debug: bool = False,
                max_workers: int = 8,
                output_base: str | Path = DATA_PROCESSED_DIR,
                overwrite: bool = False) -> pd.DataFrame:
    """
    Pull seasons in [start_year, end_year], WITHOUT any salary or injury merges.
    Ensures we only rely on nba_api rosters + career stats + W/L logs.
    
    FILTERS:
    - min_avg_minutes: Filter out players averaging < this many minutes per game
    - min_shot_attempts: Filter out players with fewer than this many total shot attempts (FGA+FTA)
    - nan_filter: If True, apply threshold-aware NaN filtering instead of dropping all rows with any NaN
    - nan_filter_percentage: Threshold for low-missing columns when nan_filter=True (default 1%)
    
    These filters help eliminate nulls from low-volume players who don't have enough
    data for meaningful percentage calculations.
    """
    output_base = Path(output_base)
    helper_debug = debug and not small_debug

    from salary_nba_data_pull.scrape_utils import _season_advanced_df
    from salary_nba_data_pull.fetch_utils import (
        fetch_season_players, fetch_league_standings, fetch_team_wl_by_season
    )
    from salary_nba_data_pull.process_utils import (
        process_player_data, calculate_percentages, add_usage_components,
        attach_wins_losses, merge_advanced_metrics,
    )

    out_frames: list[pd.DataFrame] = []
    season_summaries: list[str] = []

    for y in tqdm(range(start_year, end_year + 1),
                  desc="Seasons", disable=small_debug):
        season = f"{y}-{str(y+1)[-2:]}"
        ckpt_dir = output_base / f"season={season}"

        if helper_debug:
            print(f"[update_data] Starting season {season}")

        # 1️⃣ Fetch the complete season roster
        roster = fetch_season_players(season, debug=helper_debug)
        if helper_debug:
            print(f"[update_data] fetched {len(roster)} players for {season}")

        # 2️⃣ Build args for each player (correct signature)
        args = [
            (name, season, roster, helper_debug)
            for name in roster.keys()
            if (player_filter == "all" or player_filter.lower() in name)
        ]
        if helper_debug:
            print(f"[update_data] processing {len(args)} players after filter")

        # 3️⃣ Process each player in parallel (correct signature)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results, failures = [], 0
        with ThreadPoolExecutor(max_workers=min(max_workers, len(args) or 1)) as pool:
            futures = {pool.submit(
                lambda nm, ss, rp, dbg: process_player_data(nm, ss, rp, debug=dbg),
                *arg
            ): arg[0] for arg in args}

            for fut in as_completed(futures):
                pname = futures[fut]
                try:
                    res = fut.result()
                    if res is None:
                        if helper_debug:
                            print(f"[update_data][WARN] no data for player '{pname}' in {season}")
                    else:
                        results.append(res)
                except Exception as exc:
                    failures += 1
                    logging.exception("Player task failed for %s (%s): %s", pname, season, exc)

        if failures and debug:
            print(f"[update_data] ⚠️  {failures} player failures in {season}")

        df_season = pd.DataFrame(results)
        
        # NEW: repair legacy partitions that missed PlayerID / TeamID
        df_season = ensure_player_ids(df_season, season, debug=helper_debug)
        
        if helper_debug:
            print(f"[update_data] {season} → DataFrame with {len(df_season)} rows")

        # ------------------------------------------------------------------
        # A) **EARLY minutes-per-game filter**
        # ------------------------------------------------------------------
        if (min_avg_minutes is not None) and ("MP" in df_season.columns):
            before = len(df_season)
            df_season = df_season.query("MP >= @min_avg_minutes")
            if helper_debug:
                print(f"[filter-early] {season}: MP ≥ {min_avg_minutes}  "
                      f"→ {before}→{len(df_season)} rows")
        # ------------------------------------------------------------------

        # 4️⃣ Attach W/L using unified lookup
        df_season = df_season.pipe(
            attach_wins_losses, season=season, debug=helper_debug
        )

        # 5️⃣ Derived metrics & clean
        if helper_debug:
            print(f"[update_data] {season} before derived metrics: {len(df_season.columns)} columns")
            print(f"[update_data] {season} columns: {list(df_season.columns)}")
        
        merged = (
            df_season
            .pipe(calculate_percentages, debug=helper_debug)
            .pipe(add_usage_components, debug=helper_debug)
            .pipe(merge_advanced_metrics, season=season, debug=helper_debug)  # Re-enabled advanced metrics
            .pipe(consolidate_duplicate_columns, debug=helper_debug)
        )
        
        # -------------------------------------------------------------
        # 🔒 Debug sentinel – verify PlayerID survival after consolidation
        # -------------------------------------------------------------
        debug_checkpoint(merged, f"{season}:post-consolidate", head=0)
        assert "PlayerID" in merged.columns, f"[update_data] {season}: PlayerID lost after consolidate_duplicate_columns"
        
        debug_checkpoint(merged, f"{season}:post-derived", head=3)
        
        
        # --- season-aware guard & diagnostics (no filling) ---
        if helper_debug:
            from salary_nba_data_pull.process_utils import (
                guard_advanced_null_regress, diagnose_advanced_nulls
            )
            guard_advanced_null_regress(merged, season, base_dir=output_base, debug=True)
            _ = diagnose_advanced_nulls(merged, season, debug=True)
        
        if helper_debug:
            print(f"[update_data] {season} after derived metrics: {len(merged.columns)} columns")
            advanced_cols = ['PER', 'BPM', 'VORP', 'WS', 'DWS', 'OWS', 'WS/48', 'AST%', 'BLK%', 'TOV%', 'TRB%', 'DRB%']
            found_advanced = [col for col in advanced_cols if col in merged.columns]
            print(f"[update_data] {season} advanced columns found: {found_advanced}")

        # ── NEW: apply user‐specified filters ─────────────────────────
        # 5️⃣ Filter low-minute players  (robust to missing column)
        if min_avg_minutes is not None:
            if "MP" in merged.columns:                    # keep the guarded fallback
                before = len(merged)
                merged = merged.query("MP >= @min_avg_minutes")
                if helper_debug:
                    print(f"[filter-late] {season}: MP ≥ {min_avg_minutes}  "
                          f"→ {before}→{len(merged)} rows")
            else:
                logging.warning("[filter-late] %s: 'MP' col missing – skipped", season)
        #  ^-- Only this guarded block remains.  **The stray unconditional query is gone.**
        # ── NEW: shot-attempt filter with robust column handling ────────────
        if min_shot_attempts is not None:
            # We accept either 'FGA' directly *or* twins created by merges.
            for _candidate in ("FGA", "FGA_x", "FGA_y"):
                if _candidate in merged.columns:
                    fga_col = _candidate
                    break
            else:   # no break → not found
                raise KeyError(
                    "[filter-shots] 'FGA' column missing after merges – "
                    "check consolidate_duplicate_columns or earlier transforms."
                )

            fta_col = "FTA" if "FTA" in merged.columns else \
                      "FTA_x" if "FTA_x" in merged.columns else \
                      "FTA_y" if "FTA_y" in merged.columns else None

            if fta_col is None:
                raise KeyError("[filter-shots] 'FTA' column missing after merges.")

            before = len(merged)
            merged = (
                merged
                .assign(_shots=merged[fga_col].fillna(0) + merged[fta_col].fillna(0))
                .query("_shots >= @min_shot_attempts")
                .drop(columns=["_shots"])
            )
            if helper_debug:
                print(f"[filter-shots] {season}: ≥{min_shot_attempts} attempts "
                      f"→ {before}→{len(merged)} rows")

        # ── NEW: apply NaN filtering ───────────────────────────────────
        if nan_filter:
            # Apply threshold-aware NaN filtering
            before = len(merged)
            null_pct = merged.isna().mean()
            cols_to_strict_drop = null_pct[(null_pct > 0) & (null_pct <= nan_filter_percentage)].index.tolist()
            
            if cols_to_strict_drop:
                merged = merged.dropna(subset=cols_to_strict_drop)
                dropped = before - len(merged)
                if helper_debug:
                    print(f"[nan_filter] {season}: dropped {dropped} rows based on "
                          f"{len(cols_to_strict_drop)} low-missing columns (≤ {nan_filter_percentage*100:.1f}%)")
                    print(f"[nan_filter] {season}: low-missing columns: {cols_to_strict_drop}")
            else:
                if helper_debug:
                    print(f"[nan_filter] {season}: no columns below threshold; no rows dropped")
        else:
            # Legacy behavior: drop any row with any NaN
            before = len(merged)
            merged = merged.dropna()
            if helper_debug:
                print(f"[nan_filter] {season}: legacy dropna - dropped {before - len(merged)} rows with any NaN")
        # ── end filters ──────────────────────────────────────────────────

        # Key‐column sanity
        dups = merged.duplicated(subset=["Season","Player"], keep=False)
        if dups.any():
            sample = merged.loc[dups, ["Season","Player","Team","MP"]]
            print(f"[update_data][ERROR] Duplicate keys in {season}:\n{sample}")
            raise AssertionError(f"Duplicate (Season,Player) in {season}")

        # Trim whitespace-only strings → NA
        obj_cols = merged.select_dtypes(include=["object"]).columns
        for c in obj_cols:
            merged[c] = merged[c].replace(r"^\s*$", pd.NA, regex=True)

        # Persist per‐season partition
        parquet_path = ckpt_dir / "part.parquet"
        if not overwrite and parquet_path.exists():
            from salary_nba_data_pull.main import _season_partition_identical
            if _season_partition_identical(season, output_base, merged):
                if helper_debug:
                    print(f"[update_data] {season} unchanged, skipping write")
                out_frames.append(merged)
                continue
        merged.to_parquet(parquet_path, index=False)
        if helper_debug:
            print(f"[update_data] wrote {parquet_path}")

        out_frames.append(merged)
        season_summaries.append(f"{season}: {len(merged)} rows")

    if small_debug:
        print("\n--- Seasons Summaries ---")
        print("\n".join(season_summaries))
        print("-------------------------\n")

    return pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()

def get_timestamp():
    """Return a filesystem-safe timestamp string."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def remove_old_logs(log_dir, days_to_keep=7):
    current_time = datetime.now()
    for log_file in glob.glob(os.path.join(log_dir, 'stat_pull_log_*.txt')):
        file_modified_time = datetime.fromtimestamp(os.path.getmtime(log_file))
        if current_time - file_modified_time > timedelta(days=days_to_keep):
            os.remove(log_file)

def persist_final_dataset(new_data: pd.DataFrame, seasons_loaded: list[str],
                          *, output_base: Path, debug: bool = False,
                          numeric_atol: float = 1e-6, numeric_rtol: float = 1e-9,
                          max_print: int = 15, mean_tol_pct: float = 0.001) -> None:
    from salary_nba_data_pull.data_utils import prune_end_columns

    final_parquet = output_base / "nba_player_data_final_inflated.parquet"
    join_keys = ["Season", "Player"]

    # -- NEW: prune end-only columns BEFORE diffing/writing
    new_data = prune_end_columns(new_data, debug=debug)

    old_master = (pd.read_parquet(final_parquet)
                  if final_parquet.exists() else
                  pd.DataFrame(columns=new_data.columns))

    # -- NEW: also prune any legacy columns in the old master for a fair diff
    if not old_master.empty:
        old_master = prune_end_columns(old_master, debug=debug)

    for df in (old_master, new_data):
        for k in join_keys:
            if k in df.columns:
                df[k] = df[k].astype(str).str.strip()

    old_slice = old_master.merge(
        pd.DataFrame({"Season": seasons_loaded}).drop_duplicates(),
        on="Season", how="inner").reset_index(drop=True)
    new_slice = new_data.reset_index(drop=True)

    equal, summary, diff_cells, cols_add, cols_rem, mean_diffs = \
        _diff_report(old_slice, new_slice, key_cols=join_keys,
                     numeric_atol=numeric_atol, numeric_rtol=numeric_rtol)

    # Special case: if old_slice is empty but new_slice has data, we should write
    if len(old_slice) == 0 and len(new_slice) > 0:
        equal = False
        if debug:
            print("[persist] Creating new master parquet with fresh data")

    if equal:
        if debug:
            print("[persist] No changes detected – master Parquet left untouched")
        return

    audits = output_base / "audits"; audits.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if cols_add or cols_rem:
        pd.DataFrame({"added": [cols_add], "removed": [cols_rem]}
                     ).to_csv(audits / f"column_changes_{ts}.csv", index=False)
    if not mean_diffs.empty:
        mean_diffs.to_csv(audits / f"mean_diffs_{ts}.csv", index=False)
    if not diff_cells.empty:
        diff_cells.to_csv(audits / f"value_diffs_{ts}.csv", index=False)

    # ----- rewrite master -----
    union_cols = sorted(set(old_master.columns) | set(new_data.columns))
    remover = old_master.merge(
        pd.DataFrame({"Season": seasons_loaded}), on="Season",
        how="left", indicator=True)
    remover = remover[remover["_merge"] == "left_only"].drop(columns="_merge")
    remover = remover.reindex(columns=union_cols)
    new_slice = new_slice.reindex(columns=union_cols)

    updated_master = pd.concat([remover, new_slice], ignore_index=True)\
                       .sort_values(join_keys).reset_index(drop=True)
    updated_master.to_parquet(final_parquet, index=False)
    if debug: print(f"[persist] Master Parquet updated – {summary}")

def main(start_year: int,
         end_year: int,
         player_filter: str = "all",
         min_avg_minutes: float = 10,    # NEW default: 10 minutes
         min_shot_attempts: int = 50,    # NEW: filter on shot attempts
         nan_filter: bool = False,       # NEW: enable threshold-aware NaN filtering
         nan_filter_percentage: float = 0.01,  # NEW: threshold for low-missing columns
         debug: bool = False,
         small_debug: bool = False,      # --- NEW
         workers: int = 8,
         overwrite: bool = False,
         output_base: str | Path = DATA_PROCESSED_DIR) -> None:
    """
    Entry point for NBA data processing pipeline.
    
    NaN Filtering Options:
    - nan_filter=False (default): Legacy behavior - drop any row with any NaN
    - nan_filter=True: Threshold-aware filtering - only drop rows for columns with 
      NaN rate ≤ nan_filter_percentage
    
    Debug Options:
    - small_debug=True: Print only high-signal info
    - debug=True: Full verbose output
    - If both debug and small_debug are True, debug wins (full noise)
    """
    t0 = time.time()
    output_base = Path(output_base)


    log_dir = output_base.parent / "stat_pull_output"
    log_dir.mkdir(parents=True, exist_ok=True)
    remove_old_logs(log_dir)

    log_file = log_dir / f"stat_pull_log_{get_timestamp()}.txt"
    logging.basicConfig(filename=log_file,
                        level=logging.DEBUG if debug else logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    updated = update_data(None, start_year, end_year,
                          player_filter=player_filter,
                          min_avg_minutes=min_avg_minutes,
                          min_shot_attempts=min_shot_attempts,  # NEW: pass shot attempts filter
                          nan_filter=nan_filter,              # NEW: pass NaN filter flag
                          nan_filter_percentage=nan_filter_percentage,  # NEW: pass NaN filter threshold
                          debug=debug,
                          small_debug=small_debug,          # --- NEW
                          max_workers=workers,
                          output_base=str(output_base),
                          overwrite=overwrite)

    if not small_debug:  # keep your old prints in full/quiet modes
        print(f"✔ Completed pull: {len(updated):,} rows added")

    if not updated.empty:
        # — Skip salary‐cap entirely —
        # Validate only core columns (Season,Player,Team)
        from salary_nba_data_pull.data_utils import validate_data
        updated = validate_data(updated, name="player_dataset", save_reports=True)

        # Persist master
        seasons_this_run = sorted(updated["Season"].unique().tolist())
        persist_final_dataset(
            updated,
            seasons_loaded=seasons_this_run,
            output_base=output_base,
            debug=debug
        )

    if not small_debug:
        print(f"Process finished in {time.time() - t0:.1f} s — log: {log_file}")
    else:
        # minimal closing line
        print(f"Done in {time.time() - t0:.1f}s. Log: {log_file}")
        
# ----------------------------------------------------------------------
# argparse snippet
if __name__ == "__main__":
    cur = datetime.now().year
    p = argparse.ArgumentParser()
    p.add_argument("--start_year", type=int, default=cur-1)
    p.add_argument("--end_year",   type=int, default=cur)
    p.add_argument("--player_filter", default="all")
    p.add_argument("--min_avg_minutes", type=float, default=10,
                   help="Filter out players averaging < this many minutes per game")
    p.add_argument("--min_shot_attempts", type=int, default=50,
                   help="Filter out players with fewer than this many total shot attempts (FGA+FTA)")
    p.add_argument("--nan_filter", action="store_true",
                   help="Enable threshold-aware NaN filtering (instead of dropping all rows with any NaN)")
    p.add_argument("--nan_filter_percentage", type=float, default=0.01,
                   help="Threshold for low-missing columns when nan_filter=True (default 1%%)")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--small_debug", action="store_true")   # --- NEW
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--output_base",
                   default=str(DATA_PROCESSED_DIR),
                   help="Destination root for parquet + csv outputs")
    args = p.parse_args()
    main(**vars(args))

