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
)
from salary_nba_data_pull.scrape_utils import (
    scrape_salary_cap_history,
    load_injury_data,
    _season_advanced_df,
)
from salary_nba_data_pull.process_utils import merge_injury_data
from salary_nba_data_pull.data_utils import (
    clean_dataframe,
    merge_salary_cap_data,
    validate_data,
    load_salary_cap_csv,
    load_salary_cap_parquet,
    load_external_salary_data,
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

# ----------------------------------------------------------------------
def update_data(existing_data,
                start_year: int,
                end_year: int,
                *,
                player_filter: str = "all",
                min_avg_minutes: float | None = None,
                debug: bool = False,
                small_debug: bool = False,          # --- NEW
                max_workers: int = 8,
                output_base: str | Path = DATA_PROCESSED_DIR,
                overwrite: bool = False) -> pd.DataFrame:
    """
    Pull seasons in [start_year, end_year] and write under `output_base`.
    When `small_debug` is True, suppress per‑player chatter and show only
    concise per‑season summaries.
    """
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    # Decide low-level debug for helpers
    helper_debug = debug and not small_debug

    injury = load_injury_data(debug=helper_debug)

    # ⇩⇩  NEW  ⇩⇩  pull salary from parquet (or leave empty)
    salary_dir = Path(output_base).parent / "salary_external"
    salary_df = pd.concat(
        [load_external_salary_data(f"{y}-{str(y+1)[-2:]}", root=salary_dir)
         for y in range(start_year, end_year + 1)],
        ignore_index=True
    )

    # if salary not available we'll still proceed
    season_has_salary = set(salary_df["Season"].unique())

    out_frames: list[pd.DataFrame] = []
    season_summaries: list[str] = []  # --- NEW: collect summaries

    for y in tqdm(range(start_year, end_year + 1),
                  desc="Seasons", disable=small_debug):
        season = f"{y}-{str(y+1)[-2:]}"
        ckpt_dir = output_base / f"season={season}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Team payroll (removed - no longer scraped)
        team_payroll = pd.DataFrame(columns=["Team", "Team_Salary", "Season"])

        # --- 2. Standings (wins/losses)
        standings_df = fetch_league_standings(season, debug=helper_debug)
        if standings_df is None:
            standings_df = pd.DataFrame()

        # --- 3. Roster
        players_this_season = fetch_season_players(season, debug=helper_debug)
        rows = salary_df.query("Season == @season") if season in season_has_salary \
               else pd.DataFrame(columns=["Player", "Salary"])
        args = [
            (row.Player, season, row.Salary, players_this_season, helper_debug)
            for _, row in rows.iterrows()
        ] if not rows.empty else [
            (name.title(), season, None, players_this_season, helper_debug)
            for name in players_this_season.keys()
        ]

        # --- pre‑fetch season‑wide advanced table so workers reuse the cache
        _ = _season_advanced_df(season)        # warm cache under the lock

        # --- 4. Player processing in parallel
        with ThreadPoolExecutor(max_workers=min(max_workers or DEFAULT_WORKERS, len(args))) as pool:
            results, failures = [], 0
            for fut in tqdm(as_completed(pool.submit(_player_task, a) for a in args),
                            total=len(args), desc=f"{season} workers", disable=small_debug):
                try:
                    res = fut.result()
                    if res:
                        results.append(res)
                except Exception as exc:
                    failures += 1
                    logging.exception("Worker failed for %s: %s", season, exc)
            if failures and debug:
                print(f"⚠️  {failures} worker threads raised exceptions")

        missing = rows.loc[~rows.Player.str.lower().isin(players_this_season.keys()),
                           "Player"].unique()

        (ckpt_dir / "missing_players.txt").write_text("\n".join(missing))

        df_season = pd.DataFrame(results)
        print(f"[dbg] {season} processed players:", len(df_season))
        
        # ---- PROBE: Check for specific duplicate key ----
        key = ("2023-24", "Kj Martin")
        if season == "2023-24":
            probe_count = df_season.query("Season == @key[0] & Player == @key[1]").shape[0]
            print(f"[probe] Kj Martin count in df_season: {probe_count}")
            if probe_count > 1:
                print("[probe] Kj Martin rows:")
                print(df_season.query("Season == @key[0] & Player == @key[1]")[["Season", "Player", "Team", "MP"]])
        
        # ---------- season sanity check ----------
        if len(df_season) < 150:
            logging.warning("%s produced only %d rows; retrying after 90 s", season, len(df_season))
            time.sleep(90)
            return update_data(existing_data, y, y,  # single‑season retry
                               player_filter=player_filter,
                               min_avg_minutes=min_avg_minutes,
                               debug=debug,
                               small_debug=small_debug,
                               max_workers=max_workers,
                               output_base=output_base,
                               overwrite=True)
        if df_season.empty:
            # Build tiny summary anyway
            season_summaries.append(f"{season}: 0 players processed.")
            continue

        # --- 5. Merge W/L (validate to prevent row blow‑ups)
        if not standings_df.empty:
            stand_df = standings_df.copy()
            if 'W' in stand_df.columns:
                stand_df.rename(columns={'W': 'Wins', 'L': 'Losses'}, inplace=True)
            if 'WINS' in stand_df.columns:
                stand_df.rename(columns={'WINS': 'Wins', 'LOSSES': 'Losses'}, inplace=True)
            if 'TEAM_ID' in stand_df.columns:
                stand_df.rename(columns={'TEAM_ID': 'TeamID'}, inplace=True)
            
            print(f"[dbg] {season} before standings merge:", len(df_season))
            df_season = pd.merge(
                df_season,
                stand_df[['TeamID', 'Wins', 'Losses']].drop_duplicates('TeamID'),
                on='TeamID', how='left', validate='m:1'
            )
            print(f"[dbg] {season} after standings merge:", len(df_season))

        # --- 6. Team payroll merge (removed - no longer merged)
        merged_tmp2 = df_season if min_avg_minutes is None else df_season.query("MP >= @min_avg_minutes")
        print(f"[dbg] {season} after MP filter:", len(merged_tmp2))
        
        merged_tmp3 = merged_tmp2.pipe(merge_injury_data, injury_data=injury)
        print(f"[dbg] {season} after injury merge:", len(merged_tmp3))
        
        merged = (merged_tmp3
                    .pipe(calculate_percentages, debug=helper_debug)
                    # ── deep breath ── add component usage & load
                    .pipe(add_usage_components, debug=helper_debug)
                    .pipe(clean_dataframe))
        
        # ---- FINAL: enforce key uniqueness ----
        dups = merged.duplicated(subset=["Season","Player"], keep=False)
        if dups.any():
            print(f"[dbg] {season} DUPLICATE KEYS detected ({dups.sum()} rows). Dumping...")
            print(merged.loc[dups, ["Season","Player","Team","MP"]]
                        .sort_values(["Player","Team"]))
            # Hard fail so we never persist dirty data:
            raise AssertionError(f"Duplicate (Season,Player) keys in season {season}")

        # STEP A1: deterministic sort & string normalization
        key_cols = ["Season","Player"]
        merged = merged.sort_values(key_cols).reset_index(drop=True)
        obj_cols = merged.select_dtypes(include=["object"]).columns
        for c in obj_cols:
            merged[c] = merged[c].replace(r"^\s*$", pd.NA, regex=True)

        print(f"[dbg] {season} final merged:", len(merged))

        # Skip identical season unless overwrite (moved here to use merged DataFrame)
        if (not overwrite
            and (ckpt_dir / "part.parquet").exists()
            and _season_partition_identical(season, output_base, merged)):
            if debug and not small_debug:
                print(f"✓  {season} unchanged – skipping")
            out_frames.append(merged)
            continue
        elif debug and not small_debug and (ckpt_dir / "part.parquet").exists():
            print(f"↻  {season} differs – re-scraping")

        parquet_path = ckpt_dir / "part.parquet"
        merged.to_parquet(parquet_path, index=False)
        (ckpt_dir / "part.md5").write_text(_file_md5(parquet_path))

        out_frames.append(merged)
        logging.info("wrote %s", ckpt_dir)

        # --- NEW: concise summary
        if small_debug:
            n_players = len(merged)
            n_missing = len(missing)
            n_cols = merged.shape[1]
            season_summaries.append(
                f"{season}: {n_players} rows, {n_missing} missing roster matches, {n_cols} cols."
            )

    # Print all summaries once
    if small_debug and season_summaries:
        print("\n--- Season Summaries ---")
        for line in season_summaries:
            print(line)
        print("------------------------\n")

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
    final_parquet = output_base / "nba_player_data_final_inflated.parquet"
    join_keys = ["Season", "Player"]

    old_master = (pd.read_parquet(final_parquet)
                  if final_parquet.exists() else
                  pd.DataFrame(columns=new_data.columns))

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
         min_avg_minutes: float = 15,
         debug: bool = False,
         small_debug: bool = False,      # --- NEW
         workers: int = 8,
         overwrite: bool = False,
         output_base: str | Path = DATA_PROCESSED_DIR) -> None:
    """
    Entry point. `small_debug=True` prints only high‑signal info.
    If both `debug` and `small_debug` are True, `debug` wins (full noise).
    """
    t0 = time.time()
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

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
                          debug=debug,
                          small_debug=small_debug,          # --- NEW
                          max_workers=workers,
                          output_base=str(output_base),
                          overwrite=overwrite)

    if not small_debug:  # keep your old prints in full/quiet modes
        print(f"✔ Completed pull: {len(updated):,} rows added")

    if not updated.empty:
        # ---------------- Salary Cap -----------------
        # Prefer local Parquet; fallback to CSV, then scrape only if file missing and user allows
        cap_file = Path(output_base) / "salary_cap_history_inflated"
        use_scrape = False

        try:
            salary_cap = load_salary_cap_parquet(cap_file, debug=debug and not small_debug)
        except FileNotFoundError:
            # LAST resort – scrape (can be disabled permanently by setting use_scrape=False)
            if debug and not small_debug:
                print("[salary-cap] local file missing, attempting scrape…")
            salary_cap = scrape_salary_cap_history(debug=debug and not small_debug)
            if salary_cap is not None:
                # Save as both Parquet and CSV for compatibility
                salary_cap.to_parquet(f"{cap_file}.parquet", index=False)
                salary_cap.to_csv(f"{cap_file}.csv", index=False)

        if salary_cap is not None:
            updated = merge_salary_cap_data(updated, salary_cap, debug=debug and not small_debug)
        else:
            if debug:
                print("[salary-cap] No data merged — check local file path.")

        # --------------- Validate --------------------
        updated = validate_data(updated, name="player_dataset", save_reports=True)

        seasons_this_run = sorted(updated["Season"].unique().tolist())
        persist_final_dataset(updated,
                              seasons_loaded=seasons_this_run,
                              output_base=output_base,
                              debug=debug)

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
    p.add_argument("--min_avg_minutes", type=float, default=15)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--small_debug", action="store_true")   # --- NEW
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--output_base",
                   default=str(DATA_PROCESSED_DIR),
                   help="Destination root for parquet + csv outputs")
    args = p.parse_args()
    main(**vars(args))
