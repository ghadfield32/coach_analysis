# src/salary_nba_data_pull/quality.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Any
import pandas as pd
import numpy as np

@dataclass
class ExpectedSchema:
    """Describe what we *intended* to have in a dataframe."""
    # All columns we care about (order doesn't matter)
    expected_cols: Iterable[str]

    # Subset that must be present
    required_cols: Iterable[str] = field(default_factory=list)

    # Expected pandas dtypes (string form, e.g. 'float64', 'object')
    dtypes: Mapping[str, str] = field(default_factory=dict)

    # Columns that must be >= 0
    non_negative_cols: Iterable[str] = field(default_factory=list)

    # Columns that should not be all zeros / all NaN
    non_constant_cols: Iterable[str] = field(default_factory=list)

    # Unique key columns (together must be unique)
    unique_key: Iterable[str] = field(default_factory=list)

    # Allowed value sets (enums)
    allowed_values: Mapping[str, Iterable[Any]] = field(default_factory=dict)

def _series_is_constant(s: pd.Series) -> bool:
    return s.nunique(dropna=True) <= 1

def audit_dataframe(df: pd.DataFrame,
                    schema: ExpectedSchema,
                    *,
                    name: str = "dataset") -> dict[str, pd.DataFrame]:
    """
    Return a dict of small DataFrames summarising quality checks.
    Nothing is printed; caller decides how to persist/log.
    """
    exp = set(schema.expected_cols)
    req = set(schema.required_cols)

    present = set(df.columns)
    missing = sorted(list(exp - present))
    extra   = sorted(list(present - exp))

    # --- Column overview
    cols_overview = pd.DataFrame({
        "column": sorted(list(exp | present)),
        "expected": [c in exp for c in sorted(list(exp | present))],
        "present":  [c in present for c in sorted(list(exp | present))],
        "required": [c in req for c in sorted(list(exp | present))]
    })
    cols_overview["missing_required"] = cols_overview.apply(
        lambda r: r["required"] and not r["present"], axis=1
    )

    # --- Null report
    null_report = (df.isna().sum().to_frame("null_count")
                     .assign(total_rows=len(df))
                     .assign(null_pct=lambda d: 100 * d["null_count"] / d["total_rows"])
                     .reset_index()
                     .rename(columns={"index": "column"}))

    # --- Dtype report
    type_rows = []
    for col in df.columns:
        exp_type = schema.dtypes.get(col)
        type_rows.append({
            "column": col,
            "expected_dtype": exp_type,
            "actual_dtype": str(df[col].dtype),
            "matches": (exp_type is None) or (str(df[col].dtype) == exp_type)
        })
    type_report = pd.DataFrame(type_rows)

    # --- Value checks
    value_rows = []
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col]
        row = {
            "column": col,
            "min": series.min(skipna=True),
            "max": series.max(skipna=True),
            "negatives": int((series < 0).sum()),
            "zeros": int((series == 0).sum()),
            "non_zero_pct": 100 * (series != 0).sum() / len(series),
        }
        row["should_be_non_negative"] = col in schema.non_negative_cols
        row["violates_non_negative"] = row["negatives"] > 0 and row["should_be_non_negative"]
        value_rows.append(row)
    value_report = pd.DataFrame(value_rows)

    # Constant columns
    constant_rows = []
    for col in df.columns:
        constant_rows.append({
            "column": col,
            "is_constant": _series_is_constant(df[col]),
            "should_not_be_constant": col in schema.non_constant_cols
        })
    constant_report = pd.DataFrame(constant_rows).assign(
        violates=lambda d: d["is_constant"] & d["should_not_be_constant"]
    )

    # Allowed values
    enum_rows = []
    for col, allowed in schema.allowed_values.items():
        if col not in df.columns:
            continue
        bad = ~df[col].isin(allowed) & df[col].notna()
        enum_rows.append({
            "column": col,
            "bad_count": int(bad.sum()),
            "sample_bad": df.loc[bad, col].drop_duplicates().head(5).tolist()
        })
    enum_report = pd.DataFrame(enum_rows)

    # Unique key
    uniq_report = pd.DataFrame()
    if schema.unique_key:
        dup_mask = df.duplicated(subset=list(schema.unique_key), keep=False)
        uniq_report = pd.DataFrame({
            "duplicate_rows": [int(dup_mask.sum())],
            "subset": [list(schema.unique_key)]
        })

    return {
        "cols_overview": cols_overview,
        "null_report": null_report,
        "type_report": type_report,
        "value_report": value_report,
        "constant_report": constant_report,
        "enum_report": enum_report,
        "unique_report": uniq_report
    }

def assert_dataframe_ok(df: pd.DataFrame,
                        schema: ExpectedSchema,
                        *, name: str = "dataset") -> None:
    """
    Raise AssertionError with a concise message if critical checks fail.
    Designed for pytest or CI.
    """
    rep = audit_dataframe(df, schema, name=name)
    bad_missing = rep["cols_overview"].query("missing_required == True")
    bad_types = rep["type_report"].query("matches == False")
    bad_nonneg = rep["value_report"].query("violates_non_negative == True")
    bad_constant = rep["constant_report"].query("violates == True")
    dupes = rep["unique_report"]["duplicate_rows"].iloc[0] if not rep["unique_report"].empty else 0

    msgs = []
    if not bad_missing.empty:
        msgs.append(f"Missing required cols: {bad_missing['column'].tolist()}")
    if not bad_types.empty:
        msgs.append(f"Dtype mismatches: {bad_types[['column','expected_dtype','actual_dtype']].to_dict('records')}")
    if not bad_nonneg.empty:
        msgs.append(f"Negative values in non-negative cols: {bad_nonneg['column'].tolist()}")
    if not bad_constant.empty:
        msgs.append(f"Constant-but-shouldn't cols: {bad_constant['column'].tolist()}")
    if dupes:
        msgs.append(f"Duplicate key rows: {dupes}")

    if msgs:
        raise AssertionError(f"[{name}] data quality failures:\n" + "\n".join(msgs))

def write_audit_reports(reports: Mapping[str, pd.DataFrame],
                        out_dir: Path,
                        prefix: str) -> None:
    """
    Save each report DataFrame as CSV for later inspection.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, df in reports.items():
        df.to_csv(out_dir / f"{prefix}_{key}.csv", index=False) 
