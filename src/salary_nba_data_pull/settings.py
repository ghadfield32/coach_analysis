# src/salary_nba_data_pull/settings.py
from pathlib import Path
import os
import typing as _t

# 🗂️  Central data directory (override via env if needed)
DATA_PROCESSED_DIR = Path(
    (Path(__file__).resolve().parent.parent.parent)  # project root
    / "data"
    / "new_processed"
)

# optional: allow `DATA_PROCESSED_DIR=/tmp/demo python main.py …`
ENV_OVERRIDE: _t.Optional[str] = os.getenv("DATA_PROCESSED_DIR")
if ENV_OVERRIDE:
    DATA_PROCESSED_DIR = Path(ENV_OVERRIDE).expanduser().resolve()

# Legacy path for backward compatibility
LEGACY_DATA_PROCESSED_DIR = Path(
    (Path(__file__).resolve().parent.parent.parent)  # project root
    / "data"
    / "processed"
) 
