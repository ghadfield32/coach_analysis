# dags/nba_advanced_ingest.py
"""
Daily scrape of Basketball‑Reference season‑level advanced metrics.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os, sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from salary_nba_data_pull.scrape_utils import _season_advanced_df

default_args = {
    "owner": "data_eng",
    "email": ["alerts@example.com"],
    "email_on_failure": True,
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "sla": timedelta(hours=1),
}

with DAG(
    dag_id="nba_advanced_ingest",
    start_date=datetime(2025, 7, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["nba", "advanced", "ingest"],
    params={"season": "2024-25"},
) as dag:

    def scrape_adv(**ctx):
        season = ctx["params"]["season"]
        df = _season_advanced_df(season)
        if df.empty:
            raise ValueError(f"No advanced data for {season}")
        out_dir = Path("/workspace/data/new_processed/advanced_metrics")
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_dir / f"advanced_{season}.parquet", index=False)

    PythonOperator(
        task_id="scrape_advanced_metrics",
        python_callable=scrape_adv,
    ) 
