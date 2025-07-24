# dags/nba_api_ingest.py
"""
Pulls roster + box‑score data from nba_api once per hour and writes Parquet
partitions under data/new_processed/season=<YYYY-YY>/part.parquet.

Why hourly?
• The NBA Stats endpoints update within minutes after a game ends.
• Hourly keeps your lake near‑real‑time without hammering the API.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os, sys, pathlib

# Allow `salary_nba_data_pull` imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from salary_nba_data_pull.main import main as pull_main

default_args = {
    "owner": "data_eng",
    "email": ["alerts@example.com"],
    "email_on_failure": True,
    "depends_on_past": False,      # explicit
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "sla": timedelta(hours=1),
}

with DAG(
    dag_id="nba_api_ingest",
    start_date=datetime(2025, 7, 1),
    schedule="@hourly",            # unified scheduling API (Airflow ≥ 2.4)
    catchup=False,
    default_args=default_args,
    max_active_runs=1,             # avoid overlapping pulls
    tags=["nba", "api", "ingest"],
    params={"season": "2024-25"},  # visible & overridable in the UI
) as dag:

    def pull_season(**context):
        season = context["params"]["season"]
        start_year = int(season[:4])
        pull_main(
            start_year=start_year,
            end_year=start_year,
            small_debug=True,
            workers=8,
            overwrite=False,
        )

    PythonOperator(
        task_id="scrape_season_data",
        python_callable=pull_season,
    ) 
