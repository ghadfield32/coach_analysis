# dags/nba_data_loader.py
"""
Fan‑in loader: waits for api_ingest + advanced_ingest + injury_etl,
then materialises season tables and a joined view in DuckDB.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta
from pathlib import Path
import sys, os, duckdb, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from salary_nba_data_pull.data_utils import validate_data

DATA_ROOT = Path("/workspace/data")

default_args = {
    "owner": "data_eng",
    "email": ["alerts@example.com"],
    "email_on_failure": True,
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "sla": timedelta(hours=3),
}

with DAG(
    dag_id="nba_data_loader",
    start_date=datetime(2025, 7, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["nba", "loader", "duckdb"],
    params={"season": "2024-25"},
) as dag:

    # ─── sensors (one per upstream DAG) ────────────────────────────────
    sensor_args = dict(
        poke_interval=300,
        mode="reschedule",   # avoids tying up a worker slot
    )
    wait_api = ExternalTaskSensor(
        task_id="wait_api_ingest",
        external_dag_id="nba_api_ingest",
        external_task_id="scrape_season_data",
        timeout=3600,
        **sensor_args,
    )
    wait_adv = ExternalTaskSensor(
        task_id="wait_advanced_ingest",
        external_dag_id="nba_advanced_ingest",
        external_task_id="scrape_advanced_metrics",
        timeout=3600,
        **sensor_args,
    )
    wait_injury = ExternalTaskSensor(
        task_id="wait_injury_etl",
        external_dag_id="injury_etl",
        external_task_id="process_injury_data",
        timeout=7200,
        poke_interval=600,
        mode="reschedule",
    )

    # ─── loader task ───────────────────────────────────────────────────
    def load_to_duckdb(**ctx):
        season = ctx["params"]["season"]
        db = DATA_ROOT / "nba_stats.duckdb"
        con = duckdb.connect(db)
        sources = {
            f"player_{season}": DATA_ROOT / f"new_processed/season={season}/part.parquet",
            f"advanced_{season}": DATA_ROOT / f"new_processed/advanced_metrics/advanced_{season}.parquet",
            "injury_master": DATA_ROOT / "new_processed/injury_reports/injury_master.parquet",
        }

        for alias, path in sources.items():
            if path.exists():
                if alias.startswith("player"):
                    df = pd.read_parquet(path)
                    validate_data(df, name=alias, save_reports=True)
                con.execute(
                    f"CREATE OR REPLACE TABLE {alias.replace('-', '_')} AS "
                    f"SELECT * FROM read_parquet('{path}')"
                )

        # materialised view – wildcard parquet scan is fine too
        con.execute(f"""
            CREATE OR REPLACE VIEW v_player_full_{season.replace('-', '_')} AS
            SELECT *
            FROM player_{season.replace('-', '_')} p
            LEFT JOIN advanced_{season.replace('-', '_')} a USING(player, season)
            LEFT JOIN injury_master i USING(player, season)
        """)
        con.close()

    loader = PythonOperator(
        task_id="validate_and_load",
        python_callable=load_to_duckdb,
    )

    [wait_api, wait_adv, wait_injury] >> loader 
