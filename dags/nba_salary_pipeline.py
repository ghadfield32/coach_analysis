# dags/nba_salary_pipeline.py
# DEPRECATED: This monolithic DAG has been split into separate source-specific DAGs
# 
# New Architecture:
# - nba_salary_ingest.py: Player & team salary scraping (daily)
# - nba_advanced_ingest.py: Advanced metrics scraping (daily)  
# - salary_cap_snapshot.py: Salary cap history (yearly)
# - injury_etl.py: Injury data processing (monthly)
# - nba_data_loader.py: Unified loader with ExternalTaskSensors
#
# Benefits:
# - Isolated failures: one source failing doesn't block others
# - Different cadences: salary scraping daily, cap history yearly
# - Easier maintenance: teams can iterate on one source independently
# - Better monitoring: granular SLAs and retry policies per source

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from salary_nba_data_pull.main import main

default_args = dict(
    owner="data_eng",
    email=["alerts@example.com"],
    email_on_failure=True,
    retries=2,
    retry_delay=timedelta(minutes=5),
    sla=timedelta(hours=1),
)

with DAG(
    "nba_salary_pipeline",
    start_date=datetime(2025, 7, 1),
    schedule="@daily",
    default_args=default_args,
    catchup=False,
    tags=["nba", "deprecated", "monolithic"],
) as dag:

    def deprecated_warning(**ctx):
        """Warning function to inform users about the new architecture."""
        print("""
        ⚠️  DEPRECATED: This monolithic DAG has been split into separate DAGs
        
        New Architecture:
        - nba_salary_ingest.py: Player & team salary scraping (daily)
        - nba_advanced_ingest.py: Advanced metrics scraping (daily)  
        - salary_cap_snapshot.py: Salary cap history (yearly)
        - injury_etl.py: Injury data processing (monthly)
        - nba_data_loader.py: Unified loader with ExternalTaskSensors
        
        Benefits:
        - Isolated failures: one source failing doesn't block others
        - Different cadences: salary scraping daily, cap history yearly
        - Easier maintenance: teams can iterate on one source independently
        - Better monitoring: granular SLAs and retry policies per source
        
        Please migrate to the new architecture for better reliability and maintainability.
        """)
        
        # Still run the old logic for backward compatibility
        season = ctx["dag_run"].conf.get("season", "2024-25")
        y = int(season[:4])
        main(y, y, small_debug=True, overwrite=False)

    deprecated_task = PythonOperator(
        task_id="deprecated_warning", 
        python_callable=deprecated_warning,
        doc="DEPRECATED: Use the new split DAG architecture instead"
    )

    # Single task for backward compatibility
    deprecated_task 