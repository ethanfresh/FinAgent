"""Nightly canary DAG: re-run a fixed case subset and alert on drift.

Runs `finagent canary` inside the app's own environment (not Airflow's) via
BashOperator — the standard pattern for triggering a job whose dependencies
live in a different environment than the orchestrator's.
"""

from datetime import UTC, datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

FINAGENT_PROJECT_DIR = "/Users/ethanfreshman/Desktop/FinAgent"

default_args = {
    "owner": "finagent",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="finagent_canary",
    description="Re-run a fixed eval subset nightly and fail the DAG if the pass rate drops",
    default_args=default_args,
    schedule="0 6 * * *",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    tags=["finagent", "eval", "drift-detection"],
) as dag:
    run_canary = BashOperator(
        task_id="run_canary",
        bash_command=(
            f"cd {FINAGENT_PROJECT_DIR} && "
            "$HOME/.local/bin/uv run finagent canary --threshold 0.85"
        ),
    )
