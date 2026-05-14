"""Airflow DAG for daily air quality ETL.

DAG: air_quality_etl_pipeline
Schedule: @daily

Tasks:
1) extract -> 2) validate_raw -> 3) transform -> 4) load_raw + load_processed
-> 5) notify
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict

from airflow import DAG
from airflow.operators.python import PythonOperator

from pipeline import extract, load, transform


LOGGER_NAME: str = "dags.air_quality_etl_pipeline"


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def task_extract(**context: Any) -> Dict[str, Any]:
    """Extract unified dataset and push to XCom as JSON."""

    logger = _get_logger()
    df = extract.run_extraction(dry_run=False)
    payload = extract.df_to_xcom_json(df)
    context["ti"].xcom_push(key="raw_df_json", value=payload)
    logger.info("Extracted rows=%s", len(df))
    return {"raw_rows": int(len(df))}


def task_validate_raw(**context: Any) -> None:
    """Validate raw data shape and required columns."""

    logger = _get_logger()
    payload = context["ti"].xcom_pull(key="raw_df_json", task_ids="task_extract")
    df = extract.df_from_xcom_json(payload)

    required = {"city", "date", "source"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Raw validation failed; missing columns: {missing}")
    if len(df) <= 0:
        raise ValueError("Raw validation failed; dataset is empty")
    logger.info("Raw validation passed; rows=%s cols=%s", len(df), len(df.columns))


def task_transform(**context: Any) -> Dict[str, Any]:
    """Transform raw data and push transformed DF to XCom."""

    logger = _get_logger()
    payload = context["ti"].xcom_pull(key="raw_df_json", task_ids="task_extract")
    raw_df = extract.df_from_xcom_json(payload)
    processed_df = transform.run_transformation(raw_df)
    context["ti"].xcom_push(
        key="processed_df_json", value=extract.df_to_xcom_json(processed_df)
    )
    logger.info("Transformed rows=%s", len(processed_df))
    return {"processed_rows": int(len(processed_df))}


def task_load_raw(**context: Any) -> int:
    """Load raw dataset into PostgreSQL."""

    logger = _get_logger()
    payload = context["ti"].xcom_pull(key="raw_df_json", task_ids="task_extract")
    raw_df = extract.df_from_xcom_json(payload)
    eng = load.create_pg_engine()
    load.ensure_tables(eng)
    inserted = load.load_raw_dataframe(engine=eng, df=raw_df)
    logger.info("Loaded raw rows=%s", inserted)
    return int(inserted)


def task_load_processed(**context: Any) -> int:
    """Load processed dataset into PostgreSQL."""

    logger = _get_logger()
    payload = context["ti"].xcom_pull(
        key="processed_df_json", task_ids="task_transform"
    )
    processed_df = extract.df_from_xcom_json(payload)
    eng = load.create_pg_engine()
    load.ensure_tables(eng)
    inserted = load.load_processed_dataframe(engine=eng, df=processed_df)
    logger.info("Loaded processed rows=%s", inserted)
    return int(inserted)


def task_notify(**context: Any) -> None:
    """Print a success summary."""

    logger = _get_logger()
    raw_rows = context["ti"].xcom_pull(
        task_ids="task_extract", key="return_value"
    )
    processed_rows = context["ti"].xcom_pull(
        task_ids="task_transform", key="return_value"
    )
    now = datetime.utcnow().isoformat()
    logger.info(
        "ETL pipeline succeeded at %s | raw=%s processed=%s",
        now,
        raw_rows,
        processed_rows,
    )


with DAG(
    dag_id="air_quality_etl_pipeline",
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["etl", "air-quality", "ml"],
) as dag:
    t_extract = PythonOperator(task_id="task_extract", python_callable=task_extract)
    t_validate = PythonOperator(
        task_id="task_validate_raw", python_callable=task_validate_raw
    )
    t_transform = PythonOperator(
        task_id="task_transform", python_callable=task_transform
    )
    t_load_raw = PythonOperator(
        task_id="task_load_raw", python_callable=task_load_raw
    )
    t_load_processed = PythonOperator(
        task_id="task_load_processed", python_callable=task_load_processed
    )
    t_notify = PythonOperator(task_id="task_notify", python_callable=task_notify)

    (
        t_extract
        >> t_validate
        >> t_transform
        >> [t_load_raw, t_load_processed]
        >> t_notify
    )
