"""Airflow DAG for hourly smart building sensor ETL.

DAG: smart_building_etl_pipeline
Schedule: @hourly

Tasks:
1) extract      - read raw_sensor_buffer from PostgreSQL
2) validate     - check required columns and row count
3) transform    - feature engineering + normalization
4) load         - insert into processed_sensor_data
5) clear_buffer - delete processed rows from raw_sensor_buffer
6) notify       - log run summary
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict

from airflow import DAG
from airflow.operators.python import PythonOperator

from pipeline import extract, load, transform


LOGGER_NAME: str = "dags.smart_building_etl_pipeline"


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def task_extract(**context: Any) -> Dict[str, Any]:
    logger = _get_logger()
    df = extract.run_extraction(lookback_hours=2)
    payload = extract.df_to_xcom_json(df)
    context["ti"].xcom_push(key="raw_df_json", value=payload)
    logger.info("Extracted rows=%s", len(df))
    return {"raw_rows": int(len(df))}


def task_validate(**context: Any) -> None:
    logger = _get_logger()
    payload = context["ti"].xcom_pull(key="raw_df_json", task_ids="task_extract")
    df = extract.df_from_xcom_json(payload)

    if len(df) == 0:
        logger.warning("Buffer empty — skipping this run")
        return

    required = {"zone_id", "energy_kw", "published_at"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Raw validation failed; missing columns: {missing}")

    logger.info("Validation passed; rows=%s", len(df))


def task_transform(**context: Any) -> Dict[str, Any]:
    logger = _get_logger()
    payload = context["ti"].xcom_pull(key="raw_df_json", task_ids="task_extract")
    raw_df = extract.df_from_xcom_json(payload)

    if len(raw_df) == 0:
        context["ti"].xcom_push(key="processed_df_json", value="[]")
        return {"processed_rows": 0}

    processed_df = transform.run_transformation(raw_df)
    context["ti"].xcom_push(
        key="processed_df_json", value=extract.df_to_xcom_json(processed_df)
    )
    logger.info("Transformed rows=%s", len(processed_df))
    return {"processed_rows": int(len(processed_df))}


def task_load(**context: Any) -> int:
    logger = _get_logger()
    payload = context["ti"].xcom_pull(key="processed_df_json", task_ids="task_transform")
    processed_df = extract.df_from_xcom_json(payload)

    if len(processed_df) == 0:
        logger.info("No processed rows to load")
        return 0

    eng = load.create_pg_engine()
    load.ensure_tables(eng)
    inserted = load.load_processed_dataframe(engine=eng, df=processed_df)
    logger.info("Loaded processed rows=%s", inserted)
    return int(inserted)


def task_clear_buffer(**context: Any) -> int:
    logger = _get_logger()
    eng = load.create_pg_engine()
    deleted = load.clear_buffer(engine=eng)
    logger.info("Cleared buffer rows=%s", deleted)
    return deleted


def task_notify(**context: Any) -> None:
    logger = _get_logger()
    raw_rows = (context["ti"].xcom_pull(task_ids="task_extract", key="return_value") or {})
    processed_rows = (context["ti"].xcom_pull(task_ids="task_transform", key="return_value") or {})
    now = datetime.utcnow().isoformat()
    logger.info(
        "ETL succeeded at %s | raw=%s processed=%s",
        now,
        raw_rows,
        processed_rows,
    )


with DAG(
    dag_id="smart_building_etl_pipeline",
    schedule_interval="@hourly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["etl", "iot", "smart-building"],
) as dag:
    t_extract = PythonOperator(task_id="task_extract", python_callable=task_extract)
    t_validate = PythonOperator(task_id="task_validate", python_callable=task_validate)
    t_transform = PythonOperator(task_id="task_transform", python_callable=task_transform)
    t_load = PythonOperator(task_id="task_load", python_callable=task_load)
    t_clear = PythonOperator(task_id="task_clear_buffer", python_callable=task_clear_buffer)
    t_notify = PythonOperator(task_id="task_notify", python_callable=task_notify)

    t_extract >> t_validate >> t_transform >> t_load >> t_clear >> t_notify
