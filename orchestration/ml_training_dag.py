"""Airflow DAG for weekly ML training.

DAG: ml_training_pipeline
Schedule: @weekly

Tasks:
load_data_from_db -> preprocess_features -> train_model -> evaluate_model -> save_model
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

from forecasting import evaluate, train


LOGGER_NAME: str = "dags.ml_training_pipeline"


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


def load_data_from_db(**context: Any) -> None:
    logger = _get_logger()
    df = train.load_processed_data()
    context["ti"].xcom_push(key="processed_df_json", value=df.to_json(orient="records"))
    logger.info("Loaded processed rows=%s", len(df))


def preprocess_features(**context: Any) -> None:
    logger = _get_logger()
    payload = context["ti"].xcom_pull(key="processed_df_json", task_ids="load_data_from_db")
    df = pd.read_json(payload, orient="records")
    X, y = train.build_features_and_target(df)
    context["ti"].xcom_push(key="X_json", value=X.to_json(orient="records"))
    context["ti"].xcom_push(key="y_json", value=y.to_json(orient="records"))
    logger.info("Prepared feature matrix shape=%s", X.shape)


def train_model(**context: Any) -> Dict[str, Any]:
    logger = _get_logger()
    payload = context["ti"].xcom_pull(key="processed_df_json", task_ids="load_data_from_db")
    df = pd.read_json(payload, orient="records")
    model_path = train.train_and_save_model(df)
    context["ti"].xcom_push(key="model_path", value=str(model_path))
    logger.info("Trained model saved to %s", model_path)
    return {"model_path": str(model_path)}


def evaluate_model(**context: Any) -> Dict[str, Any]:
    logger = _get_logger()
    model_path = context["ti"].xcom_pull(key="model_path", task_ids="train_model")
    metrics = evaluate.evaluate_saved_model(model_path=str(model_path))
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


def save_model(**context: Any) -> None:
    logger = _get_logger()
    model_path = context["ti"].xcom_pull(key="model_path", task_ids="train_model")
    logger.info("Model artifact ready at %s", model_path)


with DAG(
    dag_id="ml_training_pipeline",
    schedule_interval="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ml", "iot", "smart-building"],
) as dag:
    t_load = PythonOperator(task_id="load_data_from_db", python_callable=load_data_from_db)
    t_prep = PythonOperator(task_id="preprocess_features", python_callable=preprocess_features)
    t_train = PythonOperator(task_id="train_model", python_callable=train_model)
    t_eval = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)
    t_save = PythonOperator(task_id="save_model", python_callable=save_model)

    t_load >> t_prep >> t_train >> t_eval >> t_save
