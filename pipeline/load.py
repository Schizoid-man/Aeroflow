"""Load smart building sensor data into PostgreSQL.

Creates required tables (buffer and processed) and appends processed rows
using pandas.to_sql(). The buffer clear step runs separately in the DAG.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError

from settings.db_config import get_db_config, to_sqlalchemy_url


LOGGER_NAME: str = "pipeline.load"
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SQL_CREATE_TABLES: Path = PROJECT_ROOT / "schema" / "create_tables.sql"

BUFFER_TABLE: str = "raw_sensor_buffer"
PROCESSED_TABLE: str = "processed_sensor_data"
CHUNK_SIZE: int = 500

PROCESSED_COLUMNS = [
    "zone_id",
    "zone_name",
    "temperature",
    "humidity",
    "co2",
    "occupancy",
    "energy_kw",
    "hour_of_day",
    "day_of_week",
    "is_business_hours",
    "rolling_avg_energy_7d",
    "temperature_norm",
    "humidity_norm",
    "co2_norm",
]


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


def create_pg_engine() -> Engine:
    cfg = get_db_config()
    return create_engine(to_sqlalchemy_url(cfg), pool_pre_ping=True)


def ensure_tables(engine: Engine, sql_path: Path = SQL_CREATE_TABLES) -> None:
    """Create required tables if they do not exist."""

    logger = _get_logger()
    if not sql_path.exists():
        raise FileNotFoundError(f"Missing SQL schema file: {sql_path}")

    sql_text = sql_path.read_text(encoding="utf-8")
    with engine.begin() as conn:
        try:
            conn.execute(
                text("SELECT pg_advisory_xact_lock(hashtext('aeroflow.ensure_tables'))")
            )
        except Exception:  # noqa: BLE001
            logger.info("Could not acquire advisory lock; proceeding without it")

        try:
            conn.execute(text(sql_text))
        except ProgrammingError as exc:
            pgcode = getattr(getattr(exc, "orig", None), "pgcode", None)
            if pgcode == "42P07" or "already exists" in str(exc).lower():
                logger.info("Tables already exist; continuing (%s)", exc)
            else:
                raise
    logger.info("Ensured tables exist via %s", sql_path)


def load_dataframe(*, engine: Engine, df: pd.DataFrame, table_name: str) -> int:
    logger = _get_logger()
    if df.empty:
        logger.info("No rows to load into %s", table_name)
        return 0

    df.to_sql(
        table_name,
        engine,
        if_exists="append",
        index=False,
        chunksize=CHUNK_SIZE,
        method="multi",
    )
    row_count = int(len(df))
    logger.info("Inserted %s rows into %s", row_count, table_name)
    return row_count


def prepare_processed_for_load(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PROCESSED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Processed DataFrame missing required columns: {missing}")
    return df[PROCESSED_COLUMNS].copy()


def load_processed_dataframe(*, engine: Engine, df: pd.DataFrame) -> int:
    return load_dataframe(
        engine=engine,
        df=prepare_processed_for_load(df),
        table_name=PROCESSED_TABLE,
    )


def clear_buffer(*, engine: Engine, before: Optional[datetime] = None) -> int:
    """Delete rows from raw_sensor_buffer that have been processed.

    Args:
        engine: SQLAlchemy engine.
        before: Delete rows with ingested_at < before. Defaults to now.

    Returns:
        Number of rows deleted.
    """

    logger = _get_logger()
    cutoff = before or datetime.now(timezone.utc)
    with engine.begin() as conn:
        result = conn.execute(
            text(f"DELETE FROM {BUFFER_TABLE} WHERE ingested_at < :cutoff"),  # noqa: S608
            {"cutoff": cutoff},
        )
        deleted = result.rowcount if result.rowcount is not None else 0
    logger.info("Cleared %s rows from buffer (before=%s)", deleted, cutoff.isoformat())
    return deleted
