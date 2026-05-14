"""Load air quality datasets into PostgreSQL.

This module creates required tables (raw and processed) and appends rows using
pandas.to_sql().
"""

from __future__ import annotations

import logging
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

RAW_TABLE: str = "raw_air_quality"
PROCESSED_TABLE: str = "processed_air_quality"
CHUNK_SIZE: int = 500

RAW_COLUMNS = [
    "location",
    "city",
    "country",
    "parameter",
    "value",
    "unit",
    "date_utc",
    "latitude",
    "longitude",
    "source",
]

PROCESSED_COLUMNS = [
    "date",
    "city",
    "pm2_5",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
    "aqi",
    "aqi_category",
    "hour_of_day",
    "day_of_week",
    "rolling_avg_pm25_7d",
    "pm2_5_norm",
    "pm10_norm",
    "no2_norm",
]


def _get_logger() -> logging.Logger:
    """Create or return the module logger."""

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


def create_pg_engine() -> Engine:
    """Create a SQLAlchemy engine for PostgreSQL using .env configuration."""

    cfg = get_db_config()
    return create_engine(to_sqlalchemy_url(cfg), pool_pre_ping=True)


def ensure_tables(engine: Engine, sql_path: Path = SQL_CREATE_TABLES) -> None:
    """Create required tables if they do not exist."""

    logger = _get_logger()
    if not sql_path.exists():
        raise FileNotFoundError(f"Missing SQL schema file: {sql_path}")

    sql_text = sql_path.read_text(encoding="utf-8")
    with engine.begin() as conn:
        # If multiple Airflow tasks call ensure_tables concurrently, Postgres can
        # still raise DuplicateTable despite IF NOT EXISTS due to a race.
        # Serialize creation with an advisory *transaction* lock.
        try:
            conn.execute(
                text(
                    "SELECT pg_advisory_xact_lock(hashtext('aeropulse.ensure_tables'))"
                )
            )
        except Exception:  # noqa: BLE001
            logger.info(
                "Could not acquire advisory lock; proceeding without it"
            )

        try:
            conn.execute(text(sql_text))
        except ProgrammingError as exc:
            pgcode = getattr(getattr(exc, "orig", None), "pgcode", None)
            if pgcode == "42P07" or "already exists" in str(exc).lower():
                logger.info(
                    "Tables appear to already exist; continuing (%s)", exc
                )
            else:
                raise
    logger.info("Ensured tables exist via %s", sql_path)


def load_dataframe(
    *,
    engine: Engine,
    df: pd.DataFrame,
    table_name: str,
) -> int:
    """Append DataFrame rows into a table."""

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


def prepare_raw_for_load(df: pd.DataFrame) -> pd.DataFrame:
    """Select and validate raw table columns."""

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw DataFrame missing required columns: {missing}")
    return df[RAW_COLUMNS].copy()


def prepare_processed_for_load(df: pd.DataFrame) -> pd.DataFrame:
    """Select and validate processed table columns."""

    missing = [c for c in PROCESSED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Processed DataFrame missing required columns: {missing}"
        )
    return df[PROCESSED_COLUMNS].copy()


def load_raw_dataframe(*, engine: Engine, df: pd.DataFrame) -> int:
    """Load raw dataset into raw_air_quality."""

    prepared = prepare_raw_for_load(df)
    return load_dataframe(engine=engine, df=prepared, table_name=RAW_TABLE)


def load_processed_dataframe(*, engine: Engine, df: pd.DataFrame) -> int:
    """Load processed dataset into processed_air_quality."""

    return load_dataframe(
        engine=engine,
        df=prepare_processed_for_load(df),
        table_name=PROCESSED_TABLE,
    )


def load_raw_and_processed(
    *,
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    engine: Optional[Engine] = None,
) -> None:
    """Create tables and load both raw and processed datasets."""

    logger = _get_logger()
    eng = engine or create_pg_engine()
    ensure_tables(eng)
    raw_rows = load_raw_dataframe(engine=eng, df=raw_df)
    processed_rows = load_processed_dataframe(engine=eng, df=processed_df)
    logger.info(
        "Load complete: raw_rows=%s processed_rows=%s",
        raw_rows,
        processed_rows,
    )
