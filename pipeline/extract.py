"""Extract smart building sensor data from the PostgreSQL buffer.

The mqtt-consumer service continuously populates raw_sensor_buffer.
This extractor reads rows from that buffer for the current ETL window
and returns a unified DataFrame for downstream transformation.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sqlalchemy import create_engine, text

from settings.db_config import get_db_config, to_sqlalchemy_url


LOGGER_NAME: str = "pipeline.extract"
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "datasets"
SAMPLE_CSV_PATH: Path = DATA_DIR / "sample_building_sensors.csv"

BUFFER_TABLE: str = "raw_sensor_buffer"
DEFAULT_LOOKBACK_HOURS: int = 2

BUFFER_COLUMNS = [
    "zone_id",
    "zone_name",
    "temperature",
    "humidity",
    "co2",
    "occupancy",
    "energy_kw",
    "published_at",
    "ingested_at",
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


def extract_from_buffer(
    *,
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
    since: Optional[datetime] = None,
) -> pd.DataFrame:
    """Read unprocessed rows from raw_sensor_buffer.

    Args:
        lookback_hours: How many hours back to query if since is not given.
        since: Explicit lower bound for ingested_at. Overrides lookback_hours.

    Returns:
        DataFrame with BUFFER_COLUMNS. Empty DataFrame if no rows found.
    """

    logger = _get_logger()
    cfg = get_db_config()
    engine = create_engine(to_sqlalchemy_url(cfg), pool_pre_ping=True)

    cutoff = since or (datetime.now(timezone.utc) - timedelta(hours=lookback_hours))

    query = text(
        f"SELECT {', '.join(BUFFER_COLUMNS)} FROM {BUFFER_TABLE} "  # noqa: S608
        "WHERE ingested_at >= :cutoff ORDER BY ingested_at ASC"
    )

    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn, params={"cutoff": cutoff})
        logger.info("Buffer extracted rows=%s since=%s", len(df), cutoff.isoformat())
        return df
    except Exception as exc:
        logger.warning("Buffer read failed: %s — returning empty DataFrame", exc)
        return pd.DataFrame(columns=BUFFER_COLUMNS)
    finally:
        engine.dispose()


def extract_from_csv(csv_path: Path = SAMPLE_CSV_PATH) -> pd.DataFrame:
    """Load local sample CSV as fallback / CI seed data.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame with building sensor columns.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """

    logger = _get_logger()
    if not csv_path.exists():
        raise FileNotFoundError(f"Sample CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"zone_id", "zone_name", "temperature", "humidity", "co2", "occupancy", "energy_kw"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Sample CSV missing columns: {missing}")

    logger.info("CSV extracted rows=%s", len(df))
    return df


def run_extraction(
    *,
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
    since: Optional[datetime] = None,
    use_csv_fallback: bool = False,
) -> pd.DataFrame:
    """Run extraction from the MQTT buffer (or CSV fallback for CI).

    Args:
        lookback_hours: How many hours back to query.
        since: Explicit lower bound. Overrides lookback_hours.
        use_csv_fallback: When True, read from sample CSV instead of DB.

    Returns:
        DataFrame of raw sensor readings.
    """

    logger = _get_logger()
    if use_csv_fallback:
        logger.info("CSV fallback mode — skipping buffer read")
        return extract_from_csv(SAMPLE_CSV_PATH)

    df = extract_from_buffer(lookback_hours=lookback_hours, since=since)
    if df.empty:
        logger.warning("Buffer returned no rows; pipeline will short-circuit at validate")
    return df


def df_to_xcom_json(df: pd.DataFrame) -> str:
    return df.to_json(orient="records", date_format="iso")


def df_from_xcom_json(payload: str) -> pd.DataFrame:
    records = json.loads(payload)
    return pd.DataFrame.from_records(records)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract smart building sensor data")
    parser.add_argument(
        "--csv-fallback",
        action="store_true",
        help="Use sample CSV instead of DB buffer (for CI).",
    )
    parser.add_argument(
        "--out",
        default=str(DATA_DIR / "extracted_buffer.csv"),
        help="Write extraction output to this CSV path.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    df = run_extraction(use_csv_fallback=bool(args.csv_fallback))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    _get_logger().info("Wrote extraction output to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
