"""Extract air quality data from multiple sources.

Sources:
1) OpenAQ REST API (live)
2) Local CSV dataset
3) Local SQLite database (synthetic historical data)

The extractor merges all sources into a unified pandas DataFrame and annotates
each row with a `source` column: 'api', 'csv', or 'db'.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


LOGGER_NAME: str = "pipeline.extract"
OPENAQ_V3_BASE_URL: str = "https://api.openaq.org/v3"
OPENAQ_V3_LOCATIONS_URL: str = f"{OPENAQ_V3_BASE_URL}/locations"
OPENAQ_V3_SENSOR_MEASUREMENTS_URL_TMPL: str = (
    f"{OPENAQ_V3_BASE_URL}/sensors/{{sensor_id}}/measurements"
)
OPENAQ_COUNTRY: str = "IN"
OPENAQ_PARAMETER: str = "pm25"
OPENAQ_LIMIT: int = 100
OPENAQ_TIMEOUT_SEC: int = 30
OPENAQ_MAX_PAGES: int = 10
OPENAQ_RETRIES: int = 3

OPENAQ_API_KEY_ENV: str = "OPENAQ_API_KEY"

# Keep v3 ingestion bounded so the ETL stays fast.
OPENAQ_V3_MAX_LOCATIONS: int = 25
OPENAQ_V3_MAX_SENSORS_PER_LOCATION: int = 5
OPENAQ_V3_MAX_MEAS_PAGES_PER_SENSOR: int = 2
OPENAQ_V3_DAYS_BACK: int = 7

SQLITE_ROWS: int = 500
SQLITE_CITY_COUNT: int = 10
SQLITE_YEARS_BACK: int = 2

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "datasets"
SAMPLE_CSV_PATH: Path = DATA_DIR / "sample_aqi_india.csv"
SQLITE_DB_PATH: Path = DATA_DIR / "historical_aqi.db"


def _empty_openaq_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "location",
            "city",
            "country",
            "parameter",
            "value",
            "unit",
            "date_utc",
            "latitude",
            "longitude",
        ]
    )


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


def ensure_datasets_present() -> None:
    """Ensure local datasets exist; trigger download script if missing."""

    logger = _get_logger()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if SAMPLE_CSV_PATH.exists():
        return

    logger.info(
        "Local data files missing; running setup/download_datasets.py"
    )
    try:
        from setup.download_datasets import main as download_main

        download_main(project_root=PROJECT_ROOT)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Dataset auto-download failed (%s). You can proceed with your own "
            "CSV in datasets/sample_aqi_india.csv.",
            exc,
        )


def extract_from_openaq(
    *,
    session: Optional[requests.Session] = None,
    max_pages: int = OPENAQ_MAX_PAGES,
    dry_run: bool = False,
    strict: bool = False,
) -> pd.DataFrame:
    """Extract PM2.5 measurements from OpenAQ API (v3).

    Args:
        session: Optional requests session.
        max_pages: Max pages to fetch.
        dry_run: When True, do not perform network calls.

    Returns:
        DataFrame with columns: location, city, country, parameter, value,
        unit, date_utc, latitude, longitude.

    Raises:
        RuntimeError: If API repeatedly fails and strict=True.
    """

    logger = _get_logger()
    if dry_run:
        logger.info("Dry-run enabled; skipping OpenAQ API call")
        return _empty_openaq_df()

    api_key = os.getenv(OPENAQ_API_KEY_ENV)
    if not api_key:
        logger.warning(
            "%s is not set; skipping OpenAQ v3 extraction.",
            OPENAQ_API_KEY_ENV,
        )
        return _empty_openaq_df()

    sess = session or requests.Session()
    headers = {
        "User-Agent": "air-quality-ml-pipeline/1.0",
        "X-API-Key": api_key,
    }

    def _iso_z(dt: datetime) -> str:
        # OpenAQ expects RFC3339 timestamps; keep it simple.
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def _param_matches(name: Optional[str]) -> bool:
        if not name:
            return False
        n = str(name).strip().lower().replace(".", "")
        target = str(OPENAQ_PARAMETER).strip().lower().replace(".", "")
        # Common aliases: pm2.5 vs pm25
        return n == target

    def _get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(1, OPENAQ_RETRIES + 1):
            try:
                resp = sess.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=OPENAQ_TIMEOUT_SEC,
                )
                if resp.status_code in {401, 403}:
                    raise RuntimeError(
                        "OpenAQ v3 unauthorized. Ensure a valid API key is "
                        f"provided via {OPENAQ_API_KEY_ENV}."
                    )
                if resp.status_code == 429:
                    raise RuntimeError(
                        "OpenAQ v3 rate-limited (HTTP 429). Try again later."
                    )
                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"OpenAQ HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                return resp.json()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "OpenAQ request failed (attempt=%s/%s): %s",
                    attempt,
                    OPENAQ_RETRIES,
                    exc,
                )
        raise RuntimeError(f"OpenAQ request failed after retries: {last_error}")

    # Step 1: discover locations in the country.
    locations: List[Dict[str, Any]] = []
    page = 1
    while page <= max_pages and len(locations) < OPENAQ_V3_MAX_LOCATIONS:
        params = {
            "iso": OPENAQ_COUNTRY,
            "limit": OPENAQ_LIMIT,
            "page": page,
            "order_by": "id",
            "sort_order": "asc",
        }
        try:
            payload = _get_json(OPENAQ_V3_LOCATIONS_URL, params)
        except Exception as exc:  # noqa: BLE001
            msg = f"OpenAQ locations fetch failed (page={page}): {exc}"
            if strict:
                raise RuntimeError(msg) from exc
            logger.warning("%s; continuing without OpenAQ source", msg)
            return _empty_openaq_df()

        results = payload.get("results") or []
        if not results:
            break

        for loc in results:
            locations.append(loc)
            if len(locations) >= OPENAQ_V3_MAX_LOCATIONS:
                break
        page += 1

    # Step 2: for each location, fetch measurements for a few pm2.5 sensors.
    now = datetime.now(timezone.utc)
    dt_from = _iso_z(now - timedelta(days=OPENAQ_V3_DAYS_BACK))
    dt_to = _iso_z(now)

    rows: List[Dict[str, Any]] = []
    for loc in locations:
        loc_name = loc.get("name")
        loc_city = loc.get("locality")
        country_obj = loc.get("country")
        if isinstance(country_obj, dict):
            loc_country = country_obj.get("code") or OPENAQ_COUNTRY
        elif country_obj:
            loc_country = str(country_obj)
        else:
            loc_country = OPENAQ_COUNTRY
        loc_coords = loc.get("coordinates") or {}

        sensors = loc.get("sensors") or []
        pm_sensors = []
        for s in sensors:
            param = (s or {}).get("parameter") or {}
            if _param_matches(param.get("name")):
                pm_sensors.append(s)

        pm_sensors = pm_sensors[:OPENAQ_V3_MAX_SENSORS_PER_LOCATION]
        for sensor in pm_sensors:
            sensor_id = (sensor or {}).get("id")
            if sensor_id is None:
                continue
            meas_url = OPENAQ_V3_SENSOR_MEASUREMENTS_URL_TMPL.format(
                sensor_id=sensor_id
            )

            meas_page = 1
            while meas_page <= OPENAQ_V3_MAX_MEAS_PAGES_PER_SENSOR:
                params = {
                    "datetime_from": dt_from,
                    "datetime_to": dt_to,
                    "limit": OPENAQ_LIMIT,
                    "page": meas_page,
                }
                try:
                    payload = _get_json(meas_url, params)
                except Exception as exc:  # noqa: BLE001
                    msg = (
                        "OpenAQ measurements fetch failed "
                        f"(sensor_id={sensor_id} page={meas_page}): {exc}"
                    )
                    if strict:
                        raise RuntimeError(msg) from exc
                    logger.warning(msg)
                    break

                results = payload.get("results") or []
                if not results:
                    break

                for item in results:
                    item_coords = item.get("coordinates") or {}
                    coords = item_coords or loc_coords or {}

                    period = item.get("period") or {}
                    dt_from_obj = (period.get("datetimeFrom") or {})
                    dt_to_obj = (period.get("datetimeTo") or {})
                    date_utc = dt_from_obj.get("utc") or dt_to_obj.get("utc")

                    param = item.get("parameter") or {}
                    unit = param.get("units")

                    rows.append(
                        {
                            "location": loc_name,
                            "city": loc_city,
                            "country": loc_country,
                            "parameter": OPENAQ_PARAMETER,
                            "value": item.get("value"),
                            "unit": unit,
                            "date_utc": date_utc,
                            "latitude": coords.get("latitude"),
                            "longitude": coords.get("longitude"),
                        }
                    )

                # Heuristic paging: stop if last page wasn't full.
                if len(results) < int(OPENAQ_LIMIT):
                    break
                meas_page += 1

    df = pd.DataFrame(rows)
    logger.info("OpenAQ v3 extracted rows: %s", len(df))
    if df.empty:
        return _empty_openaq_df()
    return df


def extract_from_csv(csv_path: Path) -> pd.DataFrame:
    """Load local CSV data.

    Args:
        csv_path: Path to CSV.

    Returns:
        DataFrame with expected columns.

    Raises:
        FileNotFoundError: If file is missing.
        ValueError: If required columns are missing.
    """

    logger = _get_logger()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    expected = {
        "date",
        "city",
        "pm2_5",
        "pm10",
        "no2",
        "so2",
        "co",
        "o3",
        "aqi",
    }
    missing = sorted(list(expected - set(df.columns)))
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    logger.info("CSV preview (first 5 rows):\n%s", df.head(5).to_string())
    return df


def _sqlite_connect(db_path: Path) -> sqlite3.Connection:
    """Connect to SQLite, creating the parent directory if needed."""

    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))


def _create_and_seed_sqlite(
    conn: sqlite3.Connection,
    *,
    seed_rows: int,
    city_count: int,
    years_back: int,
) -> None:
    """Create and seed the SQLite historical_aqi table if empty."""

    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS historical_aqi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            city TEXT NOT NULL,
            pm2_5 REAL,
            pm10 REAL,
            no2 REAL,
            so2 REAL,
            co REAL,
            o3 REAL,
            aqi INTEGER
        )
        """
    )

    cur.execute("SELECT COUNT(*) FROM historical_aqi")
    count = int(cur.fetchone()[0])
    if count > 0:
        return

    rng = np.random.default_rng(42)
    cities = [
        "Bengaluru",
        "Delhi",
        "Mumbai",
        "Chennai",
        "Kolkata",
        "Hyderabad",
        "Pune",
        "Ahmedabad",
        "Jaipur",
        "Lucknow",
    ][:city_count]

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * years_back)
    total_days = max(1, (end - start).days)
    records: List[Tuple[Any, ...]] = []
    for _ in range(seed_rows):
        city = str(rng.choice(cities))
        day_offset = int(rng.integers(0, total_days))
        ts = (start + timedelta(days=day_offset)).replace(
            hour=int(rng.integers(0, 24)),
            minute=int(rng.integers(0, 60)),
            second=0,
            microsecond=0,
        )

        pm2_5 = float(np.clip(rng.normal(60, 25), 1, 350))
        pm10 = float(np.clip(rng.normal(110, 40), 5, 500))
        no2 = float(np.clip(rng.normal(30, 15), 1, 200))
        so2 = float(np.clip(rng.normal(10, 6), 0.1, 100))
        co = float(np.clip(rng.normal(1.0, 0.5), 0.1, 10))
        o3 = float(np.clip(rng.normal(25, 10), 1, 200))
        aqi = int(np.clip(rng.normal(140, 70), 1, 500))

        records.append(
            (
                ts.isoformat(),
                city,
                pm2_5,
                pm10,
                no2,
                so2,
                co,
                o3,
                aqi,
            )
        )

    cur.executemany(
        """
        INSERT INTO historical_aqi (
            date, city, pm2_5, pm10, no2, so2, co, o3, aqi
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )
    conn.commit()


def extract_from_sqlite(
    db_path: Path,
    *,
    seed_rows: int = SQLITE_ROWS,
    city_count: int = SQLITE_CITY_COUNT,
    years_back: int = SQLITE_YEARS_BACK,
) -> pd.DataFrame:
    """Extract or generate synthetic AQI data from a local SQLite database.

    Args:
        db_path: Path to SQLite DB.
        seed_rows: Number of synthetic rows to generate if DB is empty.
        city_count: Number of cities to simulate.
        years_back: How many years back to simulate.

    Returns:
        DataFrame with synthetic historical data.
    """

    logger = _get_logger()
    conn = _sqlite_connect(db_path)
    try:
        _create_and_seed_sqlite(
            conn,
            seed_rows=seed_rows,
            city_count=city_count,
            years_back=years_back,
        )
        df = pd.read_sql_query(
            "SELECT date, city, pm2_5, pm10, no2, so2, co, o3, aqi "
            "FROM historical_aqi WHERE aqi > 0",
            conn,
        )
    finally:
        conn.close()

    logger.info("SQLite extracted rows: %s", len(df))
    return df


def _unify_sources(
    *,
    api_df: pd.DataFrame,
    csv_df: pd.DataFrame,
    db_df: pd.DataFrame,
) -> pd.DataFrame:
    """Unify API/CSV/SQLite shapes into one wide table.

    Args:
        api_df: API measurement rows.
        csv_df: Local CSV rows.
        db_df: SQLite historical rows.

    Returns:
        A unified DataFrame containing a superset of raw + wide columns.
    """

    logger = _get_logger()

    api_wide = pd.DataFrame()
    if not api_df.empty:
        api_wide = api_df.copy()
        api_wide["date"] = pd.to_datetime(
            api_wide["date_utc"], errors="coerce"
        )
        api_wide["pm2_5"] = pd.to_numeric(api_wide["value"], errors="coerce")
        api_wide["pm10"] = np.nan
        api_wide["no2"] = np.nan
        api_wide["so2"] = np.nan
        api_wide["co"] = np.nan
        api_wide["o3"] = np.nan
        api_wide["aqi"] = np.nan
        api_wide["source"] = "api"
        api_wide["parameter"] = "pm2_5"
        api_wide = api_wide[
            [
                "location",
                "city",
                "country",
                "parameter",
                "value",
                "unit",
                "date_utc",
                "latitude",
                "longitude",
                "date",
                "pm2_5",
                "pm10",
                "no2",
                "so2",
                "co",
                "o3",
                "aqi",
                "source",
            ]
        ]

    csv_wide = csv_df.copy()
    csv_wide["date"] = pd.to_datetime(
        csv_wide["date"], errors="coerce", utc=True
    )
    csv_wide["location"] = None
    csv_wide["country"] = OPENAQ_COUNTRY
    csv_wide["unit"] = None
    csv_wide["value"] = None
    csv_wide["date_utc"] = csv_wide["date"]
    csv_wide["latitude"] = None
    csv_wide["longitude"] = None
    csv_wide["parameter"] = "pm2_5"
    csv_wide["source"] = "csv"

    db_wide = db_df.copy()
    db_wide["date"] = pd.to_datetime(
        db_wide["date"], errors="coerce", utc=True
    )
    db_wide["location"] = None
    db_wide["country"] = OPENAQ_COUNTRY
    db_wide["unit"] = None
    db_wide["value"] = None
    db_wide["date_utc"] = db_wide["date"]
    db_wide["latitude"] = None
    db_wide["longitude"] = None
    db_wide["parameter"] = "pm2_5"
    db_wide["source"] = "db"

    unified = pd.concat([api_wide, csv_wide, db_wide], ignore_index=True)
    logger.info("Unified extracted rows: %s", len(unified))
    return unified


def run_extraction(*, dry_run: bool = False) -> pd.DataFrame:
    """Run the complete extraction (API + CSV + SQLite) and merge results."""

    logger = _get_logger()
    ensure_datasets_present()

    api_df = extract_from_openaq(dry_run=dry_run)
    csv_df = extract_from_csv(SAMPLE_CSV_PATH)
    db_df = extract_from_sqlite(SQLITE_DB_PATH)

    merged = _unify_sources(api_df=api_df, csv_df=csv_df, db_df=db_df)
    logger.info("Extraction completed")
    return merged


def df_to_xcom_json(df: pd.DataFrame) -> str:
    """Serialize DataFrame to JSON for Airflow XCom payloads."""

    return df.to_json(orient="records", date_format="iso")


def df_from_xcom_json(payload: str) -> pd.DataFrame:
    """Deserialize JSON payload back to DataFrame."""

    records = json.loads(payload)
    return pd.DataFrame.from_records(records)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(
        description="Extract air quality datasets"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip OpenAQ network calls (CI-safe).",
    )
    parser.add_argument(
        "--out",
        default=str(DATA_DIR / "extracted_unified.csv"),
        help="Write unified extraction output CSV to this path.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""

    args = _parse_args(argv)
    df = run_extraction(dry_run=bool(args.dry_run))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    _get_logger().info("Wrote unified extraction to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
