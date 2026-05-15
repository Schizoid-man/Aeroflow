"""Transform and validate extracted smart building sensor data.

Steps:
1) Cleaning: null handling, de-duplication, type coercion
2) Feature engineering: time features, is_business_hours, rolling energy average
3) Normalization: MinMax scaling for temperature, humidity, co2
4) Validation: range checks and column assertions logged to logs/validation_log.txt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


LOGGER_NAME: str = "pipeline.transform"
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
LOG_DIR: Path = PROJECT_ROOT / "logs"
VALIDATION_LOG_PATH: Path = LOG_DIR / "validation_log.txt"

SENSOR_COLS: List[str] = ["temperature", "humidity", "co2"]
NORM_SUFFIX: str = "_norm"
CRITICAL_COLS: List[str] = ["zone_id", "published_at", "energy_kw"]

TEMP_MIN: float = 0.0
TEMP_MAX: float = 60.0
HUMIDITY_MIN: float = 0.0
HUMIDITY_MAX: float = 100.0
CO2_MIN: float = 300.0
CO2_MAX: float = 5000.0
ENERGY_MIN: float = 0.0


def _get_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.FileHandler(VALIDATION_LOG_PATH, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def run_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, engineer features, normalize, and validate sensor data.

    Args:
        df: Raw sensor DataFrame from extract step.

    Returns:
        Transformed DataFrame ready for loading into processed_sensor_data.

    Raises:
        ValueError: If post-transform validation fails.
    """

    logger = _get_logger()
    df_work = df.copy()

    for col in ["zone_id", "zone_name"]:
        if col in df_work.columns:
            df_work[col] = df_work[col].astype(str).str.strip()

    if "published_at" not in df_work.columns:
        raise ValueError("Missing required column: published_at")
    df_work["published_at"] = pd.to_datetime(df_work["published_at"], errors="coerce", utc=True)

    for col in ["temperature", "humidity", "co2", "energy_kw"]:
        df_work[col] = pd.to_numeric(df_work.get(col), errors="coerce")
    df_work["occupancy"] = (
        pd.to_numeric(df_work.get("occupancy"), errors="coerce").fillna(0).astype(int)
    )

    df_work = df_work.dropna(subset=["zone_id", "published_at", "energy_kw"])

    df_work = df_work[
        df_work["temperature"].between(TEMP_MIN, TEMP_MAX, inclusive="both")
        | df_work["temperature"].isna()
    ]
    df_work = df_work[
        df_work["humidity"].between(HUMIDITY_MIN, HUMIDITY_MAX, inclusive="both")
        | df_work["humidity"].isna()
    ]
    df_work = df_work[
        df_work["co2"].between(CO2_MIN, CO2_MAX, inclusive="both")
        | df_work["co2"].isna()
    ]
    df_work = df_work[df_work["energy_kw"] >= ENERGY_MIN]

    df_work = df_work.drop_duplicates(subset=["zone_id", "published_at"])

    df_work["hour_of_day"] = df_work["published_at"].dt.hour.astype(int)
    df_work["day_of_week"] = df_work["published_at"].dt.dayofweek.astype(int)
    df_work["is_business_hours"] = (
        (df_work["day_of_week"] < 5) & df_work["hour_of_day"].between(9, 17)
    )

    df_work = df_work.sort_values(["zone_id", "published_at"]).reset_index(drop=True)
    df_work["rolling_avg_energy_7d"] = (
        df_work.groupby("zone_id")["energy_kw"]
        .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
        .astype(float)
    )

    scaler = MinMaxScaler()
    scale_cols = [c for c in SENSOR_COLS if c in df_work.columns]
    if scale_cols:
        scaled = scaler.fit_transform(df_work[scale_cols].fillna(0.0))
        for idx, col in enumerate(scale_cols):
            df_work[f"{col}{NORM_SUFFIX}"] = scaled[:, idx]

    _validate(df_work, logger)
    logger.info("Transformation successful; rows=%s", len(df_work))
    return df_work


def _validate(df: pd.DataFrame, logger: logging.Logger) -> None:
    expected_cols = {
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
    }
    missing = sorted(expected_cols - set(df.columns))
    if missing:
        msg = f"Missing expected columns after transform: {missing}"
        logger.error(msg)
        raise ValueError(msg)

    nulls = {col: int(df[col].isna().sum()) for col in CRITICAL_COLS if col in df}
    logger.info("Null counts (critical): %s", nulls)
    if any(v > 0 for v in nulls.values()):
        msg = f"Nulls remain in critical columns: {nulls}"
        logger.error(msg)
        raise ValueError(msg)

    if not (df["energy_kw"] >= ENERGY_MIN).all():
        msg = "energy_kw contains negative values"
        logger.error(msg)
        raise ValueError(msg)

    norm_cols = [c for c in df.columns if c.endswith(NORM_SUFFIX)]
    if norm_cols:
        min_val = float(df[norm_cols].min().min())
        max_val = float(df[norm_cols].max().max())
        logger.info("Normalized range: min=%s max=%s", min_val, max_val)
        if min_val < -1e-9 or max_val > 1.0 + 1e-9:
            msg = "Normalized columns contain values outside [0, 1]"
            logger.error(msg)
            raise ValueError(msg)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transform smart building sensor dataset")
    parser.add_argument(
        "--input",
        default=str(
            Path(__file__).resolve().parents[1] / "datasets" / "sample_building_sensors.csv"
        ),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "datasets" / "transformed_output.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    df_in = pd.read_csv(args.input)
    df_out = run_transformation(df_in)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    logging.getLogger(LOGGER_NAME).info("Wrote transformed dataset to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
