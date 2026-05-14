"""Transform and validate extracted air quality data.

This module performs:
1) Cleaning (null handling, de-duplication, typing)
2) Feature engineering (AQI category, time features, rolling averages)
3) Normalization of pollutant columns using MinMax scaling
4) Data validation with results logged to logs/validation_log.txt
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

AQI_MIN_VALID: int = 0
AQI_MAX_VALID: int = 500
AQI_MAX_FILTER: int = 1000

POLLUTANT_COLS: List[str] = ["pm2_5", "pm10", "no2", "so2", "co", "o3"]
CRITICAL_COLS: List[str] = ["aqi", "city", "date"]
NORM_SUFFIX: str = "_norm"


def _get_logger() -> logging.Logger:
    """Return a file logger that writes to logs/validation_log.txt."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.FileHandler(VALIDATION_LOG_PATH, encoding="utf-8")
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def aqi_to_category(aqi: float) -> str:
    """Map AQI numeric values to standard AQI categories.

    Args:
        aqi: AQI value.

    Returns:
        AQI category name.
    """

    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def run_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, transform, normalize, and validate air quality data.

    Args:
        df: Unified extracted DataFrame.

    Returns:
        Transformed DataFrame.

    Raises:
        ValueError: If validation checks fail.
    """

    logger = _get_logger()
    df_work = df.copy()

    for col in ["city", "location", "country", "parameter", "source"]:
        if col in df_work.columns:
            df_work[col] = df_work[col].astype(str).str.strip()

    if "date" not in df_work.columns:
        if "date_utc" in df_work.columns:
            df_work["date"] = pd.to_datetime(
                df_work["date_utc"], errors="coerce"
            )
        else:
            raise ValueError(
                "Missing date column (expected 'date' or 'date_utc')"
            )
    else:
        df_work["date"] = pd.to_datetime(df_work["date"], errors="coerce")

    df_work["aqi"] = pd.to_numeric(df_work.get("aqi"), errors="coerce")
    df_work["pm2_5"] = pd.to_numeric(df_work.get("pm2_5"), errors="coerce")
    df_work = df_work.dropna(subset=["aqi", "pm2_5"])

    if "parameter" not in df_work.columns:
        df_work["parameter"] = "pm2_5"

    df_work = df_work.drop_duplicates(subset=["date", "city", "parameter"])
    df_work = df_work[
        (df_work["aqi"] > 0) & (df_work["aqi"] <= AQI_MAX_FILTER)
    ].copy()
    df_work = df_work[
        (df_work["aqi"] >= AQI_MIN_VALID) & (df_work["aqi"] <= AQI_MAX_VALID)
    ]

    df_work["aqi_category"] = df_work["aqi"].apply(aqi_to_category)
    df_work["hour_of_day"] = df_work["date"].dt.hour.astype(int)
    df_work["day_of_week"] = df_work["date"].dt.dayofweek.astype(int)

    df_work = df_work.sort_values(["city", "date"]).reset_index(drop=True)
    df_work["rolling_avg_pm25_7d"] = (
        df_work.groupby("city")["pm2_5"]
        .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
        .astype(float)
    )

    for col in POLLUTANT_COLS:
        if col in df_work.columns:
            df_work[col] = pd.to_numeric(df_work[col], errors="coerce")

    scaler = MinMaxScaler()
    cols_for_scaling = [c for c in POLLUTANT_COLS if c in df_work.columns]
    if cols_for_scaling:
        scaled = scaler.fit_transform(df_work[cols_for_scaling].fillna(0.0))
        for idx, col in enumerate(cols_for_scaling):
            df_work[f"{col}{NORM_SUFFIX}"] = scaled[:, idx]

    _validate(df_work, logger)
    logger.info("Transformation successful; rows=%s", len(df_work))
    return df_work


def _validate(df: pd.DataFrame, logger: logging.Logger) -> None:
    """Run required data validation assertions.

    Args:
        df: Transformed dataset.
        logger: Logger to write validation results.

    Raises:
        ValueError: If any validation condition fails.
    """

    expected_cols = set(
        [
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
        ]
    )
    missing = sorted(list(expected_cols - set(df.columns)))
    if missing:
        msg = f"Missing expected columns after transform: {missing}"
        logger.error(msg)
        raise ValueError(msg)

    nulls = {
        col: int(df[col].isna().sum()) for col in CRITICAL_COLS if col in df
    }
    logger.info("Null counts (critical): %s", nulls)
    if any(v > 0 for v in nulls.values()):
        msg = f"Nulls remain in critical columns: {nulls}"
        logger.error(msg)
        raise ValueError(msg)

    if not df["aqi"].between(AQI_MIN_VALID, AQI_MAX_VALID).all():
        msg = "AQI values out of valid range (0-500)"
        logger.error(msg)
        raise ValueError(msg)

    norm_cols = [c for c in df.columns if c.endswith(NORM_SUFFIX)]
    if norm_cols:
        min_val = float(df[norm_cols].min().min())
        max_val = float(df[norm_cols].max().max())
        logger.info(
            "Normalized value range: min=%s max=%s", min_val, max_val
        )
        if min_val < 0.0 - 1e-9 or max_val > 1.0 + 1e-9:
            msg = "Normalized columns contain values outside [0, 1]"
            logger.error(msg)
            raise ValueError(msg)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(
        description="Transform air quality dataset"
    )
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "datasets" / "sample_aqi_india.csv"),
        help="Input CSV path (expects wide pollutant columns).",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "datasets" / "transformed_output.csv"),
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
    logging.getLogger(LOGGER_NAME).info(
        "Wrote transformed dataset to %s", out_path
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
