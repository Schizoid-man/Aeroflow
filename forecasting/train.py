"""Train an AQI regression model from processed data.

Model:
- RandomForestRegressor

Features:
- pm2_5_norm, pm10_norm, no2_norm, hour_of_day, day_of_week,
  rolling_avg_pm25_7d
Target:
- aqi
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

from settings.db_config import get_db_config, to_sqlalchemy_url


LOGGER_NAME: str = "forecasting.train"
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
MODEL_DIR: Path = Path(__file__).resolve().parent / "model"
MODEL_PATH: Path = MODEL_DIR / "aqi_model.pkl"

FEATURE_COLS: List[str] = [
    "pm2_5_norm",
    "pm10_norm",
    "no2_norm",
    "hour_of_day",
    "day_of_week",
    "rolling_avg_pm25_7d",
]
TARGET_COL: str = "aqi"

RANDOM_STATE: int = 42
N_ESTIMATORS: int = 100


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


def load_processed_data(
    table_name: str = "processed_air_quality",
) -> pd.DataFrame:
    """Load processed data from PostgreSQL."""

    cfg = get_db_config()
    engine = create_engine(to_sqlalchemy_url(cfg), pool_pre_ping=True)
    return pd.read_sql_table(table_name, engine)


def build_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build the feature matrix X and target y."""

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for training: {missing}")
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].astype(float)
    return X, y


def train_and_save_model(
    df: pd.DataFrame,
    model_path: Path = MODEL_PATH,
) -> Path:
    """Train a RandomForestRegressor model and save to disk."""

    logger = _get_logger()
    X, y = build_features_and_target(df)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Saved model to %s", model_path)
    return model_path


def main() -> int:
    """CLI entrypoint to train on DB data."""

    df = load_processed_data()
    train_and_save_model(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
