"""Train an energy consumption regression model from processed sensor data.

Model: RandomForestRegressor
Features: temperature, humidity, co2, occupancy, hour_of_day, day_of_week,
          is_business_hours, rolling_avg_energy_7d, temperature_norm,
          humidity_norm, co2_norm
Target: energy_kw
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
MODEL_PATH: Path = MODEL_DIR / "energy_model.pkl"

FEATURE_COLS: List[str] = [
    "temperature",
    "humidity",
    "co2",
    "occupancy",
    "hour_of_day",
    "day_of_week",
    "is_business_hours",
    "rolling_avg_energy_7d",
    "temperature_norm",
    "humidity_norm",
    "co2_norm",
]
TARGET_COL: str = "energy_kw"

MIN_TRAINING_ROWS: int = 100
RANDOM_STATE: int = 42
N_ESTIMATORS: int = 100


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


def load_processed_data(table_name: str = "processed_sensor_data") -> pd.DataFrame:
    cfg = get_db_config()
    engine = create_engine(to_sqlalchemy_url(cfg), pool_pre_ping=True)
    return pd.read_sql_table(table_name, engine)


def build_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for training: {missing}")
    X = df[FEATURE_COLS].copy()
    X["is_business_hours"] = X["is_business_hours"].astype(int)
    y = df[TARGET_COL].astype(float)
    return X, y


def train_and_save_model(df: pd.DataFrame, model_path: Path = MODEL_PATH) -> Path:
    logger = _get_logger()

    if len(df) < MIN_TRAINING_ROWS:
        logger.warning(
            "Insufficient data for training: %s rows (minimum %s). Skipping.",
            len(df),
            MIN_TRAINING_ROWS,
        )
        return model_path

    X, y = build_features_and_target(df)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info("Saved model to %s", model_path)
    return model_path


def main() -> int:
    df = load_processed_data()
    train_and_save_model(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
