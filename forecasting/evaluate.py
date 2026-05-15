"""Evaluate the saved energy regression model and log metrics.

Outputs:
- logs/model_metrics.txt
- reports/actual_vs_predicted.png
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from forecasting.train import FEATURE_COLS, RANDOM_STATE, TARGET_COL, load_processed_data


LOGGER_NAME: str = "forecasting.evaluate"
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
LOG_DIR: Path = PROJECT_ROOT / "logs"
OUT_DIR: Path = PROJECT_ROOT / "reports"

METRICS_PATH: Path = LOG_DIR / "model_metrics.txt"
PLOT_PATH: Path = OUT_DIR / "actual_vs_predicted.png"


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


def _train_test_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[FEATURE_COLS].copy()
    X["is_business_hours"] = X["is_business_hours"].astype(int)
    y = df[TARGET_COL].astype(float)
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def evaluate_saved_model(*, model_path: str) -> Dict[str, float]:
    logger = _get_logger()
    df = load_processed_data()
    _, X_test, _, y_test = _train_test_split(df)
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R2: {r2:.4f}\n")

    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        table = pd.DataFrame(
            {"feature": FEATURE_COLS, "importance": importances}
        ).sort_values("importance", ascending=False)
        logger.info("Feature importances:\n%s", table.to_string(index=False))

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("Actual Energy (kW)")
    plt.ylabel("Predicted Energy (kW)")
    plt.title("Actual vs Predicted HVAC Energy Consumption")
    lims = [
        float(min(y_test.min(), y_pred.min())),
        float(max(y_test.max(), y_pred.max())),
    ]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

    logger.info("Wrote metrics to %s and plot to %s", METRICS_PATH, PLOT_PATH)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> int:
    from forecasting.train import MODEL_PATH

    evaluate_saved_model(model_path=str(MODEL_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
