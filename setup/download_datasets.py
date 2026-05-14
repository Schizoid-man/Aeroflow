"""Dataset download and preparation script.

This script supports:
1) Kaggle dataset download (optional; uses kaggle CLI if configured)
2) UCI Air Quality dataset download via HTTP with retry
3) Ensures a small sample CSV exists for offline/CI usage

Notes:
- Kaggle data is not bundled in this repository.
- If Kaggle CLI is missing or credentials are not configured, the script logs a
  clear message and continues.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


LOGGER_NAME: str = "setup.download_datasets"
UCI_URL: str = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
UCI_ZIP_NAME: str = "air_quality_uci.zip"
UCI_EXTRACTED_CSV: str = "AirQualityUCI.csv"

RETRIES: int = 3
RETRY_DELAY_SEC: int = 5
HTTP_TIMEOUT_SEC: int = 30

KAGGLE_DATASET: str = "rohanrao/air-quality-data-in-india"


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


def _download_with_retry(url: str, dest: Path) -> None:
    """Download a URL to disk with retry logic."""

    logger = _get_logger()
    last_exc: Optional[Exception] = None
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.get(url, timeout=HTTP_TIMEOUT_SEC)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "Download failed attempt=%s/%s: %s", attempt, RETRIES, exc
            )
            if attempt < RETRIES:
                time.sleep(RETRY_DELAY_SEC)
    raise RuntimeError(f"Failed to download after retries: {last_exc}")


def _ensure_sample_csv(data_dir: Path) -> None:
    """Create a small synthetic sample CSV for offline/CI runs."""

    logger = _get_logger()
    sample_path = data_dir / "sample_aqi_india.csv"
    if sample_path.exists():
        return

    rows = [
        {
            "date": "2025-01-01 08:00:00",
            "city": "Bengaluru",
            "pm2_5": 42.0,
            "pm10": 88.0,
            "no2": 24.0,
            "so2": 8.0,
            "co": 0.8,
            "o3": 20.0,
            "aqi": 65,
        },
        {
            "date": "2025-01-02 08:00:00",
            "city": "Bengaluru",
            "pm2_5": 55.0,
            "pm10": 95.0,
            "no2": 30.0,
            "so2": 10.0,
            "co": 1.0,
            "o3": 25.0,
            "aqi": 90,
        },
        {
            "date": "2025-01-01 09:00:00",
            "city": "Delhi",
            "pm2_5": 160.0,
            "pm10": 260.0,
            "no2": 60.0,
            "so2": 18.0,
            "co": 1.8,
            "o3": 35.0,
            "aqi": 210,
        },
        {
            "date": "2025-01-02 09:00:00",
            "city": "Delhi",
            "pm2_5": 220.0,
            "pm10": 320.0,
            "no2": 75.0,
            "so2": 20.0,
            "co": 2.2,
            "o3": 40.0,
            "aqi": 320,
        },
        {
            "date": "2025-01-03 09:00:00",
            "city": "Delhi",
            "pm2_5": 35.0,
            "pm10": 70.0,
            "no2": 22.0,
            "so2": 7.0,
            "co": 0.6,
            "o3": 18.0,
            "aqi": 45,
        },
    ]
    pd.DataFrame(rows).to_csv(sample_path, index=False)
    logger.info("Created synthetic sample CSV at %s", sample_path)


def download_kaggle_dataset(data_dir: Path) -> None:
    """Attempt Kaggle download using CLI; safe no-op if not configured."""

    logger = _get_logger()
    kaggle_cmd = shutil.which("kaggle")
    if not kaggle_cmd:
        logger.info(
            "Kaggle CLI not found. Install/configure kaggle or use sample CSV."
        )
        return

    try:
        result = subprocess.run(
            [
                kaggle_cmd,
                "datasets",
                "download",
                "-d",
                KAGGLE_DATASET,
                "-p",
                str(data_dir),
                "--unzip",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        logger.info(
            "Kaggle CLI not configured. Falling back to sample CSV."
        )
        return

    if result.returncode != 0:
        logger.info(
            "Kaggle download skipped/failed (exit=%s). stderr=%s",
            result.returncode,
            (result.stderr or "").strip()[:300],
        )
        return

    logger.info("Kaggle dataset downloaded/unzipped into %s", data_dir)


def download_uci_dataset(data_dir: Path) -> None:
    """Download and extract UCI Air Quality dataset."""

    logger = _get_logger()
    out_zip = data_dir / UCI_ZIP_NAME
    out_csv = data_dir / "uci_air_quality.csv"
    if out_csv.exists():
        return

    logger.info("Downloading UCI Air Quality dataset")
    _download_with_retry(UCI_URL, out_zip)
    with zipfile.ZipFile(out_zip, "r") as zf:
        if UCI_EXTRACTED_CSV in zf.namelist():
            zf.extract(UCI_EXTRACTED_CSV, data_dir)
            (data_dir / UCI_EXTRACTED_CSV).rename(out_csv)
            logger.info("Saved UCI CSV to %s", out_csv)
        else:
            logger.warning("UCI zip did not contain %s", UCI_EXTRACTED_CSV)


def main(*, project_root: Optional[Path] = None) -> None:
    """Run downloads and prepare datasets."""

    root = project_root or Path(__file__).resolve().parents[1]
    data_dir = root / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)

    _ensure_sample_csv(data_dir)

    try:
        download_uci_dataset(data_dir)
    except Exception as exc:  # noqa: BLE001
        _get_logger().warning("UCI download failed: %s", exc)

    try:
        download_kaggle_dataset(data_dir)
    except Exception as exc:  # noqa: BLE001
        _get_logger().warning("Kaggle download failed: %s", exc)


if __name__ == "__main__":
    main()
