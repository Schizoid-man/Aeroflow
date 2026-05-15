"""Dataset preparation script for the smart building IoT pipeline.

Ensures the sample building sensor CSV exists for offline and CI usage.
Generates a larger synthetic dataset for local development if needed.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd


LOGGER_NAME: str = "setup.download_datasets"

ZONES = [
    {"zone_id": "lobby", "zone_name": "Lobby", "max_occupancy": 30},
    {"zone_id": "floor_1", "zone_name": "Floor 1", "max_occupancy": 50},
    {"zone_id": "floor_2", "zone_name": "Floor 2", "max_occupancy": 50},
    {"zone_id": "conference", "zone_name": "Conference", "max_occupancy": 20},
    {"zone_id": "cafeteria", "zone_name": "Cafeteria", "max_occupancy": 40},
]

SYNTHETIC_ROWS_PER_ZONE: int = 200


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


def _is_business_hours(dt: datetime) -> bool:
    return dt.weekday() < 5 and 9 <= dt.hour < 18


def _generate_synthetic_rows(zone: dict, n: int, seed: int = 42) -> List[dict]:
    rng = random.Random(seed + hash(zone["zone_id"]) % 1000)
    rows = []
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)

    for i in range(n):
        ts = start + timedelta(hours=i)
        business = _is_business_hours(ts)
        max_occ = zone["max_occupancy"]

        occupancy_ratio = (0.5 + 0.5 * rng.random()) if business else (0.05 * rng.random())
        occupancy = max(0, int(max_occ * occupancy_ratio + rng.gauss(0, 1)))

        temperature = round(21.0 + occupancy * 0.04 + rng.gauss(0, 0.3), 1)
        temperature = max(18.0, min(30.0, temperature))

        humidity = round(45.0 + occupancy * 0.08 + rng.gauss(0, 1.5), 1)
        humidity = max(30.0, min(70.0, humidity))

        co2 = int(max(400, min(2000, 400 + occupancy * 18 + rng.gauss(0, 15))))

        temp_delta = abs(temperature - 21.0)
        energy_kw = round(
            max(0.5, 2.0 + occupancy * 0.14 + temp_delta * 0.4 + rng.gauss(0, 0.15)), 2
        )

        rows.append({
            "zone_id": zone["zone_id"],
            "zone_name": zone["zone_name"],
            "temperature": temperature,
            "humidity": humidity,
            "co2": co2,
            "occupancy": occupancy,
            "energy_kw": energy_kw,
            "published_at": ts.isoformat(),
        })
    return rows


def _ensure_sample_csv(data_dir: Path) -> None:
    logger = _get_logger()
    sample_path = data_dir / "sample_building_sensors.csv"
    if sample_path.exists():
        logger.info("Sample CSV already exists at %s", sample_path)
        return

    all_rows = []
    for zone in ZONES:
        all_rows.extend(_generate_synthetic_rows(zone, SYNTHETIC_ROWS_PER_ZONE))

    df = pd.DataFrame(all_rows)
    df = df.sort_values("published_at").reset_index(drop=True)
    df.to_csv(sample_path, index=False)
    logger.info(
        "Created synthetic sample CSV at %s (%s rows)", sample_path, len(df)
    )


def main(*, project_root: Optional[Path] = None) -> None:
    root = project_root or Path(__file__).resolve().parents[1]
    data_dir = root / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)
    _ensure_sample_csv(data_dir)


if __name__ == "__main__":
    main()
