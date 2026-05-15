"""Synthetic smart building IoT publisher.

Generates realistic sensor readings for 5 building zones and publishes
them to a local Mosquitto MQTT broker every PUBLISH_INTERVAL_SEC seconds.

Topic pattern: building/zone/{zone_id}/data
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt


BROKER_HOST: str = os.getenv("MQTT_BROKER_HOST", "localhost")
BROKER_PORT: int = int(os.getenv("MQTT_BROKER_PORT", "1883"))
PUBLISH_INTERVAL_SEC: int = int(os.getenv("PUBLISH_INTERVAL_SEC", "30"))
TOPIC_TMPL: str = "building/zone/{zone_id}/data"

ZONES = [
    {"zone_id": "lobby", "zone_name": "Lobby", "max_occupancy": 30},
    {"zone_id": "floor_1", "zone_name": "Floor 1", "max_occupancy": 50},
    {"zone_id": "floor_2", "zone_name": "Floor 2", "max_occupancy": 50},
    {"zone_id": "conference", "zone_name": "Conference", "max_occupancy": 20},
    {"zone_id": "cafeteria", "zone_name": "Cafeteria", "max_occupancy": 40},
]

TEMP_SETPOINT: float = 21.0


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("iot.publisher")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _is_business_hours(now: datetime) -> bool:
    return now.weekday() < 5 and 9 <= now.hour < 18


def simulate_zone_reading(zone: dict, now: datetime) -> dict:
    max_occ = zone["max_occupancy"]
    business = _is_business_hours(now)

    occupancy_ratio = (0.5 + 0.5 * random.random()) if business else (0.05 * random.random())
    occupancy = max(0, int(max_occ * occupancy_ratio + random.gauss(0, 1)))

    temperature = round(
        TEMP_SETPOINT + occupancy * 0.04 + random.gauss(0, 0.3), 1
    )
    temperature = max(18.0, min(30.0, temperature))

    humidity = round(45.0 + occupancy * 0.08 + random.gauss(0, 1.5), 1)
    humidity = max(30.0, min(70.0, humidity))

    co2 = int(max(400, min(2000, 400 + occupancy * 18 + random.gauss(0, 15))))

    temp_delta = abs(temperature - TEMP_SETPOINT)
    energy_kw = round(
        max(0.5, 2.0 + occupancy * 0.14 + temp_delta * 0.4 + random.gauss(0, 0.15)),
        2,
    )

    return {
        "zone_id": zone["zone_id"],
        "zone_name": zone["zone_name"],
        "temperature": temperature,
        "humidity": humidity,
        "co2": co2,
        "occupancy": occupancy,
        "energy_kw": energy_kw,
        "published_at": now.isoformat(),
    }


def run_publisher() -> None:
    logger = _get_logger()

    client = mqtt.Client()

    # Retry connecting to broker — it may not be up yet on cold start.
    for attempt in range(1, 11):
        try:
            client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
            logger.info("Connected to MQTT broker %s:%s", BROKER_HOST, BROKER_PORT)
            break
        except Exception as exc:
            logger.warning(
                "MQTT connect failed attempt %s/10: %s — retrying in 5s", attempt, exc
            )
            time.sleep(5)
    else:
        raise RuntimeError(
            f"Could not connect to MQTT broker {BROKER_HOST}:{BROKER_PORT} after 10 attempts"
        )

    client.loop_start()
    logger.info(
        "Publishing to %s zones every %ss", len(ZONES), PUBLISH_INTERVAL_SEC
    )

    try:
        while True:
            now = datetime.now(timezone.utc)
            for zone in ZONES:
                reading = simulate_zone_reading(zone, now)
                topic = TOPIC_TMPL.format(zone_id=zone["zone_id"])
                client.publish(topic, json.dumps(reading), qos=1)
                logger.info(
                    "Published zone=%s energy_kw=%.2f occupancy=%d",
                    zone["zone_id"],
                    reading["energy_kw"],
                    reading["occupancy"],
                )
            time.sleep(PUBLISH_INTERVAL_SEC)
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    run_publisher()
