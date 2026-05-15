"""MQTT consumer: subscribes to all zone topics and writes to PostgreSQL.

Continuously subscribes to building/zone/+/data and inserts each message
into the raw_sensor_buffer table. Handles broker reconnection automatically
via paho loop_forever() and retries DB inserts with exponential backoff.
"""

from __future__ import annotations

import json
import logging
import os
import time

import paho.mqtt.client as mqtt
import psycopg2
from psycopg2.extras import execute_values


BROKER_HOST: str = os.getenv("MQTT_BROKER_HOST", "localhost")
BROKER_PORT: int = int(os.getenv("MQTT_BROKER_PORT", "1883"))
SUBSCRIBE_TOPIC: str = "building/zone/+/data"

DB_HOST: str = os.getenv("DB_HOST", "localhost")
DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
DB_USER: str = os.getenv("DB_USER", "airflow")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "airflow")
DB_NAME: str = os.getenv("DB_NAME", "airflow_iot")

INSERT_SQL = """
INSERT INTO raw_sensor_buffer
    (zone_id, zone_name, temperature, humidity, co2, occupancy, energy_kw, published_at)
VALUES %s
"""

CREATE_BUFFER_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS raw_sensor_buffer (
    id           SERIAL PRIMARY KEY,
    zone_id      TEXT NOT NULL,
    zone_name    TEXT NOT NULL,
    temperature  FLOAT,
    humidity     FLOAT,
    co2          FLOAT,
    occupancy    INT,
    energy_kw    FLOAT,
    published_at TIMESTAMPTZ,
    ingested_at  TIMESTAMPTZ DEFAULT NOW()
)
"""


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("iot.consumer")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _db_connect() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME,
    )


def _wait_for_db(max_attempts: int = 30) -> None:
    logger = _get_logger()
    for attempt in range(1, max_attempts + 1):
        try:
            conn = _db_connect()
            conn.close()
            logger.info("PostgreSQL is ready")
            return
        except Exception as exc:
            logger.warning(
                "DB not ready attempt %s/%s: %s — retrying in 3s", attempt, max_attempts, exc
            )
            time.sleep(3)
    raise RuntimeError("PostgreSQL not available after startup wait")


def _ensure_buffer_table() -> None:
    logger = _get_logger()
    conn = _db_connect()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_BUFFER_TABLE_SQL)
        logger.info("raw_sensor_buffer table ready")
    finally:
        conn.close()


def _insert_with_retry(payload: dict, max_retries: int = 3) -> None:
    logger = _get_logger()
    row = (
        payload.get("zone_id"),
        payload.get("zone_name"),
        payload.get("temperature"),
        payload.get("humidity"),
        payload.get("co2"),
        payload.get("occupancy"),
        payload.get("energy_kw"),
        payload.get("published_at"),
    )
    for attempt in range(1, max_retries + 1):
        try:
            conn = _db_connect()
            with conn:
                with conn.cursor() as cur:
                    execute_values(cur, INSERT_SQL, [row])
            conn.close()
            return
        except Exception as exc:
            logger.warning(
                "DB insert failed attempt %s/%s: %s", attempt, max_retries, exc
            )
            time.sleep(2 ** attempt)
    logger.error(
        "DB insert failed after %s retries for zone=%s",
        max_retries,
        payload.get("zone_id"),
    )


def _on_message(client: mqtt.Client, userdata: None, msg: mqtt.MQTTMessage) -> None:
    logger = _get_logger()
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        _insert_with_retry(payload)
        logger.info(
            "Stored zone=%s energy_kw=%s",
            payload.get("zone_id"),
            payload.get("energy_kw"),
        )
    except Exception as exc:
        logger.error("Failed to process MQTT message: %s", exc)


def run_consumer() -> None:
    logger = _get_logger()
    _wait_for_db()
    _ensure_buffer_table()

    client = mqtt.Client()
    client.on_message = _on_message

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

    client.subscribe(SUBSCRIBE_TOPIC, qos=1)
    logger.info("Subscribed to %s", SUBSCRIBE_TOPIC)
    client.loop_forever()


if __name__ == "__main__":
    run_consumer()
