# IoT MQTT Smart Building Pipeline — Design Spec
**Date:** 2026-05-14
**Status:** Approved

## Overview

Pivot the Aeroflow AQI pipeline to a synthetic smart building IoT pipeline. The OpenAQ REST API is replaced by a local MQTT broker receiving synthetic sensor data from a continuously-running publisher service. The ETL architecture (extract → transform → load → PostgreSQL → RandomForest → Airflow) is preserved.

---

## Architecture

Five Docker Compose services:

| Service | Image | Role |
|---|---|---|
| `mosquitto` | eclipse-mosquitto | MQTT broker on port 1883 |
| `iot-publisher` | custom Python | Generates synthetic sensor data, publishes to MQTT every 30s |
| `mqtt-consumer` | custom Python | Subscribes to all zone topics, inserts into `raw_sensor_buffer` |
| `postgres` | postgres:15 | Data warehouse |
| `airflow` | apache/airflow:2.8.0 | ETL + ML orchestration |

```
iot-publisher → mosquitto → mqtt-consumer → raw_sensor_buffer (PostgreSQL)
                                                      ↓
                                             Airflow ETL DAG (hourly)
                                                      ↓
                                          processed_sensor_data (PostgreSQL)
                                                      ↓
                                             Airflow ML DAG (weekly)
```

---

## Sensors, Zones & MQTT Topics

**Zones (5):** `lobby`, `floor_1`, `floor_2`, `conference`, `cafeteria`

**Publish interval:** 30 seconds per zone

**Topic pattern:** `building/zone/{zone_id}/data`

**Payload (JSON):**
```json
{
  "zone_id": "floor_1",
  "zone_name": "Floor 1",
  "temperature": 22.4,
  "humidity": 48.2,
  "co2": 612,
  "occupancy": 12,
  "energy_kw": 8.3,
  "published_at": "2026-05-14T10:30:00Z"
}
```

**Synthetic data patterns:**
- Temperature: 18–30°C with Gaussian noise; higher during business hours
- Humidity: 30–70% with slow drift
- CO2: 400–1500 ppm; correlated with occupancy
- Occupancy: 0–50 per zone; follows weekday 9am–6pm profile, near-zero nights/weekends
- Energy (kW): correlated with occupancy + temperature delta from setpoint; this is the ML target

The consumer subscribes to `building/zone/+/data`.

---

## Database Schema

```sql
CREATE TABLE raw_sensor_buffer (
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
);

CREATE TABLE processed_sensor_data (
    id                    SERIAL PRIMARY KEY,
    zone_id               TEXT NOT NULL,
    zone_name             TEXT NOT NULL,
    temperature           FLOAT,
    humidity              FLOAT,
    co2                   FLOAT,
    occupancy             INT,
    energy_kw             FLOAT,
    hour_of_day           INT,
    day_of_week           INT,
    is_business_hours     BOOLEAN,
    rolling_avg_energy_7d FLOAT,
    temp_norm             FLOAT,
    humidity_norm         FLOAT,
    co2_norm              FLOAT,
    processed_at          TIMESTAMPTZ DEFAULT NOW()
);
```

---

## Airflow ETL DAG (hourly)

Tasks in order:

1. **extract** — `SELECT * FROM raw_sensor_buffer WHERE ingested_at >= [run start]`; passes rows via XCom
2. **validate** — range checks: temp 0–60°C, humidity 0–100%, co2 400–5000 ppm, energy_kw ≥ 0; logs anomalies, drops invalid rows
3. **transform** — adds `hour_of_day`, `day_of_week`, `is_business_hours`; computes `rolling_avg_energy_7d` per zone; MinMax-normalises `temperature`, `humidity`, `co2`
4. **load** — bulk-insert into `processed_sensor_data` with PostgreSQL advisory lock
5. **clear_buffer** — `DELETE FROM raw_sensor_buffer WHERE ingested_at < [run cutoff]`
6. **notify** — logs row count and per-zone summary

---

## Airflow ML DAG (weekly)

1. **load_data** — read `processed_sensor_data`
2. **train** — RandomForestRegressor; features: `temperature`, `humidity`, `co2`, `occupancy`, `hour_of_day`, `day_of_week`, `is_business_hours`, `rolling_avg_energy_7d`, `temp_norm`, `humidity_norm`, `co2_norm`; target: `energy_kw`
3. **evaluate** — MAE, RMSE, R²; saves metrics to `logs/`
4. **save_model** — serialise with joblib to `models/`

---

## New Files

| Path | Purpose |
|---|---|
| `iot/publisher.py` | Synthetic data generator + MQTT publisher |
| `iot/consumer.py` | MQTT subscriber + PostgreSQL writer |
| `mosquitto/mosquitto.conf` | Mosquitto broker config (allow anonymous, port 1883) |

## Modified Files

| Path | Change |
|---|---|
| `pipeline/extract.py` | Replace OpenAQ HTTP calls with PostgreSQL buffer read |
| `pipeline/transform.py` | Replace AQI features with building sensor features |
| `pipeline/load.py` | Update table name to `processed_sensor_data` |
| `schema/create_tables.sql` | Replace AQI tables with new schema |
| `settings/db_config.py` | Remove `OPENAQ_API_KEY`; add `MQTT_BROKER_HOST`, `MQTT_BROKER_PORT` |
| `docker-compose.yml` | Add `mosquitto`, `iot-publisher`, `mqtt-consumer` services |
| `forecasting/train.py` | Update feature list and target variable |
| `forecasting/evaluate.py` | Update metric labels |
| `orchestration/etl_pipeline_dag.py` | Update task logic and schedule to `@hourly` |
| `orchestration/ml_training_dag.py` | Update feature references |
| `quality_assurance/test_extract.py` | Test buffer read instead of API call |
| `quality_assurance/test_transform.py` | Test building sensor feature engineering |
| `quality_assurance/test_load.py` | Update table references |
| `setup/download_datasets.py` | Replace UCI/Kaggle downloads with synthetic CSV seeder |
| `datasets/sample_building_sensors.csv` | Replace `sample_aqi_india.csv` |
| `.github/workflows/ci_cd_pipeline.yml` | Remove `--dry-run` OpenAQ flag; update env var names |
| `README.md` | Full rewrite for new domain |

---

## Environment Variables

```env
# PostgreSQL (unchanged)
DB_HOST=postgres
DB_PORT=5432
DB_USER=airflow
DB_PASSWORD=airflow
DB_NAME=airflow_iot

# MQTT
MQTT_BROKER_HOST=mosquitto
MQTT_BROKER_PORT=1883

# Airflow
AIRFLOW_FERNET_KEY=<base64>
```

`OPENAQ_API_KEY` is removed entirely.

---

## Error Handling

- **Consumer DB failure**: consumer retries insert with exponential backoff (3 attempts); logs error and continues subscribing
- **Consumer MQTT disconnect**: paho-mqtt `loop_forever()` handles automatic reconnection
- **Airflow extract returns empty**: DAG short-circuits after validate with a warning log; does not fail the run
- **ML training with insufficient data**: train task checks minimum row count (100 rows); skips and logs if below threshold
