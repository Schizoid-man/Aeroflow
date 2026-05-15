/*
SCHEMA: Smart Building IoT Pipeline (PostgreSQL)

raw_sensor_buffer
  - Continuously populated by mqtt-consumer
  - Cleared by Airflow ETL after each successful load

processed_sensor_data
  - Written by Airflow ETL after transformation
  - Source for ML training

Relationships:
  - None enforced (ETL processes buffer independently).
*/

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
);

CREATE TABLE IF NOT EXISTS processed_sensor_data (
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
  temperature_norm      FLOAT,
  humidity_norm         FLOAT,
  co2_norm              FLOAT,
  processed_at          TIMESTAMPTZ DEFAULT NOW()
);
