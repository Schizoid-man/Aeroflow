/*
SCHEMA DIAGRAM (PostgreSQL)

raw_air_quality
  - id SERIAL PRIMARY KEY
  - location TEXT
  - city TEXT
  - country TEXT
  - parameter TEXT
  - value FLOAT
  - unit TEXT
  - date_utc TIMESTAMP
  - latitude FLOAT
  - longitude FLOAT
  - source TEXT
  - ingested_at TIMESTAMP DEFAULT NOW()

processed_air_quality
  - id SERIAL PRIMARY KEY
  - date TIMESTAMP
  - city TEXT
  - pm2_5 FLOAT
  - pm10 FLOAT
  - no2 FLOAT
  - so2 FLOAT
  - co FLOAT
  - o3 FLOAT
  - aqi INT
  - aqi_category TEXT
  - hour_of_day INT
  - day_of_week INT
  - rolling_avg_pm25_7d FLOAT
  - pm2_5_norm FLOAT
  - pm10_norm FLOAT
  - no2_norm FLOAT
  - processed_at TIMESTAMP DEFAULT NOW()

Relationships:
  - None enforced (ETL appends both tables independently).
*/

CREATE TABLE IF NOT EXISTS raw_air_quality (
  id SERIAL PRIMARY KEY,
  location TEXT,
  city TEXT,
  country TEXT,
  parameter TEXT,
  value FLOAT,
  unit TEXT,
  date_utc TIMESTAMP,
  latitude FLOAT,
  longitude FLOAT,
  source TEXT,
  ingested_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS processed_air_quality (
  id SERIAL PRIMARY KEY,
  date TIMESTAMP,
  city TEXT,
  pm2_5 FLOAT,
  pm10 FLOAT,
  no2 FLOAT,
  so2 FLOAT,
  co FLOAT,
  o3 FLOAT,
  aqi INT,
  aqi_category TEXT,
  hour_of_day INT,
  day_of_week INT,
  rolling_avg_pm25_7d FLOAT,
  pm2_5_norm FLOAT,
  pm10_norm FLOAT,
  no2_norm FLOAT,
  processed_at TIMESTAMP DEFAULT NOW()
);
