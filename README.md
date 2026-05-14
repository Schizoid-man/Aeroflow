# AeroPulse — Real-Time Air Quality Forecasting Pipeline

> An end-to-end data engineering and ML pipeline that ingests live and historical air quality data, transforms it through a validated ETL process, warehouses it in PostgreSQL, and trains a regression model to forecast AQI — orchestrated with Apache Airflow and validated through GitHub Actions CI/CD.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Repository Structure](#4-repository-structure)
5. [Data Sources](#5-data-sources)
6. [Pipeline Stages](#6-pipeline-stages)
7. [Machine Learning](#7-machine-learning)
8. [Airflow DAGs](#8-airflow-dags)
9. [Database Schema](#9-database-schema)
10. [Setup & Installation](#10-setup--installation)
11. [Running the Pipeline](#11-running-the-pipeline)
12. [Testing](#12-testing)
13. [CI/CD Pipeline](#13-cicd-pipeline)
14. [Configuration Reference](#14-configuration-reference)
15. [Future Scope](#15-future-scope)

---

## 1. Project Overview

AeroPulse is a production-style data engineering system for real-time air quality monitoring and forecasting, focused on Indian cities. The system:

- **Extracts** PM2.5 measurements from three heterogeneous sources: the OpenAQ v3 REST API (live), a local CSV dataset, and a synthetic SQLite historical database.
- **Transforms** raw data through cleaning, deduplication, feature engineering (AQI categorisation, time features, 7-day rolling averages), and MinMax normalization.
- **Loads** both raw and processed records into a PostgreSQL data warehouse.
- **Trains** a `RandomForestRegressor` on processed features to predict AQI scores.
- **Evaluates** the model and generates metrics (`MAE`, `RMSE`, `R²`) and a scatter plot of actual vs. predicted values.
- **Orchestrates** all stages via Apache Airflow DAGs with daily ETL and weekly ML training schedules.
- **Validates** every commit through a three-stage GitHub Actions CI/CD workflow.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Data Sources                       │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │  OpenAQ v3  │  │  CSV File   │  │  SQLite (synth) │ │
│  │    API      │  │  (local)    │  │  historical DB  │ │
│  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘ │
└─────────┼────────────────┼─────────────────┼───────────┘
          └────────────────┼─────────────────┘
                           │ Extract & Unify
                    ┌──────▼──────┐
                    │   Extract   │  pipeline/extract.py
                    └──────┬──────┘
                           │ Validate raw shape
                    ┌──────▼──────┐
                    │  Transform  │  pipeline/transform.py
                    │  + Validate │  (clean, engineer, normalise)
                    └──────┬──────┘
                           │
               ┌───────────┴───────────┐
               │                       │
        ┌──────▼──────┐         ┌──────▼──────┐
        │  Load Raw   │         │ Load Proc.  │  pipeline/load.py
        │ raw_air_    │         │ processed_  │
        │ quality     │         │ air_quality │
        └──────┬──────┘         └──────┬──────┘
               └───────────┬───────────┘
                           │  PostgreSQL 15
                    ┌──────▼──────┐
                    │  ML Train   │  forecasting/train.py
                    │  (RF Model) │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Evaluate   │  forecasting/evaluate.py
                    │  + Report   │  (MAE / RMSE / R² / plot)
                    └─────────────┘
```

**Orchestration:** Both pipelines are managed as Airflow DAGs.  
**CI/CD:** GitHub Actions runs lint → test → pipeline validation → simulated deploy on every push.

---

## 3. Tech Stack

| Component | Tool / Library | Version |
|---|---|---|
| Orchestration | Apache Airflow | 2.8.0 |
| Data Warehouse | PostgreSQL | 15 |
| Data Processing | pandas | 2.1.0 / 2.2.2 |
| Numerical Computing | NumPy | 1.26.0 |
| DB Connectivity | SQLAlchemy + psycopg2-binary | 2.0.30 / 2.9.9 |
| ML Framework | scikit-learn | 1.3.0 / 1.5.0 |
| Model Serialisation | joblib | 1.4.2 |
| Visualisation | matplotlib | 3.7.0 / 3.8.4 |
| HTTP Client | requests | 2.31.0 |
| Config Management | python-dotenv | 1.0.0 |
| Testing | pytest + pytest-cov | 7.4.0 / 4.1.0 |
| Linting | flake8 | 6.1.0 |
| Containerisation | Docker + Docker Compose | — |
| CI/CD | GitHub Actions | — |

> Version variants reflect Python 3.10 / 3.12 compatibility pins in `requirements.txt`.

---

## 4. Repository Structure

```
AeroPulse/
│
├── pipeline/                   # Core ETL modules
│   ├── extract.py              # Ingest from OpenAQ API, CSV, SQLite
│   ├── transform.py            # Clean, engineer features, normalise
│   └── load.py                 # Write raw + processed data to PostgreSQL
│
├── forecasting/                # ML training & evaluation
│   ├── train.py                # RandomForestRegressor training
│   ├── evaluate.py             # Metrics (MAE/RMSE/R²) + scatter plot
│   └── model/                  # Saved model artefact (gitignored)
│
├── orchestration/              # Airflow DAG definitions
│   ├── etl_pipeline_dag.py     # Daily ETL DAG (extract→transform→load)
│   └── ml_training_dag.py      # Weekly ML DAG (train→evaluate→save)
│
├── quality_assurance/          # pytest unit tests
│   ├── test_extract.py
│   ├── test_transform.py
│   └── test_load.py
│
├── schema/                     # PostgreSQL DDL + analytics queries
│   ├── create_tables.sql
│   └── queries.sql
│
├── settings/                   # Database connection utilities
│   └── db_config.py            # Reads .env → DBConfig dataclass
│
├── setup/                      # Dataset bootstrapping
│   └── download_datasets.py    # Auto-downloads sample data if missing
│
├── datasets/                   # Local data files (CSV + SQLite DB)
│   ├── sample_aqi_india.csv
│   └── historical_aqi.db
│
├── .github/workflows/
│   └── ci_cd_pipeline.yml      # GitHub Actions: lint → test → validate → deploy
│
├── logs/                       # Runtime logs (gitignored)
├── reports/                    # Generated plots (gitignored)
│
├── Dockerfile.airflow          # Custom Airflow image
├── docker-compose.yml          # Airflow webserver + scheduler + PostgreSQL
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Test discovery configuration
└── .flake8                     # Linting configuration
```

---

## 5. Data Sources

| Source | Type | Description |
|---|---|---|
| **OpenAQ v3 API** | Live / REST | Fetches PM2.5 sensor measurements for India (`iso=IN`). Paginated, with retry logic and API key authentication. Up to 25 locations × 5 sensors × 2 pages per run. |
| **CSV (`sample_aqi_india.csv`)** | Static / Local | A sample dataset with columns: `date`, `city`, `pm2_5`, `pm10`, `no2`, `so2`, `co`, `o3`, `aqi`. |
| **SQLite (`historical_aqi.db`)** | Synthetic / Local | Auto-seeded with 500 rows of realistic historical data across 10 Indian cities, spanning 2 years. Seeded deterministically with `numpy.random.default_rng(42)`. |

All three sources are unified into a single wide DataFrame with a `source` column (`'api'`, `'csv'`, `'db'`) before transformation.

---

## 6. Pipeline Stages

### Extract (`pipeline/extract.py`)

- Calls `run_extraction()` to pull from all three sources concurrently.
- Handles missing API keys gracefully (logs a warning, returns an empty DataFrame).
- Retries OpenAQ requests up to 3 times with error escalation.
- Serialises output to JSON for Airflow XCom transfer (`df_to_xcom_json` / `df_from_xcom_json`).
- CLI usage: `python pipeline/extract.py [--dry-run] [--out PATH]`

### Transform (`pipeline/transform.py`)

Performs the following in sequence:

1. **Type coercion** — parses `date`, `aqi`, `pm2_5` with error handling.
2. **Deduplication** — drops rows with the same `(date, city, parameter)` triple.
3. **Range filtering** — drops rows with `aqi` outside `[0, 500]`.
4. **Feature engineering:**
   - `aqi_category` — maps AQI score to: `Good` / `Moderate` / `Unhealthy for Sensitive` / `Unhealthy` / `Very Unhealthy` / `Hazardous`.
   - `hour_of_day`, `day_of_week` — extracted from timestamp.
   - `rolling_avg_pm25_7d` — 7-day rolling mean of PM2.5, grouped by city.
5. **Normalisation** — MinMax scaling of all pollutant columns (`pm2_5`, `pm10`, `no2`, `so2`, `co`, `o3`) to `[0, 1]`.
6. **Validation** — asserts required columns exist, no nulls in critical fields (`aqi`, `city`, `date`), and normalised values are within `[0, 1]`. Results logged to `logs/validation_log.txt`.

CLI usage: `python pipeline/transform.py [--input PATH] [--output PATH]`

### Load (`pipeline/load.py`)

- Creates PostgreSQL tables via `schema/create_tables.sql` (uses an advisory transaction lock to prevent race conditions in parallel Airflow tasks).
- Appends data using `pandas.DataFrame.to_sql()` with `method="multi"` and `chunksize=500`.
- Loads into two tables: `raw_air_quality` and `processed_air_quality`.

---

## 7. Machine Learning

### Model

- **Algorithm:** `RandomForestRegressor` (scikit-learn)
- **Estimators:** 100 trees, `random_state=42`
- **Train/Test Split:** 80/20, deterministic (`random_state=42`)

### Features

| Feature | Description |
|---|---|
| `pm2_5_norm` | MinMax-normalised PM2.5 concentration |
| `pm10_norm` | MinMax-normalised PM10 concentration |
| `no2_norm` | MinMax-normalised NO₂ concentration |
| `hour_of_day` | Hour extracted from measurement timestamp |
| `day_of_week` | Day of week (0 = Monday) |
| `rolling_avg_pm25_7d` | 7-day rolling mean PM2.5, per city |

**Target:** `aqi` (integer AQI score)

### Training

```bash
# Train from PostgreSQL processed table
python forecasting/train.py
```

Model saved to `forecasting/model/aqi_model.pkl`.

### Evaluation

```bash
python forecasting/evaluate.py
```

Outputs:
- `logs/model_metrics.txt` — MAE, RMSE, R² scores
- `reports/actual_vs_predicted.png` — scatter plot with identity line

---

## 8. Airflow DAGs

### ETL DAG — `air_quality_etl_pipeline`

**Schedule:** `@daily`

```
task_extract → task_validate_raw → task_transform → task_load_raw  ─┐
                                                                      → task_notify
                                                  → task_load_processed ─┘
```

| Task | Description |
|---|---|
| `task_extract` | Runs full extraction (API + CSV + SQLite), pushes JSON to XCom |
| `task_validate_raw` | Asserts required columns and non-empty dataset |
| `task_transform` | Runs transformation, pushes processed JSON to XCom |
| `task_load_raw` | Writes raw rows to `raw_air_quality` |
| `task_load_processed` | Writes processed rows to `processed_air_quality` |
| `task_notify` | Logs success summary with row counts |

### ML Training DAG — `ml_training_pipeline`

**Schedule:** `@weekly`

```
load_data_from_db → preprocess_features → train_model → evaluate_model → save_model
```

| Task | Description |
|---|---|
| `load_data_from_db` | Reads `processed_air_quality` from PostgreSQL |
| `preprocess_features` | Builds feature matrix X and target y |
| `train_model` | Trains `RandomForestRegressor`, saves `.pkl` |
| `evaluate_model` | Computes MAE/RMSE/R², saves metrics and plot |
| `save_model` | Confirms model artefact path |

---

## 9. Database Schema

### `raw_air_quality`

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL PK | Auto-incrementing primary key |
| `location` | TEXT | Monitoring station name |
| `city` | TEXT | City name |
| `country` | TEXT | ISO country code |
| `parameter` | TEXT | Pollutant name (e.g. `pm25`) |
| `value` | FLOAT | Raw measurement value |
| `unit` | TEXT | Unit of measurement (e.g. `ug/m3`) |
| `date_utc` | TIMESTAMP | Measurement timestamp (UTC) |
| `latitude` | FLOAT | Station latitude |
| `longitude` | FLOAT | Station longitude |
| `source` | TEXT | Data origin: `api`, `csv`, or `db` |
| `ingested_at` | TIMESTAMP | Row insertion time (auto) |

### `processed_air_quality`

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL PK | Auto-incrementing primary key |
| `date` | TIMESTAMP | Measurement timestamp |
| `city` | TEXT | City name |
| `pm2_5` … `o3` | FLOAT | Pollutant concentrations |
| `aqi` | INT | Air Quality Index score |
| `aqi_category` | TEXT | Category label (Good → Hazardous) |
| `hour_of_day` | INT | Hour of day (0–23) |
| `day_of_week` | INT | Day of week (0–6) |
| `rolling_avg_pm25_7d` | FLOAT | 7-day rolling PM2.5 mean |
| `pm2_5_norm`, `pm10_norm`, `no2_norm` | FLOAT | MinMax-normalised pollutants |
| `processed_at` | TIMESTAMP | Row insertion time (auto) |

---

## 10. Setup & Installation

### Prerequisites

- Python 3.10+ (3.12 supported)
- Docker + Docker Compose (for Airflow orchestration)
- An [OpenAQ v3 API key](https://docs.openaq.org/) (optional — pipeline degrades gracefully without one)

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-org>/AeroPulse.git
cd AeroPulse

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> **Note:** `apache-airflow` is excluded from CI installs automatically. For local script runs without Docker, Airflow is not required.

### 2. Create the `.env` file

```env
# PostgreSQL connection (use 'localhost' when running scripts outside Docker)
DB_HOST=postgres
DB_PORT=5432
DB_USER=airflow
DB_PASSWORD=airflow
DB_NAME=airflow_aqi

# Airflow Fernet key (required for Docker Compose)
# Generate with: python -c "import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"
AIRFLOW_FERNET_KEY=<your-fernet-key>

# OpenAQ v3 API key (optional — extraction falls back to CSV+SQLite if not set)
OPENAQ_API_KEY=<your-openaq-api-key>
```

### 3. Start Docker services (Airflow + PostgreSQL)

```bash
docker-compose up --build -d
```

### 4. Initialise Airflow

```bash
# Run inside the webserver container
docker-compose exec airflow-webserver airflow db migrate

docker-compose exec airflow-webserver airflow users create \
  --username admin --firstname Admin --lastname User \
  --role Admin --email admin@example.com --password admin
```

Airflow UI: **http://localhost:8080**

---

## 11. Running the Pipeline

### Run individual stages locally

```bash
# Extract (dry-run skips OpenAQ network calls)
python pipeline/extract.py --dry-run --out datasets/extracted.csv

# Transform
python pipeline/transform.py --input datasets/sample_aqi_india.csv --output datasets/transformed.csv

# Train the model (requires PostgreSQL with processed data)
python forecasting/train.py

# Evaluate the model
python forecasting/evaluate.py
```

### Trigger the Airflow DAG

**From the UI:** Enable `air_quality_etl_pipeline` → click **Trigger DAG**.

**From the CLI (inside the container):**

```bash
docker-compose exec airflow-webserver airflow dags trigger air_quality_etl_pipeline
docker-compose exec airflow-webserver airflow dags trigger ml_training_pipeline
```

### Generate analysis diagrams

Use the transformed dataset to create quick visual summaries (stored in `reports/`).

```bash
python reports/generate_diagrams.py
```

---

## 12. Testing

Unit tests live in `quality_assurance/` and are discovered via `pytest.ini`.

```bash
# Run all tests
pytest -q

# Run with coverage report
pytest -q --cov=pipeline --cov=forecasting --cov-report=term-missing
```

### Test coverage

| Module | Tests |
|---|---|
| `pipeline/extract.py` | Missing API key handling, OpenAQ v3 response parsing, CSV loading, SQLite seeding |
| `pipeline/transform.py` | Null dropping, deduplication, AQI category mapping, normalisation bounds |
| `pipeline/load.py` | Column validation, prepared DataFrame shape |

---

## 13. CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci_cd_pipeline.yml`) runs on every push to `main` or `feature/*` branches, and on pull requests to `main`.

```
┌─────────────────────┐
│   lint_and_test     │  ← Always runs
│  ─ flake8 lint      │
│  ─ pytest + cov     │
│  ─ upload coverage  │
└──────────┬──────────┘
           │ on success
┌──────────▼──────────┐
│  validate_pipeline  │  ← Always runs (after lint_and_test)
│  ─ dry-run extract  │
│  ─ transform + col  │
│    assertion        │
└──────────┬──────────┘
           │ on success (main branch only)
┌──────────▼──────────┐
│    deploy_etl       │  ← main branch only
│  ─ simulated deploy │
└─────────────────────┘
```

| Job | Description |
|---|---|
| **lint_and_test** | Installs CI requirements (Airflow excluded), runs `flake8` on `pipeline/` and `forecasting/`, runs `pytest` with XML coverage upload. |
| **validate_pipeline** | Runs a dry-run extraction (no network calls), transforms sample CSV, asserts expected output columns are present. |
| **deploy_etl** | Simulates deployment. In production this step would trigger the Airflow REST API, push a Docker image, or deploy via SSH. |

---

## 14. Configuration Reference

| Environment Variable | Required | Default | Description |
|---|---|---|---|
| `DB_HOST` | Yes | — | PostgreSQL host (`postgres` in Docker, `localhost` locally) |
| `DB_PORT` | Yes | — | PostgreSQL port (usually `5432`) |
| `DB_USER` | Yes | — | Database username |
| `DB_PASSWORD` | Yes | — | Database password |
| `DB_NAME` | Yes | — | Database name |
| `OPENAQ_API_KEY` | No | — | OpenAQ v3 API key. If unset, API extraction is skipped gracefully. |
| `AIRFLOW_FERNET_KEY` | Yes (Docker) | — | Fernet key for Airflow connection encryption |

---

## 15. Future Scope

| Enhancement | Description |
|---|---|
| **Kafka Streaming** | Replace batch extraction with real-time event streaming using Apache Kafka |
| **Cloud Data Warehouse** | Migrate PostgreSQL to Snowflake or BigQuery for scalable analytics |
| **ML Experiment Tracking** | Integrate MLflow for model versioning and experiment comparison |
| **Dashboard** | Build a Grafana or Streamlit dashboard over the PostgreSQL warehouse |
| **Multi-pollutant Models** | Extend to predict PM10, NO₂, O₃ in addition to AQI |
| **Alerting** | Trigger notifications when AQI exceeds hazardous thresholds |

---

<div align="center">
  <sub>Built as part of CIE634 — Data Management for Machine Learning | M.Tech AI | Ramaiah Institute of Technology</sub>
</div>
