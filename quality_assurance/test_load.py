"""Unit tests for pipeline.load."""

from __future__ import annotations

import pandas as pd

from pipeline import load


def test_table_creation(monkeypatch, tmp_path):
    df_raw = pd.DataFrame(
        [
            {
                "location": "L",
                "city": "C",
                "country": "IN",
                "parameter": "pm2_5",
                "value": 1.0,
                "unit": "ug/m3",
                "date_utc": "2025-01-01T00:00:00Z",
                "latitude": 1.0,
                "longitude": 2.0,
                "source": "csv",
            }
        ]
    )
    df_proc = pd.DataFrame(
        [
            {
                "date": "2025-01-01 00:00:00",
                "city": "C",
                "pm2_5": 10.0,
                "pm10": 20.0,
                "no2": 5.0,
                "so2": 2.0,
                "co": 0.5,
                "o3": 8.0,
                "aqi": 50,
                "aqi_category": "Good",
                "hour_of_day": 0,
                "day_of_week": 0,
                "rolling_avg_pm25_7d": 10.0,
                "pm2_5_norm": 0.0,
                "pm10_norm": 0.0,
                "no2_norm": 0.0,
            }
        ]
    )

    class DummyBegin:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, *args, **kwargs):
            return None

    class DummyEngine:
        def begin(self):
            return DummyBegin()

    eng = DummyEngine()

    monkeypatch.setattr(load, "ensure_tables", lambda *a, **k: None)

    calls = {"to_sql": 0}

    def fake_to_sql(self, *args, **kwargs):
        calls["to_sql"] += 1

    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql, raising=True)

    load.load_raw_and_processed(raw_df=df_raw, processed_df=df_proc, engine=eng)
    assert calls["to_sql"] == 2


def test_row_count_logged(monkeypatch, caplog):
    df = pd.DataFrame([{"a": 1}, {"a": 2}])

    class DummyEngine:
        pass

    called = {"to_sql": False}

    def fake_to_sql(*args, **kwargs):
        called["to_sql"] = True

    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql, raising=True)
    caplog.set_level("INFO")
    rows = load.load_dataframe(engine=DummyEngine(), df=df, table_name="x")
    assert called["to_sql"]
    assert rows == 2
    assert any("Inserted 2 rows" in rec.message for rec in caplog.records)
