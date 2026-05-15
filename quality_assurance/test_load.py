"""Unit tests for pipeline.load."""

from __future__ import annotations

import pandas as pd

from pipeline import load


def _processed_row():
    return {
        "zone_id": "lobby",
        "zone_name": "Lobby",
        "temperature": 22.0,
        "humidity": 48.0,
        "co2": 600.0,
        "occupancy": 10,
        "energy_kw": 7.0,
        "hour_of_day": 10,
        "day_of_week": 0,
        "is_business_hours": True,
        "rolling_avg_energy_7d": 6.8,
        "temperature_norm": 0.4,
        "humidity_norm": 0.5,
        "co2_norm": 0.3,
    }


def test_prepare_processed_for_load_selects_correct_columns():
    df = pd.DataFrame([_processed_row()])
    df["extra_col"] = "should_be_dropped"
    prepared = load.prepare_processed_for_load(df)
    assert "extra_col" not in prepared.columns
    assert set(load.PROCESSED_COLUMNS).issubset(set(prepared.columns))


def test_prepare_processed_raises_on_missing_columns():
    df = pd.DataFrame([{"zone_id": "lobby", "energy_kw": 7.0}])
    try:
        load.prepare_processed_for_load(df)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "missing" in str(exc).lower()


def test_load_dataframe_inserts_rows(monkeypatch):
    df = pd.DataFrame([{"a": 1}, {"a": 2}])

    class DummyEngine:
        pass

    calls = {"to_sql": 0}

    def fake_to_sql(self, *args, **kwargs):
        calls["to_sql"] += 1

    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql, raising=True)
    rows = load.load_dataframe(engine=DummyEngine(), df=df, table_name="processed_sensor_data")
    assert calls["to_sql"] == 1
    assert rows == 2


def test_load_dataframe_empty_is_noop(monkeypatch, caplog):
    calls = {"to_sql": 0}

    def fake_to_sql(self, *args, **kwargs):
        calls["to_sql"] += 1

    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql, raising=True)
    caplog.set_level("INFO")

    class DummyEngine:
        pass

    rows = load.load_dataframe(engine=DummyEngine(), df=pd.DataFrame(), table_name="x")
    assert rows == 0
    assert calls["to_sql"] == 0


def test_load_processed_dataframe(monkeypatch):
    df = pd.DataFrame([_processed_row()])

    calls = {"to_sql": 0}

    def fake_to_sql(self, *args, **kwargs):
        calls["to_sql"] += 1

    monkeypatch.setattr(pd.DataFrame, "to_sql", fake_to_sql, raising=True)

    class DummyEngine:
        pass

    rows = load.load_processed_dataframe(engine=DummyEngine(), df=df)
    assert calls["to_sql"] == 1
    assert rows == 1
