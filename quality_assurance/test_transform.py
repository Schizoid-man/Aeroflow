"""Unit tests for pipeline.transform."""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.transform import run_transformation


def _make_df(rows):
    return pd.DataFrame(rows)


def _base_row(overrides=None):
    row = {
        "zone_id": "floor_1",
        "zone_name": "Floor 1",
        "temperature": 22.0,
        "humidity": 48.0,
        "co2": 600,
        "occupancy": 10,
        "energy_kw": 7.0,
        "published_at": "2025-01-01T10:00:00Z",
    }
    if overrides:
        row.update(overrides)
    return row


def test_nulls_in_energy_removed():
    df = _make_df([
        _base_row(),
        _base_row({"energy_kw": None, "published_at": "2025-01-01T11:00:00Z"}),
    ])
    out = run_transformation(df)
    assert len(out) == 1
    assert out.iloc[0]["zone_id"] == "floor_1"


def test_out_of_range_temperature_removed():
    df = _make_df([
        _base_row(),
        _base_row({"temperature": 999.0, "published_at": "2025-01-01T11:00:00Z"}),
    ])
    out = run_transformation(df)
    assert len(out) == 1


def test_business_hours_flag():
    df = _make_df([
        _base_row({"published_at": "2025-01-06T10:00:00Z"}),  # Mon 10am → business
        _base_row({"published_at": "2025-01-06T22:00:00Z"}),  # Mon 10pm → not business
    ])
    out = run_transformation(df)
    flags = list(out.sort_values("published_at")["is_business_hours"])
    assert flags[0] is True
    assert flags[1] is False


def test_normalized_values_in_range():
    df = _make_df([
        _base_row({"temperature": 20.0, "humidity": 40.0, "co2": 500}),
        _base_row({"temperature": 26.0, "humidity": 60.0, "co2": 900,
                   "published_at": "2025-01-01T11:00:00Z"}),
    ])
    out = run_transformation(df)
    norm_cols = [c for c in out.columns if c.endswith("_norm")]
    assert norm_cols
    assert (out[norm_cols] >= 0).all().all()
    assert (out[norm_cols] <= 1).all().all()


def test_rolling_avg_energy_computed():
    rows = [
        _base_row({
            "energy_kw": 5.0 + i,
            "published_at": f"2025-01-{i + 1:02d}T10:00:00Z",
        })
        for i in range(10)
    ]
    out = run_transformation(_make_df(rows))
    assert "rolling_avg_energy_7d" in out.columns
    assert out["rolling_avg_energy_7d"].notna().all()


def test_time_features_added():
    df = _make_df([_base_row({"published_at": "2025-01-06T14:30:00Z"})])
    out = run_transformation(df)
    assert "hour_of_day" in out.columns
    assert "day_of_week" in out.columns
    assert int(out.iloc[0]["hour_of_day"]) == 14
    assert int(out.iloc[0]["day_of_week"]) == 0  # Monday
