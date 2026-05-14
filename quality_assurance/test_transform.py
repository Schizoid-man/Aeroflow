"""Unit tests for pipeline.transform."""

from __future__ import annotations

import pandas as pd

from pipeline.transform import run_transformation


def test_nulls_removed():
    df = pd.DataFrame(
        [
            {"date": "2025-01-01", "city": "X", "pm2_5": 10, "aqi": 20},
            {"date": "2025-01-02", "city": "X", "pm2_5": None, "aqi": 30},
            {"date": "2025-01-03", "city": "X", "pm2_5": 12, "aqi": None},
        ]
    )
    df["pm10"] = 1
    df["no2"] = 1
    df["so2"] = 1
    df["co"] = 1
    df["o3"] = 1
    out = run_transformation(df)
    assert len(out) == 1


def test_aqi_categories_correct():
    df = pd.DataFrame(
        [
            {
                "date": "2025-01-01 00:00:00",
                "city": "X",
                "pm2_5": 10,
                "pm10": 10,
                "no2": 10,
                "so2": 10,
                "co": 1,
                "o3": 10,
                "aqi": 45,
            },
            {
                "date": "2025-01-02 00:00:00",
                "city": "X",
                "pm2_5": 15,
                "pm10": 12,
                "no2": 11,
                "so2": 9,
                "co": 1.1,
                "o3": 12,
                "aqi": 350,
            },
        ]
    )
    out = run_transformation(df)
    cats = list(out["aqi_category"].values)
    assert "Good" in cats
    assert "Hazardous" in cats


def test_normalized_values_in_range():
    df = pd.DataFrame(
        [
            {
                "date": "2025-01-01",
                "city": "X",
                "pm2_5": 10,
                "pm10": 20,
                "no2": 5,
                "so2": 2,
                "co": 0.5,
                "o3": 8,
                "aqi": 50,
            },
            {
                "date": "2025-01-02",
                "city": "X",
                "pm2_5": 20,
                "pm10": 40,
                "no2": 15,
                "so2": 4,
                "co": 1.0,
                "o3": 18,
                "aqi": 120,
            },
        ]
    )
    out = run_transformation(df)
    norm_cols = [c for c in out.columns if c.endswith("_norm")]
    assert norm_cols
    assert (out[norm_cols] >= 0).all().all()
    assert (out[norm_cols] <= 1).all().all()


def test_rolling_avg_computed():
    rows = []
    for i in range(10):
        rows.append(
            {
                "date": f"2025-01-{i + 1:02d}",
                "city": "X",
                "pm2_5": 10 + i,
                "pm10": 20,
                "no2": 5,
                "so2": 2,
                "co": 0.5,
                "o3": 8,
                "aqi": 60,
            }
        )
    df = pd.DataFrame(rows)
    out = run_transformation(df)
    assert "rolling_avg_pm25_7d" in out.columns
    assert out.loc[7:, "rolling_avg_pm25_7d"].isna().sum() == 0
