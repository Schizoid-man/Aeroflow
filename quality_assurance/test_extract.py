"""Unit tests for pipeline.extract."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipeline import extract


def test_extract_from_csv_valid(tmp_path):
    csv = tmp_path / "sensors.csv"
    csv.write_text(
        "zone_id,zone_name,temperature,humidity,co2,occupancy,energy_kw,published_at\n"
        "lobby,Lobby,22.0,45.0,600,10,5.5,2025-01-01T09:00:00Z\n"
    )
    df = extract.extract_from_csv(csv)
    assert len(df) == 1
    assert df.loc[0, "zone_id"] == "lobby"
    assert float(df.loc[0, "energy_kw"]) == 5.5


def test_extract_from_csv_missing_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("zone_id,temperature\nlobby,22.0\n")
    with pytest.raises(ValueError, match="missing columns"):
        extract.extract_from_csv(csv)


def test_extract_from_csv_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract.extract_from_csv(tmp_path / "nonexistent.csv")


def test_run_extraction_csv_fallback(tmp_path, monkeypatch):
    sample_csv = tmp_path / "sample_building_sensors.csv"
    sample_csv.write_text(
        "zone_id,zone_name,temperature,humidity,co2,occupancy,energy_kw,published_at\n"
        "floor_1,Floor 1,21.5,48.0,550,15,7.2,2025-01-01T10:00:00Z\n"
    )
    monkeypatch.setattr(extract, "SAMPLE_CSV_PATH", sample_csv)
    df = extract.run_extraction(use_csv_fallback=True)
    assert len(df) == 1
    assert "energy_kw" in df.columns


def test_df_xcom_roundtrip():
    df = pd.DataFrame(
        [{"zone_id": "lobby", "energy_kw": 5.5, "temperature": 22.0}]
    )
    payload = extract.df_to_xcom_json(df)
    df2 = extract.df_from_xcom_json(payload)
    assert list(df2["zone_id"]) == ["lobby"]
    assert float(df2.loc[0, "energy_kw"]) == 5.5
