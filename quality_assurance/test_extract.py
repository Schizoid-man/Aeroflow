"""Unit tests for pipeline.extract."""

from __future__ import annotations

import pandas as pd

from pipeline import extract


def test_openaq_missing_api_key_skips(monkeypatch):
    # v3 extraction requires an API key; missing key should not break the DAG.
    monkeypatch.delenv(extract.OPENAQ_API_KEY_ENV, raising=False)
    df = extract.extract_from_openaq(max_pages=1)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_openaq_v3_locations_and_measurements_parsed(monkeypatch):
    monkeypatch.setenv(extract.OPENAQ_API_KEY_ENV, "test-key")

    class DummyResp:
        def __init__(self, payload):
            self.status_code = 200
            self._payload = payload

        def json(self):
            return self._payload

        @property
        def text(self):
            return "OK"

    class DummySession:
        def get(self, url, *args, **kwargs):
            if str(url).endswith("/locations"):
                return DummyResp(
                    {
                        "results": [
                            {
                                "name": "Loc1",
                                "locality": "Delhi",
                                "country": {"code": "IN"},
                                "coordinates": {
                                    "latitude": 1.0,
                                    "longitude": 2.0,
                                },
                                "sensors": [
                                    {
                                        "id": 123,
                                        "parameter": {"name": "pm25"},
                                    }
                                ],
                            }
                        ]
                    }
                )

            if "/sensors/" in str(url) and str(url).endswith("/measurements"):
                return DummyResp(
                    {
                        "results": [
                            {
                                "value": 12.3,
                                "period": {
                                    "datetimeFrom": {
                                        "utc": "2025-01-01T00:00:00Z"
                                    }
                                },
                                "parameter": {"units": "ug/m3"},
                                "coordinates": {
                                    "latitude": 1.0,
                                    "longitude": 2.0,
                                },
                            }
                        ]
                    }
                )

            raise AssertionError(f"Unexpected URL called: {url}")

    df = extract.extract_from_openaq(session=DummySession(), max_pages=1)
    assert len(df) == 1
    assert df.loc[0, "location"] == "Loc1"
    assert df.loc[0, "city"] == "Delhi"
    assert df.loc[0, "country"] == "IN"
    assert df.loc[0, "parameter"] == extract.OPENAQ_PARAMETER
    assert float(df.loc[0, "value"]) == 12.3
    assert df.loc[0, "unit"] == "ug/m3"
    assert df.loc[0, "date_utc"] == "2025-01-01T00:00:00Z"


def test_csv_loads_correctly():
    df = extract.extract_from_csv(extract.SAMPLE_CSV_PATH)
    assert len(df) > 0
    assert "aqi" in df.columns


def test_db_creates_and_queries(tmp_path):
    db_path = tmp_path / "historical_aqi.db"
    df = extract.extract_from_sqlite(db_path, seed_rows=50, city_count=3, years_back=1)
    assert len(df) > 0
    assert set(["date", "city", "pm2_5", "aqi"]).issubset(df.columns)
