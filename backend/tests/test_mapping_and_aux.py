"""Regression tests for the Panchayat-disambiguation fix and the auxiliary
(soil / NDVI / land-cover) advisory rules."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["GROQ_API_KEY"] = "x"
os.environ["SARVAM_API_KEY"] = "x"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import advisory as adv  # noqa: E402
import routes_data  # noqa: E402
from data_store import Store  # noqa: E402
from weather_lookup import WeatherContext  # noqa: E402


@pytest.fixture
def dup_repo(tmp_path):
    """Two Panchayats share the name ALURU in different districts."""
    l2 = tmp_path / "outputs" / "layer2"
    l2.mkdir(parents=True)
    rows = [
        dict(state="ANDHRA PRADESH", district="Guntur", block_id=1, block_name="PONNUR", panchayat_id=200620,
             panchayat_name="ALURU", n_days=1, rainfall_mean_mm=3.0, rainfall_min_mm=0, rainfall_max_mm=4,
             n_wet_days=1, n_cells=1, mapping_method="direct_grid", fallback_distance_m=None),
        dict(state="KARNATAKA", district="Udupi", block_id=2, block_name="KUNDAPURA", panchayat_id=220447,
             panchayat_name="ALURU", n_days=1, rainfall_mean_mm=12.0, rainfall_min_mm=0, rainfall_max_mm=50,
             n_wet_days=1, n_cells=2, mapping_method="direct_grid", fallback_distance_m=None),
    ]
    pd.DataFrame(rows).to_csv(l2 / "panchayat_summary.csv", index=False)
    pd.DataFrame([dict(panchayat_id=200620, lat=16.30, lon=80.40),
                  dict(panchayat_id=220447, lat=13.74, lon=74.73)]).to_csv(l2 / "panchayat_index.csv", index=False)
    lat = 11.0 + 0.05 * np.arange(120)
    lon = 74.0 + 0.05 * np.arange(120)
    rain = np.full((1, 120, 120), 50.49, "float32")
    np.savez(tmp_path / "outputs" / "prediction_test.npz", dates=np.array(["2022-07-10"]),
             latitude=lat, longitude=lon, rainfall_mm=rain)
    m = tmp_path / "outputs" / "metrics"
    m.mkdir()
    json.dump({"rows": {"A": {"label": "A"}, "D": {"label": "D"}}, "selection": {"row": "D", "label": "D"},
               "results": {"val": {"A": dict(MAE=9, RMSE=10, corr=.3), "D": dict(MAE=8, RMSE=9, corr=.4)},
                           "test": {"A": dict(MAE=10, RMSE=11, corr=.3), "D": dict(MAE=8, RMSE=9, corr=.4)}}},
              open(m / "ablation.json", "w"))
    json.dump({"meta": {"channels": ["imd_rain"]}}, open(m / "metrics.json", "w"))
    return tmp_path


@pytest.fixture
def dup_client(dup_repo, monkeypatch):
    async def fake(lat, lon, date, client, dem=None):
        return WeatherContext(temperature_c=27.5, humidity_pct=80, elevation_m=120.0,
                              temperature_source="t", humidity_source="h", elevation_source="e")
    monkeypatch.setattr(routes_data, "fetch_weather", fake)
    app = FastAPI()
    routes_data.register(app, dup_repo)
    return TestClient(app)


def test_store_matches_and_find(dup_repo):
    st = Store(dup_repo)
    assert len(st.matches("ALURU")) == 2
    assert len(st.matches("ALURU", district="Udupi")) == 1
    assert int(st.matches("ALURU", district="Udupi").iloc[0].panchayat_id) == 220447
    # find() stays single-valued but no longer silently ignores ambiguity resolution
    assert int(st.find("ALURU", district="Udupi").panchayat_id) == 220447


def test_ambiguous_name_only_auth_is_409(dup_client):
    r = dup_client.post("/auth/", json={"panchayat_name": "ALURU", "date": "2022-07-10"})
    assert r.status_code == 409
    body = r.json()["detail"]
    assert body["error"].startswith("2 panchayats share the name")
    assert {c["district"] for c in body["candidates"]} == {"Guntur", "Udupi"}


def test_disambiguated_auth_returns_value(dup_client):
    r = dup_client.post("/auth/", json={"panchayat_name": "ALURU", "district": "Udupi",
                                        "block_name": "KUNDAPURA", "state": "KARNATAKA", "date": "2022-07-10"})
    assert r.status_code == 200 and r.json()["prediction"]["rainfall_mm"] == 50.49


def test_ambiguous_geocode_is_409(dup_client):
    r = dup_client.get("/api/geocode", params={"panchayat_name": "ALURU"})
    assert r.status_code == 409
    assert len(r.json()["detail"]["candidates"]) == 2


def test_health_reports_count_on_cold_start(dup_client):
    # the health route must load the store itself, not rely on a prior call
    r = dup_client.get("/")
    assert r.status_code == 200 and r.json()["panchayats_loaded"] == 2


def test_stage_synonyms_normalised():
    assert adv.PanchayatInput(panchayat="X", crop="wheat", stage="ripening", rainfall_mm=1).stage == "maturity"
    assert adv.PanchayatInput(panchayat="X", crop="wheat", stage="general", rainfall_mm=1).stage is None


def test_aux_vegetation_rule_fires_on_low_ndvi():
    i = adv.PanchayatInput(panchayat="X", crop="rice", stage="flowering", rainfall_mm=1,
                           aux=adv.AuxContext(ndvi=adv.AuxNDVI(value=0.12, month="2022-09")))
    t = adv.evaluate(i)
    r6 = next(r for r in t.rules if r.rule_id == "R6_VEGETATION")
    assert r6.fired and r6.severity == "medium"          # low NDVI + low rain


def test_aux_soil_drainage_rule_fires():
    i = adv.PanchayatInput(panchayat="X", crop="wheat", stage="general", rainfall_mm=70,
                           aux=adv.AuxContext(soil=adv.AuxSoil(clay_g_per_kg=420.0)))
    t = adv.evaluate(i)
    r7 = next(r for r in t.rules if r.rule_id == "R7_SOIL_DRAINAGE")
    assert r7.fired and r7.severity == "high"            # heavy rain over clay soil


def test_aux_landcover_rule_fires_on_water():
    i = adv.PanchayatInput(panchayat="X", crop="wheat", stage="general", rainfall_mm=5,
                           aux=adv.AuxContext(lulc=adv.AuxLULC(fractions={"water_fraction": 0.9}, dominant="water")))
    t = adv.evaluate(i)
    r8 = next(r for r in t.rules if r.rule_id == "R8_LANDCOVER")
    assert r8.fired


def test_aux_rules_not_evaluable_without_aux():
    i = adv.PanchayatInput(panchayat="X", crop="wheat", stage="general", rainfall_mm=5)
    t = adv.evaluate(i)
    for rid in ("R6_VEGETATION", "R7_SOIL_DRAINAGE", "R8_LANDCOVER"):
        r = next(x for x in t.rules if x.rule_id == rid)
        assert r.evaluable is False and r.fired is False
