import os, sys, json, pathlib
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["GROQ_API_KEY"] = "x"; os.environ["SARVAM_API_KEY"] = "x"
import numpy as np, pandas as pd, pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import routes_data
from weather_lookup import WeatherContext


@pytest.fixture
def repo(tmp_path):
    l2 = tmp_path / "outputs" / "layer2"; l2.mkdir(parents=True)
    pd.DataFrame([
        dict(state="KARNATAKA", district="Dakshina Kannada", block_id=1, block_name="MANGALURU", panchayat_id=11,
             panchayat_name="ADYAR", n_days=122, rainfall_mean_mm=9.0, rainfall_min_mm=0, rainfall_max_mm=50, n_wet_days=90,
             n_cells=4, mapping_method="area_weighted", fallback_distance_m=None),
        dict(state="KARNATAKA", district="Dakshina Kannada", block_id=1, block_name="MANGALURU", panchayat_id=12,
             panchayat_name="BALA", n_days=122, rainfall_mean_mm=7.0, rainfall_min_mm=0, rainfall_max_mm=40, n_wet_days=80,
             n_cells=1, mapping_method="nearest_fallback", fallback_distance_m=8000.0),
        dict(state="KERALA", district="Nowhere", block_id=2, block_name="X", panchayat_id=13, panchayat_name="LOST",
             n_days=122, rainfall_mean_mm=1.0, rainfall_min_mm=0, rainfall_max_mm=2, n_wet_days=3,
             n_cells=0, mapping_method="unmapped", fallback_distance_m=None),
        dict(state="KERALA", district="Nowhere", block_id=2, block_name="X", panchayat_id=14, panchayat_name="NOCOORD",
             n_days=122, rainfall_mean_mm=2.0, rainfall_min_mm=0, rainfall_max_mm=5, n_wet_days=9,
             n_cells=2, mapping_method="direct_grid", fallback_distance_m=None),
    ]).to_csv(l2 / "panchayat_summary.csv", index=False)
    pd.DataFrame([dict(panchayat_id=11, lat=12.83, lon=74.88), dict(panchayat_id=12, lat=12.9, lon=74.9)]).to_csv(l2 / "panchayat_index.csv", index=False)
    lat = 12.025 + 0.05 * np.arange(40); lon = 74.025 + 0.05 * np.arange(40)
    rain = np.full((1, 40, 40), 21.0, "float32"); rain[0, 16, 17] = 33.3       # cell nearest (12.83, 74.88)
    np.savez(tmp_path / "outputs" / "prediction_test.npz", dates=np.array(["2022-07-10"]), latitude=lat, longitude=lon, rainfall_mm=rain)
    m = tmp_path / "outputs" / "metrics"; m.mkdir()
    json.dump({"rows": {"A": {"label": "Bilinear IMD baseline"}, "D": {"label": "U-Net D"}},
               "selection": {"row": "D", "label": "U-Net D"},
               "results": {"val": {"A": dict(MAE=9, RMSE=10, corr=.3), "D": dict(MAE=8, RMSE=9, corr=.4)},
                           "test": {"A": dict(MAE=10, RMSE=11, corr=.3), "D": dict(MAE=8, RMSE=9, corr=.4)}}}, open(m / "ablation.json", "w"))
    json.dump({"meta": {"channels": ["imd_rain"], "reference": "ref"}}, open(m / "metrics.json", "w"))
    return tmp_path


@pytest.fixture
def client(repo, monkeypatch):
    async def fake(lat, lon, date, client, dem=None):
        return WeatherContext(temperature_c=27.5, humidity_pct=80, elevation_m=120.0, temperature_source="t", humidity_source="h", elevation_source="e")
    monkeypatch.setattr(routes_data, "fetch_weather", fake)
    app = FastAPI(); routes_data.register(app, repo)
    return TestClient(app)


def test_search_real_fields_and_unmapped_hidden(client):
    r = client.get("/api/panchayats", params={"q": "adyar"}).json()
    assert len(r) == 1 and r[0]["mapping_method"] == "area_weighted" and r[0]["n_cells"] == 4
    assert r[0]["lat"] == 12.83 and r[0]["location_precision"] == "exact"
    assert client.get("/api/panchayats", params={"q": "lost"}).json() == []          # unmapped hidden
    b = client.get("/api/panchayats", params={"q": "bala"}).json()[0]
    assert b["mapping_method"] == "nearest_fallback" and b["fallback_distance_m"] == 8000.0   # not hardcoded direct_grid/1


def test_multi_token_search(client):
    assert [x["panchayat_name"] for x in client.get("/api/panchayats", params={"q": "mangaluru karnataka"}).json()] == ["ADYAR", "BALA"]


def test_geocode(client):
    assert client.get("/api/geocode", params={"panchayat_name": "ADYAR"}).json()["lat"] == 12.83
    assert client.get("/api/geocode", params={"panchayat_name": "NOCOORD"}).status_code == 404
    assert client.get("/api/geocode", params={"panchayat_name": "NOPE"}).status_code == 404


def test_auth_uses_real_model_grid_and_real_weather(client):
    r = client.post("/auth/", json={"panchayat_name": "ADYAR", "district": "Dakshina Kannada", "date": "2022-07-10"}).json()
    assert r["prediction"] == {"rainfall_mm": 33.3, "temperature_c": 27.5, "humidity_pct": 80, "elevation_m": 120.0}
    assert r["model_status"] == "ready" and "prediction grid" in r["sources"]["rainfall"]


def test_auth_no_data_is_an_error_not_a_fake_number(client):
    r = client.post("/auth/", json={"panchayat_name": "ADYAR", "date": "2021-01-01"})
    assert r.status_code == 422 and "2022-07-10" in r.json()["detail"]
    r = client.post("/auth/", json={"panchayat_name": "NOCOORD", "date": "2022-07-10"})
    assert r.status_code == 422 and "coordinates" in r.json()["detail"]


def test_auth_outside_domain(client):
    r = client.post("/auth/", json={"lat": 40.0, "lon": 10.0, "date": "2022-07-10"})
    assert r.status_code == 422


def test_layer2_csv_takes_priority_when_present(repo, client):
    pd.DataFrame([dict(date="2022-07-10", panchayat_id=11, rainfall_mm=44.4)]).to_csv(repo / "outputs" / "layer2" / "panchayat_weather.csv", index=False)
    r = client.post("/auth/", json={"panchayat_name": "ADYAR", "date": "2022-07-10"}).json()
    assert r["prediction"]["rainfall_mm"] == 44.4 and "Layer-2" in r["sources"]["rainfall"]


def test_metrics_from_files(client):
    m = client.get("/api/metrics").json()
    assert m["selected_model"]["test_mae_mm"] == 8 and m["baseline"]["test_mae_mm"] == 10 and m["mae_improvement_pct"] == 20.0


def test_param_count_formula_matches_real_checkpoint():
    assert routes_data.unet_param_count(5, 16) == 117329      # counted from models/best_model.pt


def test_health(client):
    assert "message" in client.get("/").json()


def test_precision_is_kept_when_frontend_echoes_geocode_back(repo, client):
    # NOCOORD has no exact coords and no admin map in the tmp repo -> unknown; use ADYAR (exact) to check the echo path
    g = client.get("/api/geocode", params={"panchayat_name": "ADYAR"}).json()
    r = client.post("/auth/", json={"panchayat_name": "ADYAR", "lat": g["lat"], "lon": g["lon"], "date": "2022-07-10"}).json()
    assert r["sources"]["location_precision"] == "exact"
    r = client.post("/auth/", json={"panchayat_name": "ADYAR", "lat": 12.9, "lon": 74.9, "date": "2022-07-10"}).json()
    assert r["sources"]["location_precision"] == "given"
