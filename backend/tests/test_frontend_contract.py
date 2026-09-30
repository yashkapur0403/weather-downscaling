"""Requests shaped exactly like Frontend/src (XAIPanel.buildRequest + api/backend.ts)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["GROQ_API_KEY"] = "x"; os.environ["SARVAM_API_KEY"] = "x"
import pytest
from fastapi.testclient import TestClient
import main

EXPLAIN_BODY = {
    "panchayat_name": "Kondapur", "block_name": "Sangareddy", "district": "Sangareddy", "state": "Telangana",
    "date": "2022-07-10", "lat": 17.5, "lon": 78.1,
    "prediction": {"rainfall_mm": 18.4, "risk_level": "moderate", "temperature_c": 26.1, "humidity_pct": 82.0, "elevation_m": 560.0},
    "mapping": {"method": "area_weighted", "n_cells": 4, "fallback_distance_m": None},
    "model": {"name": "U-Net Model D (DEM + ERA5)", "channels": ["imd_rain", "dem", "era5_t2m", "era5_t2m_max", "era5_dewp"],
              "test_mae_mm": 6.45, "baseline_mae_mm": 7.67, "mae_improvement_pct": 15.9, "reference_product": "CHIRPS v2.0"},
    "question": None, "language": "en",
}
EXPLAIN_KEYS = {"summary", "explanation", "factors", "confidence", "confidence_note", "answer", "provider", "model", "generated_at"}
ADVISORY_KEYS = {"advisory_text", "severity", "actions", "evidence", "data_date", "disclaimer", "crop", "stage"}


@pytest.fixture
def client(app_with_data, monkeypatch):
    # `app_with_data` wires the data layer to the tiny repo (tests/conftest.py) so
    # the advisory assertions below go through the real verification path: the
    # rainfall each test sends is checked against the stored Layer-1 field.
    async def tr(text, src, tgt): return f"[{tgt}] {text}"
    monkeypatch.setattr(main, "sarvam_translate", tr)
    return TestClient(app_with_data)


def test_status(client):
    r = client.get("/api/explain/status").json()
    assert set(r) == {"enabled", "provider", "model"} and r["provider"] == "groq"


def test_explain_llm_path(client, monkeypatch):
    async def ok(model, facts_json, q): return {"explanation": "IMD's 28 km rainfall is refined to 5 km, using 4 grid cells.", "answer": None}
    monkeypatch.setattr(main, "_explain_call", ok)
    r = client.post("/api/explain", json=EXPLAIN_BODY).json()
    assert EXPLAIN_KEYS <= set(r) and r["provider"] == "groq" and r["confidence"] == "high"
    assert {f["factor"] for f in r["factors"]} >= {"IMD coarse rainfall (0.25°)", "Elevation (SRTM DEM)", "Moisture (ERA5 dewpoint)"}
    assert all(f["effect"] in ("increases", "decreases", "neutral") and 0 <= f["weight"] <= 1 for f in r["factors"])


def test_explain_hallucinated_number_falls_back_to_rules(client, monkeypatch):
    async def bad(model, facts_json, q): return {"explanation": "Rain was 250 mm because of a cyclone.", "answer": None}
    monkeypatch.setattr(main, "_explain_call", bad)
    r = client.post("/api/explain", json=EXPLAIN_BODY).json()
    assert r["provider"] == "rules" and "250" not in r["explanation"] and "cyclone" not in r["explanation"]


def test_explain_llm_down_and_question(client, monkeypatch):
    async def boom(model, facts_json, q): raise RuntimeError("down")
    monkeypatch.setattr(main, "_explain_call", boom)
    r = client.post("/api/explain", json={**EXPLAIN_BODY, "question": "why higher than nearby?"}).json()
    assert r["provider"] == "rules" and r["answer"]


def test_explain_only_lists_channels_the_model_used(client, monkeypatch):
    async def boom(model, facts_json, q): raise RuntimeError("down")
    monkeypatch.setattr(main, "_explain_call", boom)
    body = {**EXPLAIN_BODY, "model": {**EXPLAIN_BODY["model"], "channels": ["imd_rain"]}}
    names = [f["factor"] for f in client.post("/api/explain", json=body).json()["factors"]]
    assert names == ["IMD coarse rainfall (0.25°)"]


def test_explain_fallback_confidence_low(client, monkeypatch):
    async def boom(model, facts_json, q): raise RuntimeError("down")
    monkeypatch.setattr(main, "_explain_call", boom)
    body = {**EXPLAIN_BODY, "mapping": {"method": "nearest_fallback", "n_cells": 1, "fallback_distance_m": 9000}}
    assert client.post("/api/explain", json=body).json()["confidence"] == "low"


def test_explain_hindi(client, monkeypatch):
    async def boom(model, facts_json, q): raise RuntimeError("down")
    monkeypatch.setattr(main, "_explain_call", boom)
    r = client.post("/api/explain", json={**EXPLAIN_BODY, "language": "hi"}).json()
    assert r["summary"].startswith("[hi-IN]") and r["confidence"] in ("low", "medium", "high")


def _adv(client, **q):
    # GP11 stores 70.0 mm for 2022-07-10; the advisory verifies whatever we send
    # against that, so the defaults use the stored value.
    base = dict(panchayat_id=11, crop="wheat", stage="general", rainfall_mm=70.0, temperature_c=30, humidity_pct=50,
                date="2022-07-10", panchayat_name="GP11")
    base.update(q)
    return client.get("/api/advisory", params=base)


def test_advisory_contract(client, monkeypatch):
    async def ok(model, trace): return trace.headline_en
    monkeypatch.setattr(main, "_phrase_call", ok)
    r = _adv(client, rainfall_mm=70).json()
    assert ADVISORY_KEYS <= set(r) and r["severity"] == "alert" and r["actions"]
    assert r["verification"]["rainfall"] == "verified_against_layer1"
    assert set(r["evidence"]) == {"rainfall_mm", "risk_level", "temperature_c", "humidity_pct", "aux"}
    assert r["evidence"]["risk_level"] == "very_heavy" and r["crop"] == "wheat" and r["stage"] == "general"


def test_advisory_all_frontend_options_accepted(client, monkeypatch):
    async def ok(model, trace): return trace.headline_en
    monkeypatch.setattr(main, "_phrase_call", ok)
    for crop in ("general", "rice", "wheat", "cotton", "maize", "pulses"):
        for stage in ("general", "sowing", "vegetative", "flowering", "ripening", "harvest"):
            assert _adv(client, crop=crop, stage=stage).status_code == 200, (crop, stage)


def test_advisory_severity_mapping_and_normal(client, monkeypatch):
    async def ok(model, trace): return trace.headline_en
    monkeypatch.setattr(main, "_phrase_call", ok)
    # GP12 stores 30.0 mm, GP13 stores 12.0 mm (tests/conftest.py)
    assert _adv(client, panchayat_id=12, rainfall_mm=30.0).json()["severity"] == "warning"
    n = _adv(client, panchayat_id=13, rainfall_mm=12.0, humidity_pct=60).json()
    assert n["severity"] == "info" and n["actions"]


def test_advisory_bad_input_is_422_so_frontend_can_show_an_error(client):
    assert _adv(client, crop="banana").status_code == 422
    # no rainfall and an id that is not in the coordinate index -> cannot verify
    assert client.get("/api/advisory", params={"panchayat_id": 1}).status_code == 422


def test_advisory_hindi_param(client, monkeypatch):
    async def ok(model, trace): return trace.headline_en
    monkeypatch.setattr(main, "_phrase_call", ok)
    r = _adv(client, lang="hi-IN").json()
    assert r["advisory_text"].startswith("[hi-IN]") and r["actions"][0].startswith("[hi-IN]")


# ── No second advisory rule table in the browser ─────────────────────────────
# The rules live only in backend/advisory.py. A client-side copy cannot be kept in
# step with them - it cannot see temperature, soil, NDVI or land cover at all - and
# two rule tables means the farmer sees whichever one happened to answer, so an
# outage could turn an `alert` into an `info` on identical numbers. The panel must
# therefore show no advice rather than its own.
PANEL_TSX = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Frontend", "src", "components", "advisory", "AdvisoryPanel.tsx",
)


def test_advisory_panel_has_no_second_rule_table():
    if not os.path.exists(PANEL_TSX):
        pytest.skip("Frontend/ is not mounted next to backend/")
    src = open(PANEL_TSX, encoding="utf-8").read()
    assert "generateLocalAdvisory" not in src
    # ...but it must still say why there is no advice
    assert "Advisory withheld" in src and "Advisory unavailable" in src
    # the engine's rainfall tiers must not be re-declared in the browser
    for mm in ("64.5", "24.5"):
        assert mm not in src, f"AdvisoryPanel re-declares the backend tier {mm} mm"
