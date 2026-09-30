import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["GROQ_API_KEY"] = "x"; os.environ["SARVAM_API_KEY"] = "x"

import pytest
from fastapi.testclient import TestClient
import advisory as adv
import main


def mk(**kw):
    base = dict(panchayat="A", crop="wheat", stage="grain_filling", rainfall_mm=20, tmax_c=28,
                humidity_pct=60, wind_kmh=10, soil_moisture=0.25)
    base.update(kw)
    return adv.PanchayatInput(**base)


def fired_ids(t): return {r.rule_id for r in t.rules if r.fired}


def test_normal_day():
    t = adv.evaluate(mk())
    assert t.severity == "none" and t.action == "normal" and not fired_ids(t)


def test_heavy_rain():
    t = adv.evaluate(mk(rainfall_mm=70))
    assert "R1_HEAVY_RAIN" in fired_ids(t) and t.severity == "high"


def test_irrigation_needs_both_conditions():
    assert "R2_IRRIGATION" in fired_ids(adv.evaluate(mk(rainfall_mm=1, soil_moisture=0.08)))
    assert "R2_IRRIGATION" not in fired_ids(adv.evaluate(mk(rainfall_mm=1, soil_moisture=0.30)))
    assert "R2_IRRIGATION" not in fired_ids(adv.evaluate(mk(rainfall_mm=30, soil_moisture=0.08)))


def test_irrigation_rain_only_fallback_is_low_severity():
    t = adv.evaluate(mk(rainfall_mm=1, soil_moisture=None))
    r = next(r for r in t.rules if r.rule_id == "R2_IRRIGATION")
    assert r.fired and r.severity == "low"
    assert any("soil moisture" in x for x in t.confidence_reasons)


def test_heat_only_in_sensitive_stage():
    assert "R3_HEAT_STRESS" in fired_ids(adv.evaluate(mk(tmax_c=37)))
    assert "R3_HEAT_STRESS" not in fired_ids(adv.evaluate(mk(tmax_c=37, stage="vegetative")))


def test_heat_is_not_fired_when_the_stage_is_unknown():
    """A missing stage must not manufacture a hazard.

    `p["heat"]` is the threshold for the crop's SENSITIVE window, so firing
    without knowing the stage asserts something the inputs do not support. The
    rule reports itself unevaluable instead (and confidence already drops).
    """
    t = adv.evaluate(mk(tmax_c=37, stage=None))
    r = next(r for r in t.rules if r.rule_id == "R3_HEAT_STRESS")
    assert not r.fired and not r.evaluable and r.severity == "none"
    assert "crop stage" in r.flip_hint and "requires a known stage" in r.condition
    assert "R3_HEAT_STRESS" not in fired_ids(t)
    assert t.severity == "none"                       # a 37 C day, no heat alert
    assert any("crop stage" in x for x in t.confidence_reasons)
    # ...but the rule is not simply disabled: a known sensitive stage still fires
    assert "R3_HEAT_STRESS" in fired_ids(adv.evaluate(mk(tmax_c=37, stage="grain_filling")))


def test_missing_temperature_still_reports_heat_unevaluable():
    r = next(r for r in adv.evaluate(mk(tmax_c=None)).rules if r.rule_id == "R3_HEAT_STRESS")
    assert not r.fired and not r.evaluable and "temperature" in r.flip_hint


def test_missing_inputs_are_not_evaluable_and_lower_confidence():
    t = adv.evaluate(mk(humidity_pct=None, wind_kmh=None))
    assert not next(r for r in t.rules if r.rule_id == "R4_DISEASE").evaluable
    assert t.confidence < adv.evaluate(mk()).confidence


def test_every_rule_reports_condition_and_flip_hint():
    for r in adv.evaluate(mk(rainfall_mm=1, soil_moisture=0.08, tmax_c=37)).rules:
        assert r.condition


def test_downscaling_explanation_and_residual():
    t = adv.evaluate(mk(rainfall_mm=58, downscaling=adv.DownscalingInfo(
        block_rainfall_mm=50, contributions={"elevation": 6.0, "slope": 1.5})))
    d = t.downscaling
    assert d.delta_mm == 8 and d.unexplained_mm == 0.5
    assert d.drivers[0]["feature"] == "elevation" and "elevation" in d.text_en


def test_faithfulness_guard():
    t = adv.evaluate(mk(rainfall_mm=70))
    assert adv.is_faithful("Heavy rain of about 70 mm is coming.", t)
    assert not adv.is_faithful("Expect 120 mm of rain.", t)


# ---------------- API (Groq + Sarvam stubbed) ----------------
@pytest.fixture
def client(monkeypatch):
    async def fake_translate(text, src, tgt): return f"[{tgt}] {text}"
    monkeypatch.setattr(main, "sarvam_translate", fake_translate)
    return TestClient(main.app)


def payload(**kw):
    return {"data": dict(panchayat="C", crop="wheat", stage="grain_filling", rainfall_mm=2,
                         tmax_c=36, humidity_pct=50, soil_moisture=0.1, wind_kmh=5), "output_language": "hi-IN", **kw}


def test_api_llm_ok(client, monkeypatch):
    async def ok(model, trace): return "Very little rain (2 mm) is expected, so irrigate your wheat soon."
    monkeypatch.setattr(main, "_phrase_call", ok)
    r = client.post("/api/advisory", json=payload()).json()
    assert r["message_source"].startswith("llm:") and r["message"].startswith("[hi-IN]")
    assert r["fired_rules"] and r["trace"]["rules"]


def test_api_llm_hallucinated_number_falls_back_to_template(client, monkeypatch):
    async def bad(model, trace): return "Expect 150 mm rain, irrigate."
    monkeypatch.setattr(main, "_phrase_call", bad)
    r = client.post("/api/advisory", json=payload()).json()
    assert r["message_source"] == "template" and "150" not in r["message_en"]


def test_api_llm_down_falls_back_to_template(client, monkeypatch):
    async def boom(model, trace): raise RuntimeError("down")
    monkeypatch.setattr(main, "_phrase_call", boom)
    r = client.post("/api/advisory", json=payload()).json()
    assert r["message_source"] == "template" and len(r["attempts"]) == len(main.GROQ_MODELS)


def test_api_translation_failure_keeps_english(monkeypatch):
    async def fail(text, src, tgt): raise main.HTTPException(502, "x")
    async def ok(model, trace): return "Irrigate wheat, only 2 mm rain."
    monkeypatch.setattr(main, "sarvam_translate", fail)
    monkeypatch.setattr(main, "_phrase_call", ok)
    r = TestClient(main.app).post("/api/advisory", json=payload()).json()
    assert r["message"] == r["message_en"]


def test_api_block(client, monkeypatch):
    async def ok(model, trace): return trace.headline_en
    monkeypatch.setattr(main, "_phrase_call", ok)
    body = {"block": "X", "output_language": "ta-IN", "panchayats": [
        payload()["data"], {**payload()["data"], "panchayat": "A", "rainfall_mm": 70}]}
    r = client.post("/api/advisory/block", json=body).json()
    assert [p["action"] for p in r["panchayats"]] == ["heat_stress", "heavy_rain"]
    assert r["panchayats"][1]["action"] == "heavy_rain"
