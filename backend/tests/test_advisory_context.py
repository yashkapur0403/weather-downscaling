"""Part-7 tests for the crop/stage-aware advisory context and the data-grounded
soil / crop-suitability evidence.

They prove three things the product now claims:
  1. crop and stage change the advice ONLY where the rule set genuinely reacts
     (and the trace says so when they do not);
  2. every action and every soil value shown is traceable to a measured input,
     with no fabricated soil/crop value when data is absent;
  3. the Layer-1 rainfall value itself is unchanged.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["GROQ_API_KEY"] = "x"
os.environ["SARVAM_API_KEY"] = "x"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import advisory as adv  # noqa: E402
import main  # noqa: E402


def mk(**kw):
    base = dict(panchayat="A", crop="wheat", stage="flowering", rainfall_mm=20,
                tmax_c=28, humidity_pct=60, wind_kmh=10, soil_moisture=0.25)
    base.update(kw)
    return adv.PanchayatInput(**base)


def fired(t):
    return {r.rule_id for r in t.rules if r.fired}


# ── 1/2. crop + stage causality, measured not asserted ───────────────────────
def test_stage_changes_heat_only_at_a_sensitive_stage():
    hot_flowering = adv.evaluate(mk(tmax_c=36, stage="flowering"))
    hot_sowing = adv.evaluate(mk(tmax_c=36, stage="sowing"))
    assert "R3_HEAT_STRESS" in fired(hot_flowering)
    assert "R3_HEAT_STRESS" not in fired(hot_sowing)
    # flowering IS sensitive for wheat, so the crop/stage choice changed the outcome
    assert hot_flowering.context.materially_changed is True
    # sowing is outside the sensitive window: same weather, same fired set as general
    assert hot_sowing.context.materially_changed is False


def test_crop_changes_the_heat_threshold():
    # wheat heat threshold is 34 C, rice is 35 C -> 34 C separates them
    wheat = adv.evaluate(mk(crop="wheat", stage="flowering", tmax_c=34))
    rice = adv.evaluate(mk(crop="rice", stage="flowering", tmax_c=34))
    assert "R3_HEAT_STRESS" in fired(wheat)
    assert "R3_HEAT_STRESS" not in fired(rice)
    assert wheat.context.crop_parameters["heat_threshold_c"] == 34
    assert rice.context.crop_parameters["heat_threshold_c"] == 35


def test_same_crop_and_stage_different_rainfall_changes_the_advisory():
    dry = adv.evaluate(mk(rainfall_mm=1, soil_moisture=0.08))
    wet = adv.evaluate(mk(rainfall_mm=70))
    assert "R2_IRRIGATION" in fired(dry)
    assert "R1_HEAVY_RAIN" in fired(wet)
    assert fired(dry) != fired(wet)


def test_same_rain_different_stage_is_identical_when_no_stage_rule_applies():
    # cool day: heat cannot fire, so flowering and vegetative must agree
    a = adv.evaluate(mk(tmax_c=20, stage="flowering"))
    b = adv.evaluate(mk(tmax_c=20, stage="vegetative"))
    assert fired(a) == fired(b)
    assert a.context.materially_changed is False
    assert b.context.materially_changed is False
    # ...and the trace states that honestly rather than pretending otherwise
    assert "same fired rules" in a.context.explanation


def test_crop_parameters_exposed_are_the_ones_applied():
    t = adv.evaluate(mk(crop="wheat"))
    cp = t.context.crop_parameters
    assert cp["heat_threshold_c"] == adv.CROP_PARAMS["wheat"]["heat"]
    assert cp["dry_soil_moisture_threshold"] == adv.CROP_PARAMS["wheat"]["dry_sm"]
    assert "flowering" in cp["heat_sensitive_stages"]


def test_context_flags_crop_is_user_selected_not_detected():
    t = adv.evaluate(mk(crop="rice"))
    c = t.context
    assert c.crop_detected is False
    assert c.crop_dataset_available is False
    assert c.user_selected_crop == "rice"


# ── 4/5/6. soil evidence fidelity, no fabrication ────────────────────────────
def _aux(clay=389.0):
    return adv.AuxContext(soil=adv.AuxSoil(
        depth="5-15cm", sand_g_per_kg=520.0, clay_g_per_kg=clay,
        ocd_dg_per_dm3=12.0, ph=6.4, bdod=1.42))


def test_soil_characteristics_match_the_retrieved_values_exactly():
    t = adv.evaluate(mk(rainfall_mm=26.2, aux=_aux()))
    props = {p["label"]: p["value"] for p in t.soil["properties"]}
    assert props["Clay"] == 389.0 and props["Sand"] == 520.0 and props["pH"] == 6.4
    # the evidence groups must show the SAME numbers
    assert t.evidence_groups["observed"]["soil"]["Clay"] == 389.0


def test_no_soil_is_never_replaced_by_a_fabricated_value():
    t = adv.evaluate(mk())                     # no aux at all
    assert t.soil["available"] is False
    assert t.soil["properties"] == []
    assert t.crop_suitability["available"] is False
    assert "Insufficient data" in t.crop_suitability["reason"]
    assert t.crop_suitability["crop_detected"] is False
    assert t.crop_suitability["crop_dataset_available"] is False
    assert t.crop_suitability["user_selected_crop"] == "wheat"
    assert any("soil" in m for m in t.evidence_groups["not_available"])


def test_crop_suitability_refuses_even_when_soil_is_present():
    """Having soil properties is NOT enough to make a suitability claim: the
    deployment has no crop-type or soil-classification dataset."""
    t = adv.evaluate(mk(rainfall_mm=26.2, aux=_aux()))
    assert t.crop_suitability["available"] is False
    assert "Insufficient data" in t.crop_suitability["reason"]
    # evidence is still shown, but only as measured characteristics
    assert {e["label"] for e in t.crop_suitability["evidence"]} >= {"Clay", "Sand", "pH"}
    # and the dataset is never dressed up as a "soil type"
    assert "soil-type classification" in t.soil["note"]


def test_action_items_carry_the_rule_and_its_evidence():
    t = adv.evaluate(mk(rainfall_mm=70))
    items = adv.action_items_for(t)
    heavy = [i for i in items if i.rule_id == "R1_HEAVY_RAIN"]
    assert heavy and all(i.evidence["rainfall_mm"] == 70 for i in heavy)
    # the plain list stays available and agrees on count/order
    assert [i.action for i in items] == adv.actions_for(t)


def test_heavy_rain_and_heat_do_not_contradict_on_irrigation():
    """A wet AND hot day must not tell the farmer to both hold and add water."""
    t = adv.evaluate(mk(rainfall_mm=70, tmax_c=37, stage="flowering"))
    assert {"R1_HEAVY_RAIN", "R3_HEAT_STRESS"} <= fired(t)
    acts = adv.actions_for(t)
    assert "Hold back irrigation and fertiliser" in acts
    assert not any("if water is available" in a for a in acts)
    assert any("only if the soil is dry" in a for a in acts)
    assert "if water is available" not in t.headline_en


def test_vegetation_actions_follow_the_rain_context():
    """Low NDVI with heavy rain is NOT a water-stress message."""
    dry = adv.evaluate(mk(rainfall_mm=1, soil_moisture=0.05,
                          aux=adv.AuxContext(ndvi=adv.AuxNDVI(value=0.1, month="2022-07"))))
    wet = adv.evaluate(mk(rainfall_mm=70,
                          aux=adv.AuxContext(ndvi=adv.AuxNDVI(value=0.1, month="2022-07"))))
    assert any("water stress" in a.action for a in adv.action_items_for(dry) if a.rule_id == "R6_VEGETATION")
    wet_actions = [a.action for a in adv.action_items_for(wet) if a.rule_id == "R6_VEGETATION"]
    assert wet_actions and not any("water stress" in a for a in wet_actions)
    assert any("sparse cover" in a for a in wet_actions)


def test_evidence_groups_separate_provenance():
    t = adv.evaluate(mk(rainfall_mm=26.2, aux=_aux()))
    g = t.evidence_groups
    assert set(g) == {"observed", "user_provided", "derived", "not_available"}
    assert "rainfall_mm" in g["observed"]
    assert g["user_provided"]["crop"] == "wheat"
    assert "severity" in g["derived"]
    assert g["derived"]["fired_rules"] == [r.rule_id for r in t.rules if r.fired]


# ── 7. the LLM may not introduce a soil/crop number ──────────────────────────
def test_faithfulness_guard_rejects_an_invented_soil_value():
    t = adv.evaluate(mk(rainfall_mm=70, aux=_aux()))
    assert adv.is_faithful("The clay content is 389 g/kg.", t)
    assert not adv.is_faithful("The clay content is 450 g/kg.", t)


# ── 8/9/10. the API keeps its rainfall contract and exposes the new evidence ──
class FakeAux:
    """Stands in for aux_layers.AuxLayers with one Panchayat that has soil data."""

    def lookup_panchayat(self, pid, date=None):
        if int(pid) == 11:
            return {
                "cell": [16, 17],
                "soil": {"depth": "5-15cm", "sand_g_per_kg": 520.0, "clay_g_per_kg": 389.0,
                         "ocd_dg_per_dm3": 12.0, "ph": 6.4, "bdod": 1.42},
                "ndvi": {"value": 0.42, "month": "2022-07"},
                "lulc": {"fractions": {"cropland_fraction": 0.8}, "dominant": "cropland"},
            }
        return None


@pytest.fixture
def client(app_with_data, monkeypatch):
    monkeypatch.setattr(main, "aux_layers_ctx", lambda: FakeAux())

    async def ok(model, trace):
        return trace.headline_en

    monkeypatch.setattr(main, "_phrase_call", ok)
    return TestClient(app_with_data)


def _adv(client, **q):
    # GP11 stores 70.0 mm, GP12 stores 30.0 mm on 2022-07-10 (tests/conftest.py)
    base = dict(panchayat_id=11, crop="wheat", stage="flowering", rainfall_mm=70.0,
                temperature_c=30, humidity_pct=50, date="2022-07-10", panchayat_name="GP11")
    base.update(q)
    # drop None so an omitted rainfall_mm really is omitted (httpx would send "None")
    return client.get("/api/advisory", params={k: v for k, v in base.items() if v is not None})


def test_api_exposes_soil_context_and_action_evidence(client):
    r = _adv(client).json()
    assert r["crop_suitability"]["available"] is False
    assert r["advisory_context"]["crop_detected"] is False
    clay = next((p for p in r["soil_context"]["properties"] if p["label"] == "Clay"), None)
    assert clay and clay["value"] == 389.0
    assert r["evidence_groups"]["observed"]["soil"]["Clay"] == 389.0
    wet = [a for a in r["action_items"] if a["rule_id"] == "R7_SOIL_DRAINAGE"]
    assert wet and wet[0]["evidence"]["clay_g_per_kg"] == 389.0
    assert any(f["rule_id"] == "R7_SOIL_DRAINAGE" for f in r["fired_rules"])


def test_api_reports_missing_soil_without_inventing_any(client):
    r = _adv(client, panchayat_id=12, rainfall_mm=30.0, crop="general", stage="general").json()
    assert r["soil_context"]["available"] is False
    assert r["soil_context"]["properties"] == []
    assert r["crop_suitability"]["available"] is False
    assert r["crop_suitability"]["crop_detected"] is False
    assert any("soil" in m for m in r["evidence_groups"]["not_available"])


def test_api_rainfall_value_is_unchanged(client):
    r = _adv(client, rainfall_mm=None).json()          # let the server resolve it
    assert r["evidence"]["rainfall_mm"] == 70.0
    assert r["verification"]["rainfall"] == "resolved_from_layer1"
    assert r["verification"]["aux"] == "verified_against_layer2_cell"


def test_api_disease_rule_fires_when_humidity_is_supplied(client):
    """The dashboard now forwards the humidity it already fetched, so the disease
    rule is evaluated (it used to be skipped because humidity was never sent)."""
    r = _adv(client, temperature_c=28, humidity_pct=90).json()
    assert any(f["rule_id"] == "R4_DISEASE" for f in r["fired_rules"])
    assert r["evidence_groups"]["user_provided"]["humidity_pct"] == 90
    # ...and the standard caller-supplied label is still honest about provenance
    assert r["verification"]["humidity_pct"] == "caller_supplied"


def test_grain_filling_stage_is_reachable_via_get(client):
    """Regression for the StageQ gap: every engine stage must be selectable on GET."""
    for stage in ("general", "sowing", "vegetative", "flowering", "grain_filling", "ripening", "harvest"):
        assert _adv(client, stage=stage).status_code == 200, stage
