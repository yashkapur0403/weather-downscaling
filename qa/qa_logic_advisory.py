"""Logical validation - Layer 3 (advisory) experiments.

Executes the REAL advisory engine (backend/advisory.py) with controlled inputs:
  * every threshold: T-eps, T, T+eps
  * one-input-at-a-time perturbations (crop, stage, temp, rain, soil, NDVI, LULC)
  * rule-action coherence
  * LLM faithfulness guard behaviour
Writes qa_logic_advisory.json.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # repo root (this script lives in qa/)
sys.path.insert(0, str(HERE / "backend"))
import advisory as A  # noqa: E402

OUT = {}
EPS = 0.01


def run(**kw):
    """Evaluate the engine with defaults + overrides; return a compact view."""
    base = dict(panchayat="TESTGP", crop="rice", stage="flowering", rainfall_mm=0.0)
    base.update(kw)
    aux = base.pop("aux", None)
    if aux is not None:
        base["aux"] = aux
    t = A.evaluate(A.PanchayatInput(**base))
    return {
        "severity": t.severity,
        "action": t.action,
        "fired": sorted(r.rule_id for r in t.rules if r.fired),
        "evaluable_false": sorted(r.rule_id for r in t.rules if not r.evaluable),
        "n_actions": len(A.actions_for(t)),
        "confidence": t.confidence,
        "headline": t.headline_en[:160],
    }


def aux_context(ndvi=None, ndvi_month="2022-07", clay=None, sand=None,
                dominant=None, cropland_fraction=None):
    d = {}
    if ndvi is not None:
        d["ndvi"] = {"value": ndvi, "month": ndvi_month}
    if clay is not None or sand is not None:
        d["soil"] = {"clay_g_per_kg": clay, "sand_g_per_kg": sand}
    if dominant is not None or cropland_fraction is not None:
        d["lulc"] = {"dominant": dominant, "fractions":
                     {"cropland_fraction": cropland_fraction}
                     if cropland_fraction is not None else {}}
    return d


# ---------------------------------------------------------------- thresholds
thr_cases = {
    "R1_HEAVY_RAIN@64.5": [dict(rainfall_mm=x, tmax_c=25.0, stage="vegetative")
                           for x in (64.5 - EPS, 64.5, 64.5 + EPS)],
    "R1B_SUBSTANTIAL@24.5": [dict(rainfall_mm=x, tmax_c=25.0, stage="vegetative")
                             for x in (24.5 - EPS, 24.5, 24.5 + EPS)],
    "R1B_UPPER@64.5": [dict(rainfall_mm=x, tmax_c=25.0, stage="vegetative")
                       for x in (64.4, 64.5, 64.6)],
    "R2_IRRIGATION_rainonly@3.0": [dict(rainfall_mm=x) for x in (3.0 - EPS, 3.0, 3.0 + EPS)],
    "R3_HEAT_rice@35": [dict(rainfall_mm=0.0, tmax_c=x, stage="flowering")
                        for x in (35.0 - EPS, 35.0, 35.0 + EPS)],
    "R3_HEAT_wheat@34": [dict(rainfall_mm=0.0, tmax_c=x, stage="grain_filling", crop="wheat")
                         for x in (34.0 - EPS, 34.0, 34.0 + EPS)],
    "R4_DISEASE_RH@85": [dict(rainfall_mm=0.0, tmax_c=28.0, humidity_pct=x)
                         for x in (85.0 - EPS, 85.0, 85.0 + EPS)],
    "R4_DISEASE_Tlo@22": [dict(rainfall_mm=0.0, tmax_c=x, humidity_pct=90.0)
                          for x in (22.0 - EPS, 22.0, 22.0 + EPS)],
    "R4_DISEASE_Thi@32": [dict(rainfall_mm=0.0, tmax_c=x, humidity_pct=90.0)
                          for x in (32.0 - EPS, 32.0, 32.0 + EPS)],
    "R5_LODGING_wind@40": [dict(rainfall_mm=0.0, wind_kmh=x, crop="maize")
                           for x in (40.0 - EPS, 40.0, 40.0 + EPS)],
    "R6_NDVI@0.30": [dict(rainfall_mm=0.0, aux=aux_context(ndvi=x))
                     for x in (0.30 - EPS, 0.30, 0.30 + EPS)],
    "R7_CLAY@350": [dict(rainfall_mm=30.0, aux=aux_context(clay=x))
                    for x in (350.0 - EPS, 350.0, 350.0 + EPS)],
    "R8_CROPFrac@0.2": [dict(rainfall_mm=0.0,
                             aux=aux_context(dominant="cropland", cropland_fraction=x))
                        for x in (0.2 - EPS, 0.2, 0.2 + EPS)],
}
OUT["threshold_boundaries"] = {}
for name, cases in thr_cases.items():
    vals = []
    for c in cases:
        r = run(**c)
        r["input"] = {k: v for k, v in c.items() if k != "aux"}
        if "aux" in c:
            r["input"]["aux"] = json.dumps(c["aux"])
        vals.append(r)
    OUT["threshold_boundaries"][name] = vals
    print(f"\n== {name} ==")
    for v in vals:
        print(f"   {v['input']}  -> sev={v['severity']:6s} fired={v['fired']}")

# ------------------------------------------------------------- perturbations
OUT["perturbations"] = []


def pert(label, fixed, key, values, expect_note):
    rows = []
    for v in values:
        kw = dict(fixed)
        if key == "aux":
            kw["aux"] = v
        else:
            kw[key] = v
        rows.append({"value": (json.dumps(v) if key == "aux" else v), "out": run(**kw)})
    sig = {json.dumps(r["out"]["fired"]) + r["out"]["severity"] for r in rows}
    rec = {"perturbation": label, "varied": key, "expected": expect_note,
           "changed": len(sig) > 1, "rows": rows}
    OUT["perturbations"].append(rec)
    print(f"\n== perturb {label} (varied {key}) -> changed={rec['changed']} ==")
    for r in rows:
        print(f"   {r['value']} -> sev={r['out']['severity']:6s} "
              f"fired={r['out']['fired']} conf={r['out']['confidence']}")


# crop only (everything else fixed) - use a hot, humid, low-rain day
FIX = dict(rainfall_mm=0.0, tmax_c=36.0, humidity_pct=90.0, stage="flowering")
pert("crop changes", FIX, "crop", ["rice", "wheat", "maize", "cotton", "bajra", "pulses"],
     "heat threshold is crop-specific -> heat rule must change")
pert("stage changes", dict(rainfall_mm=0.0, tmax_c=34.5, humidity_pct=90.0, crop="wheat"),
     "stage", ["sowing", "vegetative", "flowering", "grain_filling", "maturity", None],
     "heat-sensitive stages for wheat are flowering/grain_filling")
pert("temperature changes", dict(rainfall_mm=0.0, crop="rice", stage="flowering"),
     "tmax_c", [20.0, 30.0, 34.9, 35.0, 40.0], "R3 heat must fire at/above 35 C")
pert("rainfall changes",
     dict(crop="rice", stage="vegetative", tmax_c=25.0),
     "rainfall_mm", [0.0, 2.9, 3.0, 24.4, 24.5, 64.4, 64.5, 120.0],
     "R2/R1B/R1 tiers must step at 3/24.5/64.5")
pert("soil clay changes", dict(rainfall_mm=30.0, crop="rice", tmax_c=25.0),
     "aux", [aux_context(clay=100.0), aux_context(clay=349.9), aux_context(clay=350.0),
             aux_context(clay=500.0)],
     "R7 fires at clay>=350 with rain>=24.5")
pert("NDVI changes", dict(rainfall_mm=0.0, crop="rice", tmax_c=25.0),
     "aux", [aux_context(ndvi=0.55), aux_context(ndvi=0.30), aux_context(ndvi=0.1)],
     "R6 fires below NDVI 0.30")
pert("LULC changes", dict(rainfall_mm=0.0, crop="rice", tmax_c=25.0),
     "aux", [aux_context(dominant="cropland", cropland_fraction=0.9),
             aux_context(dominant="built_up", cropland_fraction=0.9),
             aux_context(dominant="tree_cover", cropland_fraction=0.1)],
     "R8 fires for non-cropland dominant or cropland fraction < 0.2")
pert("aux absent vs present", dict(rainfall_mm=0.0, crop="rice", tmax_c=25.0),
     "aux", [None, aux_context(ndvi=0.6, clay=100, dominant="cropland", cropland_fraction=0.9)],
     "confidence must rise when aux is supplied")
pert("soil moisture present", dict(rainfall_mm=1.0, crop="rice", tmax_c=25.0),
     "soil_moisture", [None, 0.35, 0.29, 0.05],
     "R2 fires only when rain is low AND soil is dry (<0.30 for rice)")

# --------------------------------------------------- coherence / contradictions
OUT["coherence"] = []
combos = [
    ("very heavy rain 200mm", dict(rainfall_mm=200.0, tmax_c=25.0, crop="rice", stage="vegetative")),
    ("no rain + hot + dry", dict(rainfall_mm=0.0, tmax_c=40.0, crop="rice",
                                 stage="flowering", soil_moisture=0.05)),
    ("heavy rain over clay", dict(rainfall_mm=80.0, tmax_c=25.0, crop="rice",
                                  aux=aux_context(clay=450.0))),
    ("heavy rain + heat", dict(rainfall_mm=90.0, tmax_c=40.0, crop="rice", stage="flowering")),
    ("dry + built-up land cover", dict(rainfall_mm=0.0, tmax_c=25.0, crop="rice",
                                       aux=aux_context(dominant="built_up", cropland_fraction=0.05))),
]
for label, kw in combos:
    r = run(**kw)
    irrig = "R2_IRRIGATION" in r["fired"]
    heavy = "R1_HEAVY_RAIN" in r["fired"]
    acts = A.actions_for(A.evaluate(A.PanchayatInput(panchayat="TESTGP", **kw)))
    contradiction = bool(heavy and irrig)
    OUT["coherence"].append({"case": label, "out": r, "irrigation_with_heavy_rain": contradiction,
                             "actions": acts})
    print(f"\n== coherence {label} == {r['fired']} contradiction={contradiction}")

# ---------------------------------------------------------- faithfulness guard
traces = [A.evaluate(A.PanchayatInput(panchayat="TESTGP",
                                      **dict(rainfall_mm=80.0, tmax_c=30.0, humidity_pct=90.0,
                                             crop="rice", stage="flowering")))]
t = traces[0]
guard = {
    "trace_headline": t.headline_en,
    "accept_verbatim": A.is_faithful(t.headline_en, t),
    "accept_harmless": A.is_faithful("Heavy rain is expected today.", t),
    "reject_new_number": A.is_faithful("Rainfall will reach 137 mm.", t),
    "reject_action_bullet_48h": A.is_faithful("Avoid spraying for about 48 hours.", t),
    "allowed_numbers_sample": sorted(A.allowed_numbers(t))[:25],
}
OUT["faithfulness_guard"] = guard
print("\n== faithfulness guard ==")
print(json.dumps(guard, indent=1, default=str))

json.dump(OUT, open(HERE / "qa" / "qa_logic_advisory.json", "w"), indent=1, default=str)
print("\n-> qa_logic_advisory.json")
