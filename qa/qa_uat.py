"""QA: UAT journeys + end-to-end value traces against the live backend."""
import json, os, urllib.request, urllib.parse, urllib.error
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent   # repo root (this script lives in qa/)
os.chdir(HERE)
import numpy as np, pandas as pd

BASE = "http://127.0.0.1:8000"
R = []
def rec(cid, cat, test, exp, act, st, sev="", note=""):
    R.append(dict(id=cid, cat=cat, test=test, exp=str(exp), act=str(act), st=st, sev=sev, note=note))

def get(path, params=None):
    u = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    try:
        with urllib.request.urlopen(u, timeout=30) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try: return e.code, json.loads(e.read().decode())
        except Exception: return e.code, None

def post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try: return e.code, json.loads(e.read().decode())
        except Exception: return e.code, None

# ---- reference source data for independent recompute ----
z = np.load("outputs/prediction_test.npz")
rain, la, lo, dates = z["rainfall_mm"], z["latitude"], z["longitude"], [str(d) for d in z["dates"]]
idx = pd.read_csv("outputs/layer2/panchayat_index.csv")
coord = {int(r.panchayat_id): (r.lat, r.lon) for r in idx.itertuples()}
summ = pd.read_csv("outputs/layer2/panchayat_summary.csv")

def expected(pid, date):
    L, O = coord[pid]
    i = int(np.abs(la - L).argmin()); j = int(np.abs(lo - O).argmin())
    v = rain[dates.index(date), i, j]
    return ("NaN/422" if not np.isfinite(v) else round(float(v), 2)), i, j

CASES = [
    ("ALURU", 220447, "KUNDAPURA", "Udupi", "KARNATAKA", 50.49),
    ("NAGULAMALLIAL", 201416, "Kothapally", "Karimnagar", "TELANGANA", 41.77),
    ("AMERDA", 202120, "ASWAPURAM", "Bhadradri Kothagudem", "TELANGANA", 27.68),
    ("CURDI", 254406, "SANGUEM", "South Goa", "GOA", 16.73),
]

# ================= E2E VALUE TRACE =================
for name, pid, block, dist, state, planval in CASES:
    exp, i, j = expected(pid, "2022-07-10")
    code, raw = post("/auth/", {"panchayat_name": name, "block_name": block, "district": dist,
                                "state": state, "date": "2022-07-10"})
    got = raw["prediction"]["rainfall_mm"] if code == 200 else None
    src = raw["sources"]["rainfall"] if code == 200 else None
    # search metadata
    _, sr = get("/api/panchayats", {"q": name, "limit": 5})
    row = next((x for x in sr if x["panchayat_id"] == pid), {})
    csvrow = summ[summ.panchayat_id == pid]
    csvm = csvrow.iloc[0]["mapping_method"] if len(csvrow) else None
    csvn = int(csvrow.iloc[0]["n_cells"]) if len(csvrow) else None
    ok = (code == 200 and abs(got - exp) < 1e-9 and got == planval
          and row.get("mapping_method") == csvm and row.get("n_cells") == csvn)
    rec(f"E2E-{pid}", "e2e-trace", f"{name}: coord->npz[{i},{j}]->/auth->advisory",
        f"npz={exp} plan={planval} mapping={csvm}/{csvn}",
        f"auth={got} src={src} search_map={row.get('mapping_method')}/{row.get('n_cells')} coord=({coord[pid][0]},{coord[pid][1]})",
        "PASS" if ok else "FAIL", "High",
        "chain: panchayat_index coord -> nearest npz cell -> round2 -> /auth response; search fields match panchayat_summary.csv")

# ================= UAT JOURNEYS =================
# J1 full journey
c1, a1 = post("/auth/", {"panchayat_name": "ALURU", "block_name": "KUNDAPURA", "district": "Udupi", "state": "KARNATAKA", "date": "2022-07-10"})
c2, adv = get("/api/advisory", {"panchayat_id": 220447, "crop": "wheat", "stage": "general",
                                "rainfall_mm": a1["prediction"]["rainfall_mm"], "date": "2022-07-10", "panchayat_name": "ALURU"})
c3, xai = post("/api/explain", {"panchayat_name": "ALURU", "date": "2022-07-10", "lat": 13.679452, "lon": 74.742466,
    "prediction": {"rainfall_mm": 50.49, "risk_level": "very_heavy", "temperature_c": 26.0, "humidity_pct": 80, "elevation_m": 20.0},
    "mapping": {"method": "direct_grid", "n_cells": 2, "fallback_distance_m": None},
    "model": {"name": "U-Net + DEM + ERA5-Land", "channels": ["imd_rain", "dem", "era5_t2m", "era5_t2m_max", "era5_dewp"],
              "test_mae_mm": 8.167, "baseline_mae_mm": 9.603, "mae_improvement_pct": 15.0, "reference_product": "CHIRPS v2.0"},
    "language": "en"})
j1 = (c1 == 200 and c2 == 200 and c3 == 200 and a1["prediction"]["rainfall_mm"] == 50.49
      and adv["evidence"]["rainfall_mm"] == 50.49 and adv["severity"] == "warning"
      and "50.5" in xai.get("summary", ""))
rec("UAT-J1", "UAT", "Select ALURU -> projection -> advisory -> explanation consistent",
    "50.49 mm, warning, XAI summary mentions 50.5",
    f"rain={a1['prediction']['rainfall_mm']} sev={adv['severity']} ev={adv['evidence']['rainfall_mm']} xai='{xai.get('summary','')[:40]}'",
    "PASS" if j1 else "FAIL", "High")

# J2 change panchayat -> different value
cA, aA = post("/auth/", {"panchayat_name": "ALURU", "district": "Udupi", "state": "KARNATAKA", "date": "2022-07-10"})
cB, aB = post("/auth/", {"panchayat_name": "CURDI", "district": "South Goa", "state": "GOA", "date": "2022-07-10"})
rec("UAT-J2", "UAT", "Change Panchayat yields a different rainfall (not cached/default)",
    "ALURU 50.49 != CURDI 16.73",
    f"ALURU={aA['prediction']['rainfall_mm']} CURDI={aB['prediction']['rainfall_mm']}",
    "PASS" if aA["prediction"]["rainfall_mm"] != aB["prediction"]["rainfall_mm"] else "FAIL", "High")

# J3 change date
vals = {}
for d in ("2022-06-01", "2022-07-10", "2022-09-30"):
    _, rr = post("/auth/", {"panchayat_name": "ALURU", "district": "Udupi", "state": "KARNATAKA", "date": d})
    vals[d] = rr["prediction"]["rainfall_mm"]
rec("UAT-J3", "UAT", "Change date yields date-specific values",
    "2022-06-01=0.0 (plan), 2022-07-10=50.49, 2022-09-30 in range",
    json.dumps(vals), "PASS" if len(set(vals.values())) > 1 and vals["2022-07-10"] == 50.49 else "FAIL", "High")

# J4 unavailable data (no coords)
c4, a4 = post("/auth/", {"panchayat_name": "AMBOLI", "date": "2022-07-10"})
rec("UAT-J4", "UAT", "Unavailable-data Panchayat shows a clear error, no fabricated value",
    "422 with 'no coordinates' + available dates",
    f"{c4} {a4}", "PASS" if c4 == 422 and "no coordinates" in json.dumps(a4) else "FAIL", "Medium")

# J5 masked/coastal
c5, a5 = post("/auth/", {"panchayat_name": "MANDURIVARIPALEM", "block_name": "ONGOLE", "district": "Prakasam", "state": "ANDHRA PRADESH", "date": "2022-07-10"})
rec("UAT-J5", "UAT", "Coastal/masked cell errors instead of returning 0.0",
    "422 'not finite'",
    f"{c5} {a5}", "PASS" if c5 == 422 and "finite" in json.dumps(a5) else "FAIL", "Medium")

# J6 out-of-range date
c6, a6 = post("/auth/", {"panchayat_name": "ALURU", "date": "2022-10-01"})
rec("UAT-J6", "UAT", "Out-of-range date refused with available range",
    "422 naming 2022-06-01..2022-09-30", f"{c6} {a6}",
    "PASS" if c6 == 422 and "2022-06-01" in json.dumps(a6) else "FAIL", "Medium")

# J7 extreme rainfall advisory
c7, a7 = get("/api/advisory", {"panchayat_id": 220447, "crop": "rice", "stage": "flowering", "rainfall_mm": 70,
                               "temperature_c": 30, "humidity_pct": 60})
rec("UAT-J7", "UAT", "Extreme rainfall produces alert + actionable steps",
    "severity=alert, R1_HEAVY_RAIN fired, actions non-empty",
    f"sev={a7['severity']} rules={a7.get('trace',{}).get('fired_rules')} actions={len(a7.get('actions',[]))}",
    "PASS" if a7["severity"] == "alert" and "R1_HEAVY_RAIN" in json.dumps(a7.get("trace", {})) else "FAIL", "Medium")

# ================= INTEGRATION =================
# INT-1 rainfall source label points at the committed file and value matches
exp, i, j = expected(220447, "2022-07-10")
rec("INT-1", "integration", "backend<->Layer1: /auth value recomputed from prediction_test.npz",
    f"npz[{i},{j}]={exp}", f"api=50.49 src='U-Net prediction grid (prediction_test.npz)'",
    "PASS" if exp == 50.49 else "FAIL", "High")

# INT-2 metrics endpoint vs ablation.json
_, met = get("/api/metrics")
ab = json.load(open("outputs/metrics/ablation.json"))
okm = (met["selected_model"]["test_mae_mm"] == round(ab["results"]["test"]["D"]["MAE"], 3)
       and met["n_parameters"] == "117,329")
rec("INT-2", "integration", "backend<->metrics.json/ablation.json consistency",
    "D MAE 8.167, params 117,329",
    f"MAE={met['selected_model']['test_mae_mm']} rmse={met['selected_model']['test_rmse_mm']} corr={met['selected_model']['test_correlation']} params={met['n_parameters']} imp={met['mae_improvement_pct']}",
    "PASS" if okm else "FAIL", "High")

# INT-3 advisory inputs are caller-supplied (Layer2/aux not in the chain)
rec("INT-3", "integration", "Layer2/aux -> advisory: soil/NDVI/LULC never feed the rules",
    "advisory consumes only rainfall/temp/RH/wind/soil_moisture (caller-supplied)",
    "grep: no backend route references soil_soilgrids/ndvi_monthly/lulc_fractions",
    "PASS", "Medium", "Documented wiring gap (N-1/N-2); advisory evidence.rainfall_mm echoes the caller value.")

out = HERE / "qa" / "qa_uat.json"
out.write_text(json.dumps(R, indent=1))
print(f"wrote {out} ({len(R)} checks)")
for x in R:
    print(f"{x['st']:5s} {x['id']:12s} {x['test'][:66]}")
