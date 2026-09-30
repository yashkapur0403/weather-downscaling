"""Logical validation - HTTP layer: Panchayat identity binding, cache/state
isolation, and whether Layer 3 (advisory) is provably bound to the Panchayat
and date whose rainfall it uses.

Run against the live backend on 127.0.0.1:8000. Writes qa_logic_http.json.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent   # repo root (this script lives in qa/)
BASE = "http://127.0.0.1:8000"
DATE = "2022-07-10"
OUT = {}

z = np.load(HERE / "outputs" / "prediction_test.npz")
NPZ_DATES = [str(x) for x in z["dates"]]
RAIN, LAT, LON = z["rainfall_mm"], z["latitude"], z["longitude"]
di = NPZ_DATES.index(DATE)
idx = pd.read_csv(HERE / "outputs" / "layer2" / "panchayat_index.csv")
summ = pd.read_csv(HERE / "outputs" / "layer2" / "panchayat_summary.csv")
coords = {int(r.panchayat_id): (float(r.lat), float(r.lon)) for r in idx.itertuples()}


def cell_val(pid, date=DATE):
    lat, lon = coords[pid]
    i = int(np.abs(LAT - lat).argmin())
    j = int(np.abs(LON - lon).argmin())
    v = float(RAIN[NPZ_DATES.index(date), i, j])
    return None if not np.isfinite(v) else round(v, 2)


c = httpx.Client(base_url=BASE, timeout=40)

# ---------------------------------------------------------- 1 identity binding
print("=" * 78)
print("1. IDENTITY BINDING (a unique name must resolve to ITS OWN coordinates)")
print("=" * 78)
names = summ.groupby(summ.panchayat_name.str.upper()).size()
uniq = set(names[names == 1].index)
ident = []
for meth in ("direct_grid", "area_weighted"):
    sub = summ[(summ.mapping_method == meth) & summ.panchayat_name.str.upper().isin(uniq)]
    for r in sub.head(6).itertuples():
        exp = cell_val(int(r.panchayat_id))
        if exp is None:
            continue
        body = {"panchayat_name": r.panchayat_name, "block_name": r.block_name,
                "district": r.district, "state": r.state, "date": DATE}
        resp = c.post("/auth/", json=body)
        got = resp.json().get("prediction", {}).get("rainfall_mm") if resp.status_code == 200 else None
        ident.append({"panchayat_id": int(r.panchayat_id), "name": r.panchayat_name,
                      "district": r.district, "status": resp.status_code,
                      "expected_from_own_cell": exp, "served": got,
                      "match": got is not None and abs(got - exp) < 0.011,
                      "prec": resp.json().get("sources", {}).get("location_precision")
                      if resp.status_code == 200 else None})
OUT["identity_binding"] = ident
for t in ident:
    print(f"  pid={t['panchayat_id']} {str(t['name'])[:22]:22s} {str(t['district'])[:12]:12s} "
          f"exp={t['expected_from_own_cell']:7.2f} served={t['served']} match={t['match']} "
          f"prec={t['prec']} HTTP={t['status']}")

# ------------------------------------------------- 2 ambiguity is refused
print("\n" + "=" * 78)
print("2. AMBIGUOUS NAME-ONLY REQUESTS")
print("=" * 78)
amb = summ.groupby(summ.panchayat_name.str.upper()).size().sort_values(ascending=False)
dupname = amb.index[0]
rows = summ[summ.panchayat_name.str.upper() == dupname]
r1 = c.post("/auth/", json={"panchayat_name": dupname, "date": DATE})
r2 = c.post("/auth/", json={"panchayat_name": dupname, "district": rows.iloc[0].district,
                            "block_name": rows.iloc[0].block_name, "date": DATE})
amb_res = {
    "name": dupname, "n_rows_sharing_name": int(len(rows)),
    "name_only_status": r1.status_code,
    "name_only_candidates": len(r1.json().get("detail", {}).get("candidates", []))
    if r1.status_code == 409 else None,
    "contextual_status": r2.status_code,
    "contextual_value": r2.json().get("prediction", {}).get("rainfall_mm") if r2.status_code == 200 else None,
    "contextual_expected": cell_val(int(rows.iloc[0].panchayat_id)),
    "geocode_name_only_status": c.post("/api/geocode", json={"panchayat_name": dupname}).status_code,
}
OUT["ambiguity"] = amb_res
print(json.dumps(amb_res, indent=1))

# --------------------------------------------- 3 cache / state isolation
print("\n" + "=" * 78)
print("3. CACHE / STATE ISOLATION (sequenced requests)")
print("=" * 78)
seq_pids = [int(x) for x in summ[summ.mapping_method == "direct_grid"].panchayat_id.head(4)]
seq = []
seq_plan = []
for i, pid in enumerate(seq_pids):
    r = summ[summ.panchayat_id == pid].iloc[0]
    d = NPZ_DATES[(i * 7) % len(NPZ_DATES)]
    seq_plan.append((pid, r, d))
for pid, r, d in seq_plan:
    exp = cell_val(pid, d)
    resp = c.post("/auth/", json={"panchayat_name": r.panchayat_name, "district": r.district,
                                  "block_name": r.block_name, "state": r.state, "date": d})
    got = resp.json().get("prediction", {}).get("rainfall_mm") if resp.status_code == 200 else None
    seq.append({"panchayat_id": pid, "name": r.panchayat_name, "date": d,
                "expected": exp, "served": got,
                "match": got is not None and exp is not None and abs(got - exp) < 0.011})
OUT["sequence_isolation"] = seq
for s in seq:
    print(f"  {s['date']} pid={s['panchayat_id']} {str(s['name'])[:20]:20s} "
          f"exp={s['expected']} served={s['served']} match={s['match']}")

# concurrent isolation
def one(s):
    pid, r, d = s
    resp = c.post("/auth/", json={"panchayat_name": r.panchayat_name, "district": r.district,
                                  "block_name": r.block_name, "state": r.state, "date": d})
    got = resp.json().get("prediction", {}).get("rainfall_mm") if resp.status_code == 200 else None
    return {"panchayat_id": pid, "date": d, "expected": cell_val(pid, d), "served": got}


with ThreadPoolExecutor(max_workers=8) as ex:
    conc = list(ex.map(one, seq_plan))
OUT["concurrent_isolation"] = conc
bad = [x for x in conc if x["served"] is None or x["expected"] is None
       or abs(x["served"] - x["expected"]) >= 0.011]
print(f"  concurrent: {len(conc) - len(bad)}/{len(conc)} matched under 8-way concurrency")

# ------------------------------------------------------ 4 zero-rain reachable
print("\n" + "=" * 78)
print("4. IS AN EXACT 0.00 mm ANSWER REACHABLE? (frontend falsy-zero fallback)")
print("=" * 78)
zero_pids = []
for r in summ[summ.mapping_method == "direct_grid"].head(4000).itertuples():
    v = cell_val(int(r.panchayat_id))
    if v == 0.0:
        zero_pids.append((int(r.panchayat_id), r))
    if len(zero_pids) >= 3:
        break
zres = []
for pid, r in zero_pids:
    resp = c.post("/auth/", json={"panchayat_name": r.panchayat_name, "district": r.district,
                                  "block_name": r.block_name, "date": DATE})
    body = resp.json()
    zres.append({"panchayat_id": pid, "name": r.panchayat_name, "status": resp.status_code,
                 "rainfall_mm": body.get("prediction", {}).get("rainfall_mm"),
                 "answer": body.get("answer"),
                 "search_result_season_mean": float(r.rainfall_mean_mm)})
OUT["zero_rain_case"] = zres
print(json.dumps(zres, indent=1))
print("  => a legitimate 0.0 mm answer exists; the frontend uses `x || fallback` in")
print("     HomePage.tsx, so 0.0 is falsy and would be replaced by the previous")
print("     selection's rainfall.")

# ----------------------------------------- 5 Layer-3 binding to rainfall/date
print("\n" + "=" * 78)
print("5. IS THE ADVISORY BOUND TO THE PANCHAYAT / DATE / RAINFALL?")
print("=" * 78)
pid_probe = next(pid for pid, r in zero_pids) if zero_pids else seq_pids[0]
correct = cell_val(pid_probe)
adv = {}
for label, mm in (("correct", correct), ("zero", 0.0), ("inflated", 999.0)):
    resp = c.get("/api/advisory", params={"panchayat_id": pid_probe, "crop": "rice",
                                          "stage": "flowering", "rainfall_mm": mm,
                                          "temperature_c": 30.0, "date": DATE,
                                          "panchayat_name": "PROBE"})
    b = resp.json()
    adv[label] = {"status": resp.status_code, "severity": b.get("severity"),
                  "evidence_rainfall": b.get("evidence", {}).get("rainfall_mm"),
                  "risk_level": b.get("evidence", {}).get("risk_level"),
                  "data_date": b.get("data_date"),
                  "fired": [r["rule_id"] for r in b.get("fired_rules", [])],
                  "n_actions": len(b.get("actions", []))}
# out-of-range date accepted?
resp = c.get("/api/advisory", params={"panchayat_id": pid_probe, "crop": "rice",
                                      "stage": "flowering", "rainfall_mm": 50.0,
                                      "date": "1999-01-01", "panchayat_name": "PROBE"})
adv["out_of_range_date"] = {"status": resp.status_code,
                            "data_date": resp.json().get("data_date")}
# mismatched panchayat_id (aux from a different panchayat)
resp = c.get("/api/advisory", params={"panchayat_id": 999999999, "crop": "rice",
                                      "stage": "flowering", "rainfall_mm": 50.0,
                                      "date": DATE, "panchayat_name": "WRONGID"})
adv["bogus_panchayat_id"] = {"status": resp.status_code,
                             "aux": resp.json().get("evidence", {}).get("aux"),
                             "severity": resp.json().get("severity")}
OUT["layer3_binding"] = adv
print(json.dumps(adv, indent=1))

# --------------------------------------------------------------- 6 enums, errors
print("\n" + "=" * 78)
print("6. ENUM / ERROR BEHAVIOUR")
print("=" * 78)
enums = {}
enums["GET_mustard"] = c.get("/api/advisory", params={"panchayat_id": pid_probe, "crop": "mustard",
                                                      "rainfall_mm": 30.0}).status_code
enums["GET_ripening"] = c.get("/api/advisory", params={"panchayat_id": pid_probe, "crop": "rice",
                                                       "stage": "ripening", "rainfall_mm": 30.0}).status_code
enums["GET_general"] = c.get("/api/advisory", params={"panchayat_id": pid_probe, "crop": "general",
                                                      "stage": "general", "rainfall_mm": 30.0}).status_code
enums["GET_grain_filling_rejected"] = c.get("/api/advisory",
                                            params={"panchayat_id": pid_probe, "stage": "grain_filling",
                                                    "rainfall_mm": 30.0}).status_code
for crop in ("mustard", "ripening"):
    enums[f"POST_{crop}"] = c.post("/api/advisory", json={"data": {
        "panchayat": "P", "crop": "mustard" if crop == "mustard" else "rice",
        "stage": "general" if crop == "mustard" else "ripening", "rainfall_mm": 30.0}}).status_code
# out-of-domain date, NaN cell
r_bad = c.post("/auth/", json={"panchayat_name": "ALURU", "district": "Udupi",
                               "block_name": "KUNDAPURA", "date": "2021-07-10"})
enums["auth_out_of_range_date"] = r_bad.status_code
nan_pid = None
for r in summ[summ.mapping_method == "nearest_fallback"].head(300).itertuples():
    if cell_val(int(r.panchayat_id)) is None:
        nan_pid = (int(r.panchayat_id), r)
        break
if nan_pid:
    pid, r = nan_pid
    rr = c.post("/auth/", json={"panchayat_name": r.panchayat_name, "district": r.district,
                                "block_name": r.block_name, "date": DATE})
    enums["auth_masked_cell"] = {"status": rr.status_code, "panchayat_id": pid,
                                 "name": r.panchayat_name,
                                 "summary_season_mean_mm": float(r.rainfall_mean_mm),
                                 "body": str(rr.json())[:160]}
enums["health"] = c.get("/").status_code
OUT["enums_errors"] = enums
print(json.dumps(enums, indent=1, default=str))

json.dump(OUT, open(HERE / "qa" / "qa_logic_http.json", "w"), indent=1, default=str)
print("\n-> qa_logic_http.json")
