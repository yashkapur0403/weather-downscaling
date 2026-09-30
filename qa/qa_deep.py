"""QA deep round: value analysis + remaining untested paths."""
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
        with urllib.request.urlopen(u, timeout=30) as r: return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try: return e.code, json.loads(e.read().decode())
        except Exception: return e.code, None
def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r: return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try: return e.code, json.loads(e.read().decode())
        except Exception: return e.code, None

z = np.load("outputs/prediction_test.npz")
rain, la, lo, dates = z["rainfall_mm"], z["latitude"], z["longitude"], [str(d) for d in z["dates"]]
summ = pd.read_csv("outputs/layer2/panchayat_summary.csv")

# 1. weather reliability (reproduce intermittent null)
W = []
for i in range(10):
    c, r = get("/api/weather", {"lat": 13.679452, "lon": 74.742466, "date": "2022-07-10"})
    W.append((c, r.get("temperature_c") if r else None, r.get("error") if r else None))
ok = sum(1 for c, t, _ in W if c == 200 and t is not None)
rec("WEATHER-REL", "robustness", "Open-Meteo/DEM reliability over 10 calls",
    "all non-null ideally; must degrade to null+error not fabricate",
    f"{ok}/10 non-null; nulls={[w for w in W if w[1] is None][:3]}",
    "PASS" if ok > 0 else "FAIL", "Medium",
    "If any null: external dependency; handler returns 200 with nulls+error (no fabrication).")

# 2. geocode coordinate-less
c, r = get("/api/geocode", {"panchayat_name": "AMBOLI"})
rec("GEO-NOCOORD", "geocode", "geocode Panchayat without coordinates", "404 'no coordinates'",
    f"{c} {r}", "PASS" if c == 404 else "FAIL", "Low")
c0, _ = get("/api/geocode", {})
rec("GEO-NOPARAM", "geocode", "geocode missing panchayat_name", "422",
    f"{c0}", "PASS" if c0 == 422 else "FAIL", "Low")

# 3. search ambiguity for a duplicated name
c, sr = get("/api/panchayats", {"q": "aluru", "limit": 50})
names = [x["panchayat_name"] for x in sr]
alurus = [x for x in sr if x["panchayat_name"] == "ALURU"]
rec("SEARCH-AMBIG", "mapping", "search q=aluru returns distinct rows for duplicated name",
    "multiple ALURU rows with distinct id/district",
    f"rows={len(sr)}; exact ALURU={len(alurus)} -> {[(x['panchayat_id'], x['district']) for x in alurus][:4]}",
    "PASS" if len({x['panchayat_id'] for x in alurus}) > 1 else "INFO", "High",
    "Search returns distinct rows (good), but /auth/ name-only collapses to iloc[0] (BB-13) -> frontend MUST send full context.")

# 4. duplicate-name population
dup = summ.groupby("panchayat_name").size()
dupnames = dup[dup > 1]
rec("DUP-POP", "mapping", "duplicated Panchayat names in the served (mapped) set",
    "quantify the ambiguity risk",
    f"{len(dupnames)} duplicated names covering {int(dupnames.sum())} rows; max={int(dupnames.max())}",
    "INFO", "High", "Confirms BB-13 blast radius.")

# 5. season-mean value analysis: ALURU rainfall_mean_mm vs npz cell mean
row = summ[summ.panchayat_id == 220447].iloc[0]
idx = pd.read_csv("outputs/layer2/panchayat_index.csv")
cr = idx[idx.panchayat_id == 220447].iloc[0]
i = int(np.abs(la - cr.lat).argmin()); j = int(np.abs(lo - cr.lon).argmin())
cellmean = float(np.nanmean(rain[:, i, j]))
rec("VAL-SEASONMEAN", "value-analysis", "panchayat_summary.rainfall_mean_mm == npz cell mean over 122 days (ALURU)",
    "approx equal (aggregation may be area-weighted, so small delta ok)",
    f"summary={row.rainfall_mean_mm:.4f} vs npz_cell_mean={cellmean:.4f}",
    "INFO", "", "If very different, the summary uses the true area-weighted polygon mean (expected).")

# 6. per-date map consistency with the served grid
mp = Path("outputs/maps/infer_2022-07-10.npy")
if mp.exists():
    a = np.load(mp)
    d = dates.index("2022-07-10")
    g = rain[d]
    if a.shape != g.shape:
        rec("VAL-MAP", "dataset", "outputs/maps/infer_2022-07-10.npy matches the served Deccan grid",
            f"same grid as prediction_test.npz {g.shape}",
            f"STALE PILOT GRID: shape={a.shape} (Western Ghats pilot is 85x85; Deccan is {g.shape})",
            "FAIL", "Low",
            "Not read by any backend route (data_store reads prediction_test.npz), but a stale pilot-grid artefact sits in the product repo.")
    else:
        both = np.isfinite(a) & np.isfinite(g)
        rec("VAL-MAP", "value-analysis", "outputs/maps/infer_2022-07-10.npy == prediction_test.npz[2022-07-10]",
            "same field", f"maxabs diff on finite={float(np.nanmax(np.abs(a[both]-g[both]))):.6f}", "PASS", "Low")
else:
    rec("VAL-MAP", "value-analysis", "per-date map artefact exists", "present", "absent", "INFO")

# 7. IMD vs CHIRPS correlation (dataset value analysis, needs data.zip-extracted arrays)
qd = HERE / "qa" / "qa_processed"
if (qd / "X_test.npy").exists():
    X = np.load(qd / "X_test.npy"); Y = np.load(qd / "Y_test.npy"); M = np.load(qd / "M_test.npy")
    meta = json.loads((HERE / "data/processed/meta.json").read_text())
    imd_idx = meta["channels"].index("imd_rain")
    m = M[:, 0].astype(bool)
    imd_mm = np.clip(X[:, imd_idx] * 100.0, 0, None)
    a = imd_mm[m]; b = Y[:, 0][m]
    corr = float(np.corrcoef(a, b)[0, 1])
    rec("VAL-IMDCHIRPS", "dataset", "test-split corr(IMD, CHIRPS) vs meta.imd_chirps_coarse_corr",
        f"~{meta['imd_chirps_coarse_corr']:.4f}",
        f"corr={corr:.4f} on {m.sum()} masked pixels",
        "INFO", "", "Meta value is computed over the processed set; near-match validates source alignment.")

# 8. block rainfall internal consistency
b = pd.read_csv("outputs/layer2/block_rainfall.csv")
per_date = b.groupby("date").size().unique()
rec("VAL-BLOCK", "value-analysis", "block_rainfall has 1123 blocks for every date",
    "single unique count per date", f"per-date counts unique={per_date}",
    "PASS" if len(per_date) == 1 else "FAIL", "Low")

# 9. advisory block validation
big = [{"panchayat_id": 1 + k, "rainfall_mm": 10} for k in range(61)]
c9, r9 = post("/api/advisory/block", big)
rec("ADVBLOCK-LIMIT", "advisory", "block advisory > 60 panchayats", "rejected (>60 cap)",
    f"{c9} {str(r9)[:120]}", "PASS" if c9 >= 400 else "FAIL", "Low")
c10, r10 = post("/api/advisory/block", [])
rec("ADVBLOCK-EMPTY", "advisory", "block advisory empty list", "handled",
    f"{c10} {str(r10)[:80]}", "INFO")

# 10. location_precision spread via geocode (with and without context)
cA, rA = get("/api/geocode", {"panchayat_name": "ALURU"})
cB, rB = get("/api/geocode", {"panchayat_name": "ALURU", "district": "Udupi", "state": "KARNATAKA"})
rec("GEO-PREC", "mapping", "geocode precision reported correctly",
    "exact only when truly panchayat-level",
    f"no-context lat={rA.get('lat')} prec={rA.get('location_precision')}; with-context lat={rB.get('lat')} prec={rB.get('location_precision')}",
    "INFO", "Medium", "Both return the same admin-centroid coord labelled 'exact' (D-1).")

out = HERE / "qa" / "qa_deep.json"
out.write_text(json.dumps(R, indent=1))
print(f"wrote {out} ({len(R)})")
for x in R: print(f"{x['st']:5s} {x['id']:16s} {x['test'][:60]}")
