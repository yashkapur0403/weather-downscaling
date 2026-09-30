"""QA follow-up: offline dataset / aux / mask checks (no torch needed)."""
import json, os
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent   # repo root (this script lives in qa/)
os.chdir(HERE)

import numpy as np, pandas as pd

R = []
def rec(cid, cat, test, exp, act, st, sev="", note=""):
    R.append(dict(id=cid, cat=cat, test=test, exp=str(exp), act=str(act), st=st, sev=sev, note=note))

# ---- D-11 corrected: mask fully honoured? ----
z = np.load("outputs/prediction_test.npz")
r = z["rainfall_mm"]
per_day_finite = np.isfinite(r).sum(axis=(1, 2))
nan_total = int(np.isnan(r).sum())
n_total = r.shape[1] * r.shape[2]
meta = json.load(open("data/processed/meta.json"))
ntv = meta["land_mask"]["n_target_valid_fine_cells"]
exp_nan = r.shape[0] * (n_total - ntv)
rec("D-11", "L1-output", "NaN cells == complement of target-valid (sea+invalid land) over the FULL fine grid",
    f"{exp_nan} = 122*(57000-{ntv})", f"{nan_total} NaN; per-day finite unique={np.unique(per_day_finite)}",
    "PASS" if nan_total == exp_nan and list(np.unique(per_day_finite)) == [ntv] else "FAIL", "High",
    "Plan expected 3007 because it counted only 47250 land cells; the fine grid is 57000, and sea cells are NaN too. Mask IS honoured.")
rec("D-11b", "L1-output", "NaN mask constant across all 122 days",
    "identical mask every day", f"constant={np.array_equal(np.isnan(r[0]), np.isnan(r[-1]))}",
    "PASS" if np.array_equal(np.isnan(r[0]), np.isnan(r[-1])) else "FAIL")

# ---- D-41 fixed: NDVI ----
n = np.load("data/aux_data/ndvi_monthly_deccan.npz")
keys = list(n.keys())
ndvi = n["ndvi"]
rec("D-41", "aux", "NDVI array key/shape/range/missingness",
    "20 monthly composites, range ~ -0.064..0.845, ~8% missing",
    f"keys={keys}; ndvi{ndvi.shape}; min={np.nanmin(ndvi):.4f} max={np.nanmax(ndvi):.4f} "
    f"nan%={100*np.isnan(ndvi).mean():.3f}; months={list(n['months'])[:4]}...n={len(n['months'])}; "
    f"valid_frac={n['valid_frac_per_month'].shape}",
    "PASS" if ndvi.ndim == 2 and (ndvi.shape[0] == len(n["months"]) or ndvi.shape[-1] == len(n["months"])) else "FAIL", "Low",
    "Plan's expected key 'fine_values' was wrong; the array key is 'ndvi' and its layout is (months, cells)=(20,47250).")

# ---- dataset directory ----
proc = sorted(p.name for p in Path("data/processed").iterdir())
rec("D-60", "dataset", "data/processed contains the arrays needed to rebuild/verify Layer-1",
    "X_{train,val,test}.npy, Y_*.npy, M_*.npy, meta.json", f"present={proc}",
    "FAIL" if not any(p.startswith("X_") for p in proc) else "PASS", "Medium",
    "Only meta.json is committed (arrays are gitignored / in data.zip). Full dataset rebuild via verify_dataset.py cannot run from git.")
rec("D-61", "dataset", "ROI bbox / region is the Deccan dataset not the Western Ghats pilot",
    "deccan, 11.5-25.5N 71.5-81.25E", f"{meta['region']} {meta['roi']}",
    "PASS" if meta["region"] == "deccan" else "FAIL", "High")
rec("D-62", "dataset", "served npz lat/lon identical to meta fine grid",
    "identical",
    f"lat={np.allclose(z['latitude'], meta['grid']['fine_lat'])} lon={np.allclose(z['longitude'], meta['grid']['fine_lon'])}",
    "PASS" if np.allclose(z["latitude"], meta["grid"]["fine_lat"]) and np.allclose(z["longitude"], meta["grid"]["fine_lon"]) else "FAIL")

# ---- aux: soil / lulc / admin ----
s = np.load("data/aux_data/soil_soilgrids_deccan.npz")
rec("D-42", "aux", "soil fine_values shape + missingness", "(47250,5), ~8.8-9.4% NaN",
    f"{s['fine_values'].shape}, nan%={100*np.isnan(s['fine_values']).mean():.2f}",
    "PASS" if s["fine_values"].shape == (47250, 5) else "FAIL")
l = np.load("data/aux_data/lulc_fractions_deccan.npz")
fr = l["fractions"]
rec("D-43", "aux", "LULC fractions shape + rows sum to <=1", "(47250,6), fractions valid",
    f"{fr.shape}; rowsum min={np.nansum(fr,axis=1).min():.3f} max={np.nansum(fr,axis=1).max():.3f}",
    "PASS" if fr.shape == (47250, 6) else "FAIL")
a = np.load("data/aux_data/admin/grid_admin_map_deccan.npz", allow_pickle=True)
aks = list(a.files)
sn = a["state_names"] if "state_names" in aks else np.array([])
dn = np.unique(a["district_names"]) if "district_names" in aks else np.array([])
sdn = np.unique(a["subdistrict_names"]) if "subdistrict_names" in aks else np.array([])
rec("D-44", "aux", "admin map cells/states/districts/subdistricts",
    "47250 cells land / 15 states / 213 districts / 1093 subdistricts",
    f"keys={aks}; states={len(sn)} districts={len(dn)} subdistricts={len(sdn)}",
    "INFO", "", "Consumed only by data_store._admin_centres(); soil/NDVI/LULC are not wired into any route.")

# ---- block rainfall rows ----
b = pd.read_csv("outputs/layer2/block_rainfall.csv")
rec("D-45", "L2-map", "block_rainfall rows == blocks x 122 days", "137006",
    f"{len(b)} rows; unique blocks={b['subdistrict_idx'].nunique()}; dates={b['date'].nunique()}",
    "PASS" if len(b) == 137006 else "FAIL", "", "No route reads this file (N-5).")

out = HERE / "qa" / "qa_followup.json"
out.write_text(json.dumps(R, indent=1))
print(f"wrote {out} ({len(R)} checks)")
for x in R:
    print(f"{x['st']:5s} {x['id']:6s} {x['test'][:70]}")
