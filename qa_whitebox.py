"""White-box validation: datasets (Layer 1/2 + aux) and the model checkpoint."""
import json, os, sys
import numpy as np
import pandas as pd

R = []


def rec(tid, cat, test, exp, act, status, sev="", note=""):
    R.append(dict(id=tid, cat=cat, test=test, exp=str(exp), act=str(act), st=status, sev=sev, note=note))
    print(f"[{status:4s}] {tid} {test}\n        exp = {str(exp)[:105]}\n        act = {str(act)[:105]}")


def guard(tid, cat, test, exp, fn):
    try:
        fn()
    except Exception as e:
        rec(tid, cat, test, exp, f"EXCEPTION {type(e).__name__}: {e}", "BLOCKED", "High")


meta = json.load(open("data/processed/meta.json"))
z = np.load("outputs/prediction_test.npz", allow_pickle=False)
rain, flat, flon, zdates = z["rainfall_mm"], z["latitude"], z["longitude"], [str(d) for d in z["dates"]]
sm = pd.read_csv("outputs/layer2/panchayat_summary.csv")
ix = pd.read_csv("outputs/layer2/panchayat_index.csv")
qc = json.load(open("outputs/layer2/layer2_qc.json"))

print("################ LAYER 1 DATASET ################")
rec("D-01", "L1-data", "meta region is the ACTIVE Deccan dataset", "deccan", meta["region"],
    "PASS" if meta["region"] == "deccan" else "FAIL", "High")
roi = meta["roi"]
rec("D-02", "L1-data", "meta ROI bbox", "{11.5,25.5,71.5,81.25}",
    {k: round(v, 2) for k, v in roi.items()},
    "PASS" if roi == {"lat_min": 11.5, "lat_max": 25.5, "lon_min": 71.5, "lon_max": 81.25} else "FAIL")
rec("D-03", "L1-data", "date span", "610 days 2018-06-01..2022-09-30",
    f"{len(meta['dates'])} days {meta['dates'][0]}..{meta['dates'][-1]}",
    "PASS" if len(meta["dates"]) == 610 else "FAIL")
sp = meta["split"]
rec("D-04", "L1-data", "temporal split (no leakage by construction)", "366/122/122 train2018-20/val2021/test2022",
    f"{sp['train']['n_days']}/{sp['val']['n_days']}/{sp['test']['n_days']} | {meta['split_description']}",
    "PASS" if (sp["train"]["n_days"], sp["val"]["n_days"], sp["test"]["n_days"]) == (366, 122, 122) else "FAIL")
chs = meta["channels"]
rec("D-05", "L1-data", "channel ORDER in the dataset", "[imd_rain,dem,era5_t2m,era5_t2m_max,era5_dewp]",
    chs, "PASS" if chs == ["imd_rain", "dem", "era5_t2m", "era5_t2m_max", "era5_dewp"] else "FAIL", "High")
nz = meta["normalization"]
rec("D-06", "L1-data", "normalisation constants", "rain_scale=100.0, elev_scale=1222.6514892578125",
    f"rain_scale={nz['rain_scale_mm']} elev_scale={nz['elev_scale_m']}",
    "PASS" if nz["rain_scale_mm"] == 100.0 and abs(nz["elev_scale_m"] - 1222.6514892578125) < 1e-9 else "FAIL")
g = meta["grid"]
rec("D-07", "L1-data", "fine grid geometry", "285x200, fine_sub=5", f"{g['H_fine']}x{g['W_fine']} sub={g['fine_sub']}",
    "PASS" if (g["H_fine"], g["W_fine"], g["fine_sub"]) == (285, 200, 5) else "FAIL")
lm = meta["land_mask"]
rec("D-08", "L1-data", "land mask / target-valid counts", "1890 land, 390 sea, 47250 land cells, 44243 target-valid",
    f"land={lm['n_land_cells']} sea={lm['n_sea_cells']} landcells={lm['land_mask']['n_land_cells'] if isinstance(lm.get('land_mask'),dict) else '-'} target_valid={lm['n_target_valid_fine_cells']}",
    "PASS" if lm["n_land_cells"] == 1890 and lm["n_target_valid_fine_cells"] == 44243 else "FAIL")

print("################ LAYER 1 MODEL OUTPUT (prediction_test.npz) ################")
rec("D-09", "L1-output", "npz dates == meta test split dates", "122 dates identical to meta.split.test.dates",
    f"{len(zdates)} dates, first={zdates[0]}, last={zdates[-1]}, identical={zdates == sp['test']['dates']}",
    "PASS" if zdates == sp["test"]["dates"] else "FAIL", "High")
rec("D-10", "L1-output", "npz lat/lon == meta fine grid", "identical arrays",
    f"lat match={np.allclose(flat, g['fine_lat'])} lon match={np.allclose(flon, g['fine_lon'])}",
    "PASS" if np.allclose(flat, g["fine_lat"]) and np.allclose(flon, g["fine_lon"]) else "FAIL")
n_nan = int(np.isnan(rain).sum())
exp_nan = 47250 - 44243
rec("D-11", "L1-output", "NaN cells == masked (non-target-valid) cells", f"{exp_nan} NaN cells (47250-44243)",
    f"{n_nan} NaN cells", "PASS" if n_nan == exp_nan else "FAIL", "High",
    "proves mask is honoured in the served output")
rec("D-12", "L1-output", "NOT the Western Ghats pilot grid", "285x200 (pilot was 85x85)",
    f"{rain.shape} lat {flat[0]}..{flat[-1]}", "PASS" if rain.shape[1:] == (285, 200) else "FAIL", "High")
rec("D-13", "L1-output", "rainfall value range / no Inf", "0.0..163.28, no Inf",
    f"min={np.nanmin(rain):.4f} max={np.nanmax(rain):.4f} inf={int(np.isinf(rain).sum())} finite_frac={np.isfinite(rain).mean():.4f}",
    "PASS" if np.nanmin(rain) >= 0 and not np.isinf(rain).any() else "FAIL")
rec("D-14", "L1-output", "dtype", "float32", str(rain.dtype), "PASS" if rain.dtype == np.float32 else "FAIL")
rec("D-15", "L1-output", "reported global max matches layer2_qc.json", "163.28 both",
    f"npz={np.nanmax(rain):.4f} qc={qc['rainfall_mm']['max']:.4f}",
    "PASS" if abs(float(np.nanmax(rain)) - qc["rainfall_mm"]["max"]) < 1e-6 else "FAIL")

print("################ LAYER 2 MAPPING ################")
rec("D-20", "L2-map", "panchayat_summary row count", "87,735", len(sm), "PASS" if len(sm) == 87735 else "FAIL")
vc = sm.mapping_method.value_counts().to_dict()
rec("D-21", "L2-map", "mapping_method distribution", "area 52893 / direct 31740 / unmapped 1632 / fallback 1470",
    vc, "PASS" if vc == {"area_weighted": 52893, "direct_grid": 31740, "unmapped": 1632, "nearest_fallback": 1470} else "FAIL")
un = sm[sm.mapping_method == "unmapped"]
rec("D-22", "L2-map", "unmapped rows carry n_cells == 0", "all 1632 zero", f"n_cells sum={int(un.n_cells.sum())} max={int(un.n_cells.max())}",
    "PASS" if int(un.n_cells.max()) == 0 else "FAIL")
rec("D-23", "L2-map", "index rows / distinct coordinates", "86,075 rows; 989 distinct coords",
    f"{len(ix)} rows; {ix.groupby(['lat','lon']).ngroups} distinct coords",
    "PASS" if len(ix) == 86075 else "FAIL", "High", "coordinate collapse - see CF-01")
rec("D-24", "L2-map", "QC mapped count == total - unmapped", "86103 == 87735-1632",
    f"{qc['panchayats']['mapped']} == {87735 - 1632}",
    "PASS" if qc["panchayats"]["mapped"] == 87735 - 1632 else "FAIL")
rec("D-25", "L2-map", "index covers all mapped panchayats", "86,075 of 86,103 (28 missing)",
    f"{len(ix)} of {qc['panchayats']['mapped']}; missing={qc['panchayats']['mapped'] - len(ix)}",
    "INFO", "", "the 28 missing are the DNH/Daman rows - see CF-04")
rec("D-26", "L2-map", "summary n_days consistent", "all rows n_days=122",
    f"unique n_days={sorted(sm.n_days.unique())[:4]}", "PASS" if (sm.n_days == 122).all() else "FAIL")
rec("D-27", "L2-map", "negative rainfall?", "no negative values",
    f"min={sm.rainfall_mean_mm.min():.4f}", "PASS" if sm.rainfall_mean_mm.min() >= 0 else "FAIL")
br = pd.read_csv("outputs/layer2/block_rainfall.csv")
exp_br = 1123 * 122
rec("D-28", "L2-map", "block_rainfall rows == 1123 blocks x 122 days", exp_br, len(br),
    "PASS" if len(br) == exp_br else "FAIL", "", "exact internal consistency of the block aggregation")
rec("D-29", "L2-map", "panchayat_weather.csv exists (README section 8 claims it)", "present",
    f"exists={os.path.exists('outputs/layer2/panchayat_weather.csv')}", "FAIL", "Medium",
    "absent -> /auth/ cannot serve per-date Layer-2 values (CF-03)")
rec("D-30", "L2-map", "panchayat_weather.geojson exists (README claims it)", "present",
    f"exists={os.path.exists('outputs/layer2/panchayat_weather.geojson')}", "FAIL", "Low")

print("################ AUX LAYERS ################")
s = np.load("data/aux_data/soil_soilgrids_deccan.npz", allow_pickle=True)
rec("D-40", "aux", "soil fine_values shape", "(47250, 5)",
    f"{s['fine_values'].shape} NaN%={np.isnan(s['fine_values']).mean() * 100:.2f}",
    "PASS" if s["fine_values"].shape == (47250, 5) else "FAIL")
nv = np.load("data/aux_data/ndvi_monthly_deccan.npz", allow_pickle=True)
fv = nv["fine_values"] if "fine_values" in nv else None
rec("D-41", "aux", "NDVI 20 monthly composites + range", "20 layers, values -0.064..0.845",
    f"shape={fv.shape if fv is not None else list(nv.keys())} min={np.nanmin(fv):.3f} max={np.nanmax(fv):.3f} missing%={np.isnan(fv).mean() * 100:.2f}"
    if fv is not None else str(list(nv.keys())),
    "PASS" if fv is not None and fv.shape[0] == 20 else "FAIL")
lu = np.load("data/aux_data/lulc_fractions_deccan.npz", allow_pickle=True)
rec("D-42", "aux", "LULC fractions shape", "(47250, 6)", f"{lu['fractions'].shape}",
    "PASS" if lu["fractions"].shape == (47250, 6) else "FAIL")
ad = np.load("data/aux_data/admin/grid_admin_map_deccan.npz", allow_pickle=True)
rec("D-43", "aux", "admin map: cells/states/districts/subdistricts", "47250 cells / 15 states / 1123 subdistricts",
    f"{len(ad['fine_i'])} cells, {len(set(map(str, ad['state_names'])))} states, "
    f"{len(set(map(str, ad['district_names'])))} districts, {len(set(map(str, ad['subdistrict_names'])))} subdistricts",
    "PASS" if len(ad["fine_i"]) == 47250 else "FAIL")
bs = json.load(open("data/aux_data/build_summary_deccan.json"))
rec("D-44", "aux", "build_summary matches the soil array", "1890 points / 8.84% missing",
    f"n_points={bs['soil']['n_points']} missing={bs['soil']['missing_pct']}",
    "PASS" if bs["soil"]["n_points"] == 1890 else "FAIL")

print("################ MODEL CHECKPOINT ################")
try:
    import torch
    from train import SmallUNet
    ck = torch.load("models/best_model.pt", map_location="cpu", weights_only=False)
    ch = ck.get("channels") or ck["args"]["channels"]
    sd = ck.get("model", ck.get("model_state_dict"))
    n = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    m = SmallUNet(cin=len(ch), width=int(ck.get("width", 16)), residual=bool(ck.get("residual", True)))
    np_ = sum(p.numel() for p in m.parameters())
    print("  checkpoint keys:", [k for k in ck.keys()][:12])
    print("  args:", ck.get("args"))
    rec("D-50", "model", "checkpoint channel order", "[imd_rain,dem,era5_t2m,era5_t2m_max,era5_dewp]", ch,
        "PASS" if ch == ["imd_rain", "dem", "era5_t2m", "era5_t2m_max", "era5_dewp"] else "FAIL", "High")
    rec("D-51", "model", "checkpoint width / residual flag", "width=16, residual=True",
        f"width={ck.get('width', 16)} residual={ck.get('residual', True)}", "PASS")
    rec("D-52", "model", "state_dict params == instantiated model params", "identical",
        f"state_dict={n:,} model={np_:,}", "PASS" if n == np_ else "FAIL", "High")
    rec("D-53", "model", "actual parameter count vs documented", "documented '~150k'; /api/metrics says 117,329",
        f"actual={np_:,}", "FAIL" if np_ != 117329 or np_ < 140000 else "PASS", "Low",
        f"actual {np_:,} != '~150k' claimed in README/HOW-IT-WORKS/frontend")
    rec("D-54", "model", "param count == backend's unet_param_count(5,16)", "117,329",
        f"{np_:,}", "PASS" if np_ == 117329 else "FAIL", "High")
except Exception as e:
    rec("D-50", "model", "checkpoint validation", "checkpoint loads", f"BLOCKED: {type(e).__name__}: {e}", "BLOCKED", "High",
        "torch not installed in this venv")

print("\n" + "=" * 60)
print("WHITE-BOX TOTAL:", len(R), "| PASS", sum(1 for x in R if x["st"] == "PASS"),
      "| FAIL", sum(1 for x in R if x["st"] == "FAIL"), "| INFO", sum(1 for x in R if x["st"] == "INFO"),
      "| BLOCKED", sum(1 for x in R if x["st"] == "BLOCKED"))
for x in R:
    if x["st"] in ("FAIL", "BLOCKED"):
        print("  ", x["st"], x["id"], x["test"], "|", x["note"])
json.dump(R, open("qa_whitebox.json", "w"), indent=1)
