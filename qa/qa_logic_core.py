"""Logical validation - Layer 1 -> Layer 2 numerics, spatial logic, temporal
index logic, model-output sanity and administrative joins.

Everything is recomputed independently from the raw artefacts (the NPZ, the
processed arrays, the LGD polygons) and compared with what the backend's own
Store serves. Writes qa_logic_core.json.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent   # repo root (this script lives in qa/)
ROOT = HERE
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))
import data_store as DS  # noqa: E402
from live_infer import DataUnavailable, OutsideDomain, sample_grid  # noqa: E402

DATE = "2022-07-10"
OUT = {}
meta = json.loads((ROOT / "data" / "processed" / "meta.json").read_text())
z = np.load(ROOT / "outputs" / "prediction_test.npz")
NPZ_DATES = [str(x) for x in z["dates"]]
RAIN = z["rainfall_mm"]
LAT, LON = z["latitude"], z["longitude"]
Y = np.load(ROOT / "data" / "processed" / "Y_test.npy")
M = np.load(ROOT / "data" / "processed" / "M_test.npy")
X = np.load(ROOT / "data" / "processed" / "X_test.npy", mmap_mode="r")
if M.ndim == 4:
    M = M[:, 0]
RS = float(meta["normalization"]["rain_scale_mm"])
store = DS.Store(ROOT)


def cell_of(lat, lon):
    return int(np.abs(LAT - lat).argmin()), int(np.abs(LON - lon).argmin())


print("=" * 78)
print("1. LAYER-1 -> LAYER-2 NUMERIC TRACES")
print("=" * 78)
idx = pd.read_csv(ROOT / "outputs" / "layer2" / "panchayat_index.csv")
summ = pd.read_csv(ROOT / "outputs" / "layer2" / "panchayat_summary.csv")
idx_by_pid = {int(r.panchayat_id): (float(r.lat), float(r.lon)) for r in idx.itertuples()}

# spread the sample across states / mapping methods
sample = []
summ["_n_cells"] = summ["n_cells"].fillna(0)
for meth in ("direct_grid", "area_weighted", "nearest_fallback"):
    sub = summ[summ.mapping_method == meth]
    for st in ("KARNATAKA", "MAHARASHTRA", "TELANGANA", "ANDHRA PRADESH", "TAMIL NADU"):
        s2 = sub[sub.state == st]
        if len(s2):
            sample.append(int(s2.iloc[len(s2) // 2].panchayat_id))
extra = [int(summ[summ.n_cells == 1].iloc[0].panchayat_id),
         int(summ[summ.n_cells >= 10].iloc[0].panchayat_id),
         int(summ.n_cells.idxmax().__index__() and summ.iloc[int(summ.n_cells.argmax())].panchayat_id)]
sample = list(dict.fromkeys(sample + extra))
print(f"sampled panchayat ids: {sample}")

traces = []
for pid in sample:
    if pid not in idx_by_pid:
        continue
    lat0, lon0 = idx_by_pid[pid]
    row = summ[summ.panchayat_id == pid].iloc[0]
    di = NPZ_DATES.index(DATE)
    i, j = cell_of(lat0, lon0)
    raw = float(RAIN[di, i, j])
    # orientation / indexing controls: these MUST give a different answer
    i_sw, j_sw = cell_of(lon0, lat0)
    swapped_coord = float(RAIN[di, i_sw, j_sw])
    transposed = float(RAIN[di, j % RAIN.shape[1], i % RAIN.shape[2]])
    rec = {
        "panchayat_id": pid, "panchayat_name": row.panchayat_name,
        "state": row.state, "district": row.district, "block": row.block_name,
        "mapping_method": row.mapping_method, "n_cells": int(row.n_cells),
        "coord_from_index": [lat0, lon0], "cell": [i, j],
        "npz_raw": raw, "npz_rounded": round(raw, 2),
        "npz_cell_is_nan": bool(not np.isfinite(raw)),
        "summary_rainfall_mean_mm": float(row.rainfall_mean_mm),
        "control_latlon_swapped": swapped_coord,
        "control_transposed": transposed,
        "controls_differ": bool(abs(swapped_coord - raw) > 1e-9
                                 and abs(transposed - raw) > 1e-9),
    }
    try:
        served = store.grid_rainfall(lat0, lon0, DATE)
        rec.update(served_rainfall_mm=served["rainfall_mm"],
                   served_cell=[served["fine_i"], served["fine_j"]],
                   served_source=served["source"],
                   served_error=None,
                   match=bool(abs(round(raw, 2) - served["rainfall_mm"]) < 1e-9))
    except (DataUnavailable, OutsideDomain) as e:
        rec.update(served_rainfall_mm=None, served_error=f"{type(e).__name__}: {e}",
                   match=None)
    traces.append(rec)
OUT["l1_l2_traces"] = traces
eligible = [t for t in traces if t["match"] is not None]
ok = sum(t["match"] for t in eligible)
print(f"{ok}/{len(eligible)} served values == independently recomputed (round(raw,2)); "
      f"{len(traces) - len(eligible)} panchayats errored (masked cell)")
for t in traces:
    print(f"  {t['panchayat_name'][:22]:22s} {t['district'][:11]:11s} {t['mapping_method']:16s} "
          f"raw={t['npz_raw']:8.4f} served={t['served_rainfall_mm']} "
          f"match={t['match']} ctrl_differ={t['controls_differ']} "
          f"{(t['served_error'] or '')[:52]}")
print(f"  rounding check: raw!=rounded for "
      f"{sum(1 for t in traces if abs(t['npz_raw'] - t['npz_rounded']) > 0) }/{len(traces)}")

# ---------------------------------------------------------------- units
print("\n" + "=" * 78)
print("2. UNITS / SCALING")
print("=" * 78)
di_u = NPZ_DATES.index(DATE)
day = RAIN[di_u]                                           # ONE day, all cells
imd_mm = np.asarray(X[di_u, 0]) * RS                       # normalized -> mm
y_mm = Y[di_u, 0]
m = M[di_u].astype(bool)
units = {
    "day": DATE,
    "npz_day_max_mm": float(np.nanmax(day)),
    "chirps_ref_day_max_mm": float(np.nanmax(y_mm[m])),
    "imd_bilinear_day_max_mm": float(imd_mm.max()),
    "npz_day_mean_over_valid_mm": float(np.nanmean(day)),
    "chirps_day_mean_over_valid_mm": float(y_mm[m].mean()),
    "imd_day_mean_mm": float(imd_mm.mean()),
    "npz_day_mean_bias_vs_chirps_mm": float(np.nanmean(day) - y_mm[m].mean()),
    "npz_min_mm": float(np.nanmin(day)),
    "npz_has_negative": bool(np.nanmin(day) < 0),
    "npz_has_inf": bool(np.isinf(day).any()),
    "rain_scale_mm": RS,
    "npz_over_imd_ratio_at_max": float(np.nanmax(day) / imd_mm.max()),
    # unit sanity across the whole split: the served field must stay the same
    # order of magnitude as the reference (mm/day), never 100x (double scaling)
    "split_npz_max_mm": float(np.nanmax(RAIN)),
    "split_chirps_max_mm": float(np.nanmax(np.where(M.astype(bool), Y[:, 0], np.nan))),
    "split_npz_mean_mm": float(np.nanmean(RAIN)),
    "split_chirps_mean_mm": float(np.nanmean(np.where(M.astype(bool), Y[:, 0], np.nan))),
}
# a double-scaling bug would push values ~100x
units["looks_double_scaled"] = units["npz_over_imd_ratio_at_max"] > 5
OUT["units"] = units
print(json.dumps(units, indent=1))

# --------------------------------------------- polygon vs nearest-cell
print("\n" + "=" * 78)
print("3. POLYGON AGGREGATION VS NEAREST CELL")
print("=" * 78)
poly_cmp = []
try:
    import geopandas as gpd
    from shapely.geometry import box

    lgd = gpd.read_parquet(ROOT / "data" / "raw" / "administrative" / "panchayat"
                           / "LGD_Panchayats.parquet")
    lgd["gpcode"] = pd.to_numeric(lgd["gpcode"], errors="coerce")
    lgd = lgd[lgd["gpcode"].notna()].copy()
    lgd["gpcode"] = lgd["gpcode"].astype("int64")
    lgd = lgd[lgd.gpcode.isin(sample)]
    di = NPZ_DATES.index(DATE)
    step = 0.05
    seas = {pid: [] for pid in lgd.gpcode}
    for pid, geom in zip(lgd.gpcode, lgd.geometry):
        if geom is None or geom.is_empty:
            continue
        lat0, lon0 = idx_by_pid[pid]
        i0, j0 = cell_of(lat0, lon0)
        nearest = float(RAIN[di, i0, j0])
        minx, miny, maxx, maxy = geom.bounds
        ii = np.where((LAT >= miny - step) & (LAT <= maxy + step))[0]
        jj = np.where((LON >= minx - step) & (LON <= maxx + step))[0]
        num = den = 0.0
        vals, ws = [], []
        for a in ii:
            for b in jj:
                clat, clon = LAT[a], LON[b]
                cb = box(clon - step / 2, clat - step / 2, clon + step / 2, clat + step / 2)
                if not cb.intersects(geom):
                    continue
                w = cb.intersection(geom).area
                if w <= 0:
                    continue
                v = float(RAIN[di, a, b])
                if not np.isfinite(v):
                    continue
                num += w * v
                den += w
                vals.append(v)
                ws.append(w)
        area_w = num / den if den else float("nan")
        simple = float(np.mean(vals)) if vals else float("nan")
        try:
            served_v = store.grid_rainfall(*idx_by_pid[pid], DATE)["rainfall_mm"]
            served_err = None
        except (DataUnavailable, OutsideDomain) as e:
            served_v, served_err = None, f"{type(e).__name__}"
        srow = summ[summ.panchayat_id == pid].iloc[0]
        poly_cmp.append({
            "panchayat_id": int(pid), "n_cells_touched": len(vals),
            "n_cells_summary": int(srow.n_cells),
            "mapping_method": srow.mapping_method,
            "nearest_cell_at_index_point": nearest,
            "grid_cell_mean_unweighted": simple,
            "area_weighted_mean": area_w,
            "served": served_v, "served_error": served_err,
            "summary_season_mean_mm": float(srow.rainfall_mean_mm),
            "nearest_minus_areaweighted": (None if not np.isfinite(area_w)
                                           else nearest - area_w),
        })
    OUT["polygon_vs_nearest"] = poly_cmp
    for c in poly_cmp:
        d = c["nearest_minus_areaweighted"]
        print(f"  pid={c['panchayat_id']} cells={c['n_cells_touched']:3d} "
              f"({c['mapping_method']:16s}) nearest={c['nearest_cell_at_index_point']:7.2f} "
              f"cellmean={c['grid_cell_mean_unweighted']:7.2f} "
              f"areaweighted={c['area_weighted_mean']:7.2f} served={c['served']} "
              f"delta_nearest_minus_area={None if d is None else round(d, 3)} "
              f"{c['served_error'] or ''}")
except Exception as e:  # pragma: no cover
    OUT["polygon_vs_nearest"] = {"error": repr(e)}
    print("  polygon test failed:", repr(e))

# --------------------------------------------- coordinate collapse / identity
print("\n" + "=" * 78)
print("4. COORDINATE COLLAPSE + PANCHAYAT IDENTITY")
print("=" * 78)
crd = idx.groupby(["lat", "lon"]).size()
names = summ.groupby(summ.panchayat_name.str.upper()).size()
identity = {
    "summary_rows": int(len(summ)),
    "index_rows": int(len(idx)),
    "distinct_coords": int(len(crd)),
    "max_panchayats_per_coord": int(crd.max()) if len(crd) else 0,
    "median_panchayats_per_coord": float(crd.median()) if len(crd) else 0,
    "n_coords_shared_by_more_than_one": int((crd > 1).sum()),
    "distinct_panchayat_names": int(len(names)),
    "duplicated_names": int((names > 1).sum()),
    "max_rows_sharing_a_name": int(names.max()) if len(names) else 0,
    "mapping_method_counts": summ.mapping_method.value_counts().to_dict(),
    "unmapped_rows": int((summ.mapping_method == "unmapped").sum()),
}
OUT["identity"] = identity
print(json.dumps(identity, indent=1))

# ------------------------------- summary staleness (area-weighted season mean)
print("\n" + "=" * 78)
print("5. LAYER-2 SUMMARY: IS IT STILL THE FIELD WE SERVE?")
print("=" * 78)
staleness = []
try:
    import geopandas as gpd
    from shapely.geometry import box as _box
    lgd2 = gpd.read_parquet(ROOT / "data" / "raw" / "administrative" / "panchayat"
                            / "LGD_Panchayats.parquet")
    lgd2["gpcode"] = pd.to_numeric(lgd2["gpcode"], errors="coerce")
    lgd2 = lgd2[lgd2.gpcode.notna()]
    check = [276458, 202270, 203962]
    gsel = lgd2[lgd2.gpcode.isin(check)]
    step = 0.05
    for pid in check:
        rows = summ[summ.panchayat_id == pid]
        if not len(rows):
            continue
        srow = rows.iloc[0]
        geoms = list(gsel[gsel.gpcode == pid].geometry)
        if not geoms:
            continue
        try:
            from shapely.ops import unary_union
            geom = unary_union(geoms) if len(geoms) > 1 else geoms[0]
        except Exception:
            geom = geoms[0]
        minx, miny, maxx, maxy = geom.bounds
        ii = np.where((LAT >= miny - step) & (LAT <= maxy + step))[0]
        jj = np.where((LON >= minx - step) & (LON <= maxx + step))[0]
        cells = []
        for a in ii:
            for b in jj:
                cb = _box(LON[b] - step / 2, LAT[a] - step / 2,
                          LON[b] + step / 2, LAT[a] + step / 2)
                if not cb.intersects(geom):
                    continue
                w = cb.intersection(geom).area
                if w > 0:
                    cells.append((a, b, w))
        daily = []
        for t in range(RAIN.shape[0]):
            num = den = 0.0
            for a, b, w in cells:
                v = float(RAIN[t, a, b])
                if np.isfinite(v):
                    num += w * v
                    den += w
            daily.append(num / den if den else np.nan)
        daily = np.array(daily)
        staleness.append({
            "panchayat_id": int(pid),
            "summary_n_cells": int(srow.n_cells),
            "my_cells": len(cells),
            "summary_rainfall_mean_mm": float(srow.rainfall_mean_mm),
            "recomputed_area_weighted_season_mean_mm": float(np.nanmean(daily)),
            "abs_diff_mm": float(abs(np.nanmean(daily) - srow.rainfall_mean_mm)),
            "summary_max_mm": float(srow.rainfall_max_mm),
            "recomputed_max_mm": float(np.nanmax(daily)),
        })
except Exception as e:
    staleness = [{"error": repr(e)}]
OUT["layer2_summary_staleness"] = staleness
for s in staleness:
    print("  ", json.dumps(s))

# aggregate, decisive check: is the summary distribution consistent with the
# Layer-1 field we serve right now?
_day = RAIN[NPZ_DATES.index(DATE)]
_own = []
for pid in summ.panchayat_id.head(4000):
    if pid in idx_by_pid:
        la, lo = idx_by_pid[pid]
        i2, j2 = cell_of(la, lo)
        v = float(_day[i2, j2])
        if np.isfinite(v):
            _own.append(v)
_obs_own = []
for pid in summ.panchayat_id.head(4000):
    if pid in idx_by_pid:
        la, lo = idx_by_pid[pid]
        i2, j2 = cell_of(la, lo)
        v = float(Y[NPZ_DATES.index(DATE), 0, i2, j2])
        if np.isfinite(v):
            _obs_own.append(v)
agg = {
    "npz_day_global_mean_mm": float(np.nanmean(_day)),
    "chirps_day_global_mean_mm": float(np.nanmean(np.where(M[NPZ_DATES.index(DATE)].astype(bool), Y[NPZ_DATES.index(DATE), 0], np.nan))),
    "sample_panchayat_own_cell_model_mean_mm": float(np.mean(_own)),
    "sample_panchayat_own_cell_chirps_mean_mm": float(np.mean(_obs_own)),
    "summary_rainfall_mean_mm_median": float(summ.rainfall_mean_mm.median()),
    "summary_rainfall_mean_mm_mean": float(summ.rainfall_mean_mm.mean()),
    "summary_rainfall_mean_mm_p95": float(summ.rainfall_mean_mm.quantile(0.95)),
    "summary_rainfall_max_mm_max": float(summ.rainfall_max_mm.max()),
    "npz_split_mean_mm": float(np.nanmean(RAIN)),
    "npz_split_max_mm": float(np.nanmax(RAIN)),
    "chirps_split_mean_mm": float(np.nanmean(np.where(M.astype(bool), Y[:, 0], np.nan))),
    "chirps_split_max_mm": float(np.nanmax(np.where(M.astype(bool), Y[:, 0], np.nan))),
}
agg["summary_to_npz_split_mean_ratio"] = (
    agg["summary_rainfall_mean_mm_mean"] / agg["npz_split_mean_mm"])
OUT["layer2_summary_vs_field"] = agg
print("\n  aggregate consistency of Layer-2 summary vs the served Layer-1 field:")
print(json.dumps(agg, indent=1))

# ------------------------------------------------------------ temporal index
print("\n" + "=" * 78)
print("6. TEMPORAL INDEX TRACES")
print("=" * 78)
temporal = {"date_index_ok": 0, "date_index_bad": 0, "traces": []}
pids = [int(x) for x in summ[summ.mapping_method == "direct_grid"].panchayat_id.head(3)]
dates_to_test = ([NPZ_DATES[0], NPZ_DATES[1], NPZ_DATES[61], NPZ_DATES[-2], NPZ_DATES[-1]]
                 + [NPZ_DATES[k] for k in (30, 31, 60, 61, 91, 92, 100, 110, 115)]
                 + ["2022-06-30", "2022-07-01", "2022-08-31", "2022-09-01"])
for pid in pids:
    if pid not in idx_by_pid:
        continue
    lat0, lon0 = idx_by_pid[pid]
    i, j = cell_of(lat0, lon0)
    for d in dates_to_test:
        t = NPZ_DATES.index(d)
        raw = float(RAIN[t, i, j])
        if not np.isfinite(raw):
            continue
        served = store.grid_rainfall(lat0, lon0, d)
        good = abs(round(raw, 2) - served["rainfall_mm"]) < 1e-9
        temporal["date_index_ok" if good else "date_index_bad"] += 1
        temporal["traces"].append({"panchayat_id": pid, "date": d, "npz_index": t,
                                   "raw": raw, "served": served["rainfall_mm"], "ok": good})
for bad_date in ("2021-07-10", "2022-10-01", "2022-05-31", "2022-13-01", "not-a-date", ""):
    try:
        r = store.grid_rainfall(*idx_by_pid[pids[0]], bad_date)
        temporal.setdefault("out_of_range", []).append({"date": bad_date, "result": r})
    except (DataUnavailable, OutsideDomain, ValueError) as e:
        temporal.setdefault("out_of_range", []).append({"date": bad_date,
                                                         "error": type(e).__name__})
print(f"  {temporal['date_index_ok']}/{temporal['date_index_ok'] + temporal['date_index_bad']} "
      f"date->index->value traces exact")
print(f"  out-of-range/invalid dates: "
      f"{json.dumps(temporal.get('out_of_range'))[:400]}")
OUT["temporal"] = temporal

# ------------------------------------------------------------ spatial sanity
print("\n" + "=" * 78)
print("7. SPATIAL SANITY (neighbouring panchayats)")
print("=" * 78)
pts = idx.copy()
pts["i"] = [cell_of(a, b)[0] for a, b in zip(pts.lat, pts.lon)]
pts["j"] = [cell_of(a, b)[1] for a, b in zip(pts.lat, pts.lon)]
pts["cell"] = list(zip(pts.i, pts.j))
dup = pts.groupby("cell").size()
di = NPZ_DATES.index(DATE)
cellvals = {c: float(RAIN[di, int(c[0]), int(c[1])]) for c in dup.index}
nan_cells = sum(1 for v in cellvals.values() if not np.isfinite(v))
spatial = {
    "panchayats_on_identical_lattice_cell": int((dup > 1).sum()),
    "max_panchayats_on_one_cell": int(dup.max()),
    "distinct_lattice_cells": int(len(dup)),
    "cells_with_nan_field": nan_cells,
    "mean_abs_diff_between_neighbouring_cells_mm": None,
    "n_neighbour_pairs": 0,
}
diffs = []
keys = set(dup.index)
for (a, b) in keys:
    for (da, db) in ((0, 1), (1, 0)):
        nb = (a + da, b + db)
        if nb in keys:
            v1, v2 = cellvals[(a, b)], cellvals[nb]
            if np.isfinite(v1) and np.isfinite(v2):
                diffs.append(abs(v1 - v2))
if diffs:
    spatial["n_neighbour_pairs"] = len(diffs)
    spatial["mean_abs_diff_between_neighbouring_cells_mm"] = float(np.mean(diffs))
    spatial["p95_abs_diff_mm"] = float(np.percentile(diffs, 95))
    spatial["max_abs_diff_mm"] = float(np.max(diffs))
OUT["spatial"] = spatial
print(json.dumps(spatial, indent=1))

# ----------------------------------------------------------- temporal sanity
print("\n" + "=" * 78)
print("8. TEMPORAL SANITY (monsoon structure)")
print("=" * 78)
mm = np.array([d[5:7] for d in NPZ_DATES])
dom = np.array([np.nanmean(RAIN[t]) for t in range(RAIN.shape[0])])
obs = np.array([np.nanmean(np.where(M[t].astype(bool), Y[t, 0], np.nan))
                for t in range(RAIN.shape[0])])
imdd = np.array([np.mean(np.asarray(X[t, 0]) * RS) for t in range(RAIN.shape[0])])
months = {}
for mth in ("06", "07", "08", "09"):
    s = mm == mth
    months[mth] = {"n_days": int(s.sum()), "model_mean_mm": float(dom[s].mean()),
                   "chirps_mean_mm": float(obs[s].mean()), "imd_bilinear_mean_mm": float(imdd[s].mean())}
ts = {
    "monthly_means": months,
    "corr_model_vs_chirps_daily_domain_mean": float(np.corrcoef(dom, obs)[0, 1]),
    "corr_model_vs_imd_daily_domain_mean": float(np.corrcoef(dom, imdd)[0, 1]),
    "corr_chirps_vs_imd_daily_domain_mean": float(np.corrcoef(obs, imdd)[0, 1]),
    "n_identical_consecutive_domain_means": int(np.sum(np.diff(dom) == 0)),
    "n_identical_consecutive_full_days": int(sum(
        bool(np.array_equal(RAIN[t], RAIN[t + 1])) for t in range(RAIN.shape[0] - 1))),
    "model_daily_domain_mean_range": [float(dom.min()), float(dom.max())],
    "chirps_daily_domain_mean_range": [float(obs.min()), float(obs.max())],
    "split": meta.get("split_description"),
}
OUT["temporal_sanity"] = ts
print(json.dumps(ts, indent=1))

# --------------------------------------------------------- model output sanity
print("\n" + "=" * 78)
print("9. MODEL OUTPUT SANITY (extremes)")
print("=" * 78)
for thr in (10, 25, 50, 100):
    pass
mmask = M.astype(bool)
rows = []
for thr in (10, 25, 50, 100):
    o = (Y[:, 0] >= thr) & mmask
    p = (RAIN >= thr) & mmask
    tp = int((o & p).sum()); fp = int((~o & p).sum()); fn = int((o & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    rows.append({"thr_mm": thr, "n_obs_events": int(o.sum()), "n_pred_events": int(p.sum()),
                 "precision": prec, "recall": rec,
                 "f1": (2 * prec * rec / (prec + rec)) if (prec == prec and rec == rec and prec + rec) else float("nan"),
                 "n_missed": fn, "n_false_alarms": fp})
obs_col = Y[:, 0][mmask]
mdl_col = RAIN[mmask]
sel = obs_col >= 50
model_sanity = {
    "event_table": rows,
    "obs_ge50_mean_pred_mm": float(mdl_col[sel].mean()),
    "obs_ge50_median_pred_mm": float(np.median(mdl_col[sel])),
    "obs_ge50_mean_obs_mm": float(obs_col[sel].mean()),
    "obs_ge50_frac_predicted_ge50": float(np.mean(mdl_col[sel] >= 50)),
    "frac_of_reference_max_achieved": float(np.nanmax(RAIN) / np.nanmax(Y[:, 0][mmask])),
    "p99_obs": float(np.percentile(obs_col, 99)),
    "p99_model": float(np.percentile(mdl_col, 99)),
}
OUT["model_sanity"] = model_sanity
print(json.dumps(model_sanity, indent=1, default=str))

# -------------------------------------------------- administrative hierarchy
print("\n" + "=" * 78)
print("10. ADMINISTRATIVE HIERARCHY JOIN")
print("=" * 78)
admin_res = {}
try:
    lgd3 = gpd.read_parquet(ROOT / "data" / "raw" / "administrative" / "panchayat"
                            / "LGD_Panchayats.parquet")
    lgd3["gpcode"] = pd.to_numeric(lgd3["gpcode"], errors="coerce")
    lgd3 = lgd3[lgd3.gpcode.notna()]
    n_dup_keys = int(lgd3.gpcode.duplicated().sum())
    mrg = summ.merge(lgd3[["gpcode", "gpname", "stname", "dtname", "blkname"]],
                     left_on="panchayat_id", right_on="gpcode", how="left")
    admin_res.update({
        "lgd_rows": int(len(lgd3)),
        "lgd_duplicated_gpcodes": n_dup_keys,
        "joined_rows": int(len(mrg)),
        "summary_rows": int(len(summ)),
        "join_row_inflation": int(len(mrg) - len(summ)),
        "name_mismatch": int((mrg.gpname.str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
                              != mrg.panchayat_name.str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)).sum()),
        "district_mismatch": int((mrg.dtname.str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
                                  != mrg.district.str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)).sum()),
        "state_mismatch": int((mrg.stname.str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
                               != mrg.state.str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)).sum()),
    })
except Exception as e:
    admin_res = {"error": repr(e)}
OUT["admin_hierarchy"] = admin_res
print(json.dumps(admin_res, indent=1))

json.dump(OUT, open(HERE / "qa" / "qa_logic_core.json", "w"), indent=1, default=str)
print("\n-> qa_logic_core.json")
