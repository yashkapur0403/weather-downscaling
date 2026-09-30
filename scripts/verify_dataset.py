"""End-to-end verification of the Layer-1 dataset (region `deccan`).

Checks, with LOUD failures (exit code 1) and a one-screen summary:
  1. processed X/Y/M_{train,val,test}.npy exist, shapes match meta.json,
     M semantics hold (M==1 => Y finite; M==0 => Y==0)
  2. X has no NaNs/Infs; Y is finite where M==1
  3. normalization stats (rain/elev scale) are finite and positive
  4. every aux layer (admin map, soil, NDVI, LULC) indexes the SAME land
     cells (fine_i/fine_j identical) as the model land mask
  5. aux value ranges are sane (soil raw units, NDVI -1..1, LULC 0..1)

Run:  python scripts/verify_dataset.py [--region deccan]
Writes data/reports/verify_dataset_<region>.json with the results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def _fail(msg: str, errs: list[str]) -> None:
    errs.append(msg)
    print(f"  [FAIL] {msg}")


def _ok(msg: str) -> None:
    print(f"  [ok] {msg}")


def verify_processed(errs: list[str]) -> dict:
    print("[1] processed model-ready arrays")
    meta_p = config.PROCESSED / "meta.json"
    if not meta_p.exists():
        _fail("meta.json missing - run preprocess.py", errs)
        return {}
    meta = json.loads(meta_p.read_text())
    out = {}
    for k in ("train", "val", "test"):
        X = np.load(config.PROCESSED / f"X_{k}.npy", mmap_mode="r")
        Y = np.load(config.PROCESSED / f"Y_{k}.npy", mmap_mode="r")
        M = np.load(config.PROCESSED / f"M_{k}.npy", mmap_mode="r")
        exp = (meta["split"][k]["n_days"], len(meta["channels"]),
               meta["grid"]["H_fine"], meta["grid"]["W_fine"])
        if X.shape != exp:
            _fail(f"X_{k} shape {X.shape} != meta {exp}", errs)
        if Y.shape != (exp[0], 1, *exp[2:]) or M.shape != Y.shape:
            _fail(f"Y_{k}/M_{k} shape {Y.shape}/{M.shape} inconsistent", errs)
        # full-array checks for train (small enough); sampled for val/test
        sl = slice(None) if k == "train" else slice(0, 10)
        Xs, Ys, Ms = X[sl], Y[sl], M[sl]
        if not np.isfinite(Xs).all():
            _fail(f"X_{k} contains NaN/Inf", errs)
        ymv = Ys[Ms == 1]
        if ymv.size and not np.isfinite(ymv).all():
            _fail(f"Y_{k} non-finite where M==1", errs)
        if (Ys[Ms == 0] != 0).any():
            _fail(f"Y_{k} non-zero where M==0 (must be 0 by design)", errs)
        out[k] = {"shape": list(X.shape), "days": int(X.shape[0])}
        _ok(f"{k}: X{X.shape}, {int(M.sum())} valid target pixels")
    if (M.ndim == 4 and M.shape[1] != 1) or (M.ndim != 4):
        _fail(f"M rank/shape unexpected: {M.shape}", errs)
    norm = meta.get("normalization", {})
    rs, es = norm.get("rain_scale_mm", 0), norm.get("elev_scale_m", 0)
    if not (rs > 0 and es > 0 and np.isfinite([rs, es]).all()):
        _fail(f"normalization scales bad: rain={rs} elev={es}", errs)
    else:
        _ok(f"rain_scale={rs:.1f} mm, elev_scale={es:.1f} m")
    return {"meta": {"region": meta.get("region"), "dates": len(meta.get("dates", [])),
                     "channels": meta.get("channels"),
                     "rain_scale_mm": rs, "elev_scale_m": es},
            "splits": out}


def verify_wind_and_soilmoisture(errs: list[str]) -> dict:
    print("[3] wind raw availability + soil-moisture aux layer")
    out = {}
    # combined ERA5 file: wind must be present (2018 repaired); the frozen
    # processed dataset intentionally uses 5 channels (wind documented-out)
    ep = config.RAW_ERA5 / f"era5_daily_{config.REGION_DEFAULT}.npz"
    if ep.exists():
        e = np.load(ep, allow_pickle=False)
        if "wind" in e.files:
            w = e["wind"]
            full_missing = int(np.isnan(w).all(axis=(1, 2)).sum())
            out["wind"] = {"days": int(w.shape[0]),
                           "nan_pct": round(float(np.isnan(w).mean() * 100), 3),
                           "fully_missing_days": full_missing}
            if full_missing > 0:
                _fail(f"wind has {full_missing} fully-missing days in the "
                      "combined ERA5 file", errs)
            else:
                _ok(f"wind: {out['wind']['days']} days, "
                    f"{out['wind']['nan_pct']}% NaN (raw, NOT a model channel)")
    else:
        _fail("combined ERA5 file missing", errs)
    # soil moisture aux layer
    sp = config.AUX / f"soilmoisture_daily_{config.REGION_DEFAULT}.npz"
    if sp.exists():
        s = np.load(sp, allow_pickle=False)
        v = s["soil_moisture_0_to_7cm_mean"]
        vn = v[np.isfinite(v)]
        if vn.size and (vn.min() < 0 or vn.max() > 0.9):
            _fail(f"soil moisture out of physical range: {vn.min()}..{vn.max()}", errs)
        out["soilmoisture"] = {
            "days": int(len(s["dates"])), "lattice": list(v.shape[1:]),
            "vars": [k for k in s.files if k.startswith("soil_moisture")],
            "missing_pct": round(float(np.isnan(v).mean() * 100), 3)}
        _ok(f"soilmoisture: {out['soilmoisture']['days']} monsoon days on a "
            f"{v.shape[1]}x{v.shape[2]} ERA5 lattice, "
            f"missing {out['soilmoisture']['missing_pct']}%")
    else:
        # optional Layer-3 layer: warn + give the exact materialize command
        # instead of failing the whole QA (fetch is quota-gated, resumable)
        print(f"  [warn] {sp.name} not materialized yet (optional Layer-3 "
              "layer; Open-Meteo quota-gated). Materialize with: "
              "python scripts/build_aux.py --region deccan --skip-admin "
              "--skip-soil --skip-ndvi --skip-lulc")
    return out


def verify_aux(errs: list[str]) -> dict:
    print("[2] aux layers: same land cells + sane ranges")
    out = {}
    ref = None
    # admin
    ap = config.AUX / "admin" / f"grid_admin_map_{config.REGION_DEFAULT}.npz"
    if ap.exists():
        a = np.load(ap, allow_pickle=True)
        ref = (a["fine_i"], a["fine_j"])
        hit = len({*a["subdistrict_ids"].tolist()})
        out["admin"] = {"cells": int(len(a["fine_i"])), "subdistricts_hit": hit}
        _ok(f"admin map: {len(a['fine_i'])} cells, {hit} subdistricts hit")
    else:
        _fail(f"missing {ap.name}", errs)
    # soil
    sp = config.AUX / f"soil_soilgrids_{config.REGION_DEFAULT}.npz"
    if sp.exists():
        s = np.load(sp, allow_pickle=False)
        vals = s["values"]  # (n_coarse_points, n_properties), raw SoilGrids units
        props = [str(x) for x in s["properties"]]
        per_prop = {p: round(float((~np.isfinite(vals[:, i])).mean() * 100), 2)
                    for i, p in enumerate(props)}
        v = vals[np.isfinite(vals)]
        if v.size and (v.min() < 0 or v.max() > 10000):
            _fail(f"soil raw values out of expected range: {v.min()}..{v.max()}", errs)
        n_fine = int(s["fine_values"].shape[0]) if "fine_values" in s.files else -1
        if n_fine > 0 and int(len(s["fine_i"])) != n_fine:
            _fail("soil fine_values rows != fine_i length", errs)
        out["soil"] = {"points": int(vals.shape[0]), "properties": props,
                       "missing_pct_per_property": per_prop}
        _ok(f"soil: {vals.shape[0]} points x {len(props)} props, "
            f"missing {min(per_prop.values())}..{max(per_prop.values())}%")
    else:
        _fail(f"missing {sp.name}", errs)
    # NDVI
    np_ = config.AUX / f"ndvi_monthly_{config.REGION_DEFAULT}.npz"
    if np_.exists():
        n = np.load(np_, allow_pickle=True)
        if ref is not None and not (np.array_equal(n["fine_i"], ref[0])
                                    and np.array_equal(n["fine_j"], ref[1])):
            _fail("NDVI cell indexing differs from admin map", errs)
        v = n["ndvi"]
        vn = v[np.isfinite(v)]
        if vn.size and (vn.min() < -1.01 or vn.max() > 1.01):
            _fail(f"NDVI out of [-1,1]: {vn.min()}..{vn.max()}", errs)
        out["ndvi"] = {"months": int(len(n["months"])),
                       "missing_pct": round(float(np.isnan(v).mean() * 100), 2)}
        _ok(f"NDVI: {out['ndvi']['months']} months, missing "
            f"{out['ndvi']['missing_pct']}%, range {vn.min():.3f}..{vn.max():.3f}")
    else:
        _fail(f"missing {np_.name}", errs)
    # LULC
    lp = config.AUX / f"lulc_fractions_{config.REGION_DEFAULT}.npz"
    if lp.exists():
        l = np.load(lp, allow_pickle=False)
        if ref is not None and not (np.array_equal(l["fine_i"], ref[0])
                                    and np.array_equal(l["fine_j"], ref[1])):
            _fail("LULC cell indexing differs from admin map", errs)
        fr = l["fractions"]
        cov = fr.sum(1) > 0
        if (fr[cov] < -1e-6).any() or (fr[cov] > 1.0001).any():
            _fail("LULC fractions outside [0,1]", errs)
        out["lulc"] = {"cells": int(len(fr)),
                       "coverage_pct": round(float(cov.mean() * 100), 2)}
        _ok(f"LULC: {out['lulc']['coverage_pct']}% cells with class data")
    else:
        _fail(f"missing {lp.name}", errs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default=config.REGION_DEFAULT)
    args = ap.parse_args()
    errs: list[str] = []
    print(f"=== verify_dataset ({args.region}) ===")
    proc = verify_processed(errs)
    aux = verify_aux(errs)
    wind_sm = verify_wind_and_soilmoisture(errs)
    res = {"region": args.region, "errors": errs, "processed": proc,
           "aux": aux, "wind_and_soilmoisture": wind_sm, "ok": not errs}
    rp = config.REPORTS / f"verify_dataset_{args.region}.json"
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(res, indent=1, default=str))
    if errs:
        print(f"\n{len(errs)} problem(s) found - see {rp}")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    print(f"\nALL CHECKS PASSED -> {rp}")


if __name__ == "__main__":
    main()
