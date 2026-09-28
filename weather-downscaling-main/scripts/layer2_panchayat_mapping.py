"""
Layer 2: Panchayat-Level Weather Mapping
=========================================
Consumes the Layer 1 contract (prediction/*.npz with dates, latitude,
longitude, rainfall_mm) and maps each 0.05-deg field onto LGD Gram
Panchayat polygons.

Does not modify Layer 1 models or training data.

Outputs (default: outputs/layer2/):
  panchayat_weather.csv      one row per Panchayat per date
  panchayat_summary.csv      one row per Panchayat (mean/max/wet days)
  panchayat_weather.geojson  polygons + map-date rainfall
  panchayat_weather_map.png  choropleth for the map date
  layer2_qc.json             coverage / fallback / input provenance

Usage:
  python scripts/layer2_panchayat_mapping.py
  python scripts/layer2_panchayat_mapping.py --input prediction/infer_2022-07-10.npz
  python scripts/layer2_panchayat_mapping.py --block MANGALURU
  python scripts/layer2_panchayat_mapping.py --state MAHARASHTRA --out outputs/layer2_mh
  python scripts/layer2_panchayat_mapping.py --method area --dates 2022-07-10
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import config  # noqa: E402
from grids import fine_grid  # noqa: E402

DEFAULT_PARQUET = config.RAW_PANCHAYAT / "LGD_Panchayats.parquet"
DEFAULT_OUT = config.OUT_LAYER2
METRIC_CRS = "EPSG:32643"  # UTM 43N covers the default Western Ghats ROI
CSV_ATTRS = ["state", "district", "block_id", "block_name",
             "panchayat_id", "panchayat_name"]


@dataclass
class Layer1Field:
    """Layer-1 rainfall on the fine grid (Layer-2 contract)."""
    dates: list[str]
    latitude: np.ndarray
    longitude: np.ndarray
    rainfall_mm: np.ndarray  # (n_days, H, W)
    source: Path
    reconstructed_grid: bool


# ---------------------------------------------------------------------------
# Input discovery / load
# ---------------------------------------------------------------------------
def discover_layer1_path(explicit: Path | None) -> Path:
    """Prefer the Layer-1 .npz contract, then a 2-D infer_*.npy map."""
    if explicit is not None:
        if not explicit.exists():
            sys.exit(f"[ERROR] Layer 1 file not found: {explicit}")
        return explicit

    npz_hits = sorted(config.PRED_DIR.glob("infer_*.npz")) if config.PRED_DIR.exists() else []
    if npz_hits:
        return npz_hits[-1]
    npy_hits = sorted(config.OUT_MAPS.glob("infer_*.npy")) if config.OUT_MAPS.exists() else []
    if npy_hits:
        return npy_hits[-1]
    sys.exit(
        "[ERROR] No Layer 1 input found. Pass --input to a "
        "prediction/infer_*.npz (preferred) or outputs/maps/infer_*.npy. "
        "Run scripts/infer.py first."
    )


def _dates_from_name(path: Path) -> list[str]:
    stem = path.stem  # infer_2022-07-10 or infer_2022_07_10
    for prefix in ("infer_",):
        if stem.startswith(prefix):
            raw = stem[len(prefix):]
            if "_" in raw and "-" not in raw:
                parts = raw.split("_")
                if len(parts) >= 3:
                    return ["-".join(parts[:3])]
            return [raw]
    return ["unknown"]


def reconstruct_grid(shape_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Fine-grid centers from the same ROI / FINE_SUB Layer 1 used."""
    roi = config.ROI_DEFAULT
    sub = config.FINE_SUB
    imd_lat = np.arange(roi["lat_min"], roi["lat_max"] + 1e-9, config.IMD_STEP)
    imd_lon = np.arange(roi["lon_min"], roi["lon_max"] + 1e-9, config.IMD_STEP)
    fine_lat, fine_lon = fine_grid(imd_lat, imd_lon, sub)
    expected = (len(imd_lat) * sub, len(imd_lon) * sub)
    if shape_hw != expected:
        sys.exit(
            f"[ERROR] Array shape {shape_hw} does not match expected "
            f"{expected} for ROI {roi}."
        )
    return fine_lat, fine_lon


def load_layer1(path: Path) -> Layer1Field:
    """Load .npz contract or a single-day .npy rainfall map."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        z = np.load(path, allow_pickle=True)
        need = ("dates", "latitude", "longitude", "rainfall_mm")
        missing = [k for k in need if k not in z.files]
        if missing:
            sys.exit(f"[ERROR] {path.name} missing Layer-1 fields: {missing}")
        rain = np.asarray(z["rainfall_mm"], dtype="float64")
        if rain.ndim == 2:
            rain = rain[None, ...]
        if rain.ndim != 3:
            sys.exit(f"[ERROR] rainfall_mm must be (n,H,W) or (H,W), got {rain.shape}")
        dates = [str(d) for d in np.asarray(z["dates"]).tolist()]
        if len(dates) == 1 and rain.shape[0] != 1 and dates[0].count("-") == 0:
            dates = [str(d) for d in dates]
        if len(dates) != rain.shape[0]:
            sys.exit(
                f"[ERROR] dates length {len(dates)} != rainfall days {rain.shape[0]}"
            )
        lat = np.asarray(z["latitude"], dtype="float64").reshape(-1)
        lon = np.asarray(z["longitude"], dtype="float64").reshape(-1)
        if rain.shape[1:] != (len(lat), len(lon)):
            sys.exit(
                f"[ERROR] rainfall {rain.shape} vs lat {len(lat)} lon {len(lon)}"
            )
        field = Layer1Field(dates, lat, lon, rain, path, reconstructed_grid=False)
    elif path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim == 3:
            rain = arr.astype("float64")
            dates = _dates_from_name(path)
            if len(dates) != rain.shape[0]:
                dates = [f"{_dates_from_name(path)[0]}_t{i}" for i in range(rain.shape[0])]
        elif arr.ndim == 2:
            rain = arr.astype("float64")[None, ...]
            dates = _dates_from_name(path)
        else:
            sys.exit(f"[ERROR] Expected 2-D or 3-D array, got {arr.shape}")
        lat, lon = reconstruct_grid(rain.shape[1:])
        field = Layer1Field(dates, lat, lon, rain, path, reconstructed_grid=True)
    else:
        sys.exit(f"[ERROR] Unsupported Layer 1 file type: {path.suffix}")

    finite = np.isfinite(field.rainfall_mm)
    print(
        f"[layer1] {path.name}: days={len(field.dates)} "
        f"grid={field.rainfall_mm.shape[1]}x{field.rainfall_mm.shape[2]} "
        f"range={np.nanmin(field.rainfall_mm):.2f}-{np.nanmax(field.rainfall_mm):.2f} mm "
        f"finite={finite.mean():.1%} "
        f"{'(grid reconstructed from ROI)' if field.reconstructed_grid else '(lat/lon from file)'}"
    )
    return field


def select_dates(field: Layer1Field, wanted: list[str] | None) -> Layer1Field:
    if not wanted:
        return field
    idx = []
    have = {d: i for i, d in enumerate(field.dates)}
    for d in wanted:
        if d not in have:
            sys.exit(f"[ERROR] date {d} not in Layer 1 file ({field.dates[0]} .. {field.dates[-1]})")
        idx.append(have[d])
    rain = field.rainfall_mm[idx]
    dates = [field.dates[i] for i in idx]
    return Layer1Field(dates, field.latitude, field.longitude, rain,
                       field.source, field.reconstructed_grid)


# ---------------------------------------------------------------------------
# Grid points / cell polygons
# ---------------------------------------------------------------------------
def grid_to_points(field: Layer1Field) -> gpd.GeoDataFrame:
    """One point per fine-grid cell center, with a stable cell_id."""
    lat, lon = field.latitude, field.longitude
    lats, lons = np.meshgrid(lat, lon, indexing="ij")
    n_lat, n_lon = len(lat), len(lon)
    cell_i = np.repeat(np.arange(n_lat), n_lon)
    cell_j = np.tile(np.arange(n_lon), n_lat)
    cell_id = cell_i * n_lon + cell_j
    day0 = field.rainfall_mm[0].ravel()
    valid = np.isfinite(day0) | np.isfinite(field.rainfall_mm).any(axis=0).ravel()
    geom = [Point(x, y) for x, y in zip(lons.ravel()[valid], lats.ravel()[valid])]
    gdf = gpd.GeoDataFrame(
        {
            "cell_id": cell_id[valid],
            "cell_i": cell_i[valid],
            "cell_j": cell_j[valid],
        },
        geometry=geom,
        crs="EPSG:4326",
    )
    print(f"[points] {len(gdf):,} cell centers "
          f"({int((~valid).sum())} all-NaN cells dropped)")
    return gdf


def grid_to_cells(field: Layer1Field) -> gpd.GeoDataFrame:
    """Rectangle per 0.05-deg cell, for area-weighted zonal mean."""
    lat, lon = field.latitude, field.longitude
    dlat = float(np.median(np.diff(lat))) if len(lat) > 1 else 0.05
    dlon = float(np.median(np.diff(lon))) if len(lon) > 1 else 0.05
    n_lat, n_lon = len(lat), len(lon)
    recs = []
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            recs.append({
                "cell_id": i * n_lon + j,
                "cell_i": i,
                "cell_j": j,
                "geometry": box(lo - dlon / 2, la - dlat / 2,
                                lo + dlon / 2, la + dlat / 2),
            })
    return gpd.GeoDataFrame(recs, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# Panchayat polygons
# ---------------------------------------------------------------------------
def load_panchayats(parquet_path: Path, block_name: str | None,
                    state_name: str | None, roi_box) -> gpd.GeoDataFrame:
    if not parquet_path.exists():
        sys.exit(
            f"[ERROR] Panchayat file not found: {parquet_path}\n"
            "Place LGD_Panchayats.parquet under data/raw/administrative/panchayat/"
        )

    print(f"[panchayat] Reading {parquet_path.name} ...")
    df = gpd.read_parquet(parquet_path)
    if df.crs is None:
        df = df.set_crs("EPSG:4326")
    else:
        df = df.to_crs("EPSG:4326")

    if block_name:
        subset = df[df["blkname"].astype(str).str.strip().str.upper()
                    == block_name.upper()].copy()
        if subset.empty:
            sys.exit(f"[ERROR] Block '{block_name}' not found in LGD file.")
        region = f"block '{block_name}'"
    elif state_name:
        subset = df[df["stname"].astype(str).str.strip().str.upper()
                    == state_name.upper()].copy()
        if subset.empty:
            sys.exit(f"[ERROR] State '{state_name}' not found in LGD file.")
        region = f"state '{state_name}'"
    else:
        subset = df.copy()
        region = "all states (ROI clip)"

    n_before = len(subset)
    subset = subset[subset["gpcode"].notna()].copy()
    subset = subset[subset["gpcode"].astype(str).str.strip() != ""].copy()
    if n_before - len(subset):
        print(f"[panchayat] Dropped {n_before - len(subset)} rows with empty gpcode")

    invalid = ~subset.geometry.is_valid
    if invalid.any():
        print(f"[panchayat] Fixing {invalid.sum()} invalid geometries ...")
        subset.geometry = subset.geometry.buffer(0)

    agg_cols = {c: "first" for c in
                ("gpname", "stname", "dtname", "blklgdcode", "blkname")
                if c in subset.columns}
    dissolved = subset.dissolve(by="gpcode", aggfunc=agg_cols).reset_index()

    if roi_box is not None:
        before = len(dissolved)
        dissolved = dissolved[dissolved.intersects(roi_box)].copy()
        print(f"[panchayat] {region}: {before} GPs -> {len(dissolved)} intersecting Layer 1 ROI")
    else:
        print(f"[panchayat] {region}: {len(dissolved)} unique Panchayats")

    if dissolved.empty:
        sys.exit("[ERROR] No Panchayats intersect the Layer 1 coverage window.")
    return dissolved


def rename_admin(panchayats: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return panchayats.rename(columns={
        "gpcode": "panchayat_id",
        "gpname": "panchayat_name",
        "stname": "state",
        "dtname": "district",
        "blklgdcode": "block_id",
        "blkname": "block_name",
    })


# ---------------------------------------------------------------------------
# Spatial assignment (once) then multi-day aggregation
# ---------------------------------------------------------------------------
def _overlap_unmatched(unmatched: gpd.GeoDataFrame, points: gpd.GeoDataFrame,
                       lat_step: float, lon_step: float) -> pd.DataFrame:
    """Area-weight grid cells that intersect GPs with no cell center inside."""
    empty_cols = ["gpcode", "n_cells", "cell_ids", "weights",
                  "mapping_method", "fallback_distance_m"]
    if unmatched.empty or points.empty:
        return pd.DataFrame(columns=empty_cols)
    cells = points.copy()
    cells["geometry"] = [
        box(p.x - lon_step / 2, p.y - lat_step / 2,
            p.x + lon_step / 2, p.y + lat_step / 2)
        for p in points.geometry
    ]
    overlay = gpd.overlay(
        cells[["cell_id", "geometry"]],
        unmatched[["gpcode", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    if overlay.empty:
        return pd.DataFrame(columns=empty_cols)
    overlay = overlay.to_crs(METRIC_CRS)
    overlay["weight"] = overlay.geometry.area
    overlay = overlay[overlay["weight"] > 0]
    rows = []
    for gp, g in overlay.groupby("gpcode"):
        w = g.groupby("cell_id")["weight"].sum()
        w = w / w.sum()
        rows.append({
            "gpcode": gp,
            "n_cells": int(len(w)),
            "cell_ids": tuple(int(i) for i in w.index),
            "weights": tuple(float(x) for x in w.values),
            "mapping_method": "area_weighted",
            "fallback_distance_m": np.nan,
        })
    return pd.DataFrame(rows)


def assign_cells_center(points: gpd.GeoDataFrame, panchayats: gpd.GeoDataFrame,
                        max_fallback_km: float,
                        lat_step: float = 0.05, lon_step: float = 0.05) -> pd.DataFrame:
    """
    Pass 1: cell center within polygon.
    Pass 2: unmatched GP centroid -> nearest cell, if within max_fallback_km.
    """
    buf = config.LAYER2_BBOX_BUF_DEG
    minx, miny, maxx, maxy = panchayats.total_bounds
    nearby = points.cx[minx - buf:maxx + buf, miny - buf:maxy + buf].copy()
    print(f"[sjoin]  {len(nearby)} points vs {len(panchayats)} Panchayats")

    joined = gpd.sjoin(
        nearby,
        panchayats[["gpcode", "geometry"]],
        how="inner",
        predicate="within",
    )
    pass1 = joined.groupby("gpcode", as_index=False).agg(
        n_cells=("cell_id", "nunique"),
        cell_ids=("cell_id", lambda s: tuple(sorted(int(x) for x in s.unique()))),
    )
    pass1["mapping_method"] = "direct_grid"
    pass1["fallback_distance_m"] = np.nan
    pass1["weights"] = pass1["cell_ids"].map(
        lambda ids: tuple(1.0 / len(ids) for _ in ids) if ids else ()
    )
    matched = set(pass1["gpcode"].tolist())
    print(f"[sjoin]  Pass 1 (within): {len(pass1)} Panchayats")

    unmatched = panchayats[~panchayats["gpcode"].isin(matched)].copy()
    pass1b = _overlap_unmatched(unmatched, nearby, lat_step, lon_step)
    if len(pass1b):
        matched |= set(pass1b["gpcode"].tolist())
        print(f"[sjoin]  Pass 1b (cell overlap): {len(pass1b)} Panchayats")
    unmatched = panchayats[~panchayats["gpcode"].isin(matched)].copy()
    rows2 = []
    if not unmatched.empty and len(nearby) > 0:
        centroids = unmatched.to_crs(METRIC_CRS).copy()
        centroids.geometry = centroids.geometry.centroid
        nearby_m = nearby.to_crs(METRIC_CRS)
        nearest = gpd.sjoin_nearest(
            centroids[["gpcode", "geometry"]],
            nearby_m[["cell_id", "geometry"]],
            how="left",
            distance_col="fallback_distance_m",
        )
        nearest = nearest.drop_duplicates("gpcode", keep="first")
        cap_m = max_fallback_km * 1000.0
        for _, row in nearest.iterrows():
            dist = row.get("fallback_distance_m")
            cid = row.get("cell_id")
            if pd.isna(cid) or pd.isna(dist) or float(dist) > cap_m:
                continue
            rows2.append({
                "gpcode": row["gpcode"],
                "n_cells": 1,
                "cell_ids": (int(cid),),
                "weights": (1.0,),
                "mapping_method": "nearest_fallback",
                "fallback_distance_m": float(dist),
            })
        print(f"[sjoin]  Pass 2 (nearest <= {max_fallback_km:g} km): {len(rows2)} Panchayats")

    return pd.concat([pass1, pass1b, pd.DataFrame(rows2)], ignore_index=True)


def assign_cells_area(cells: gpd.GeoDataFrame, panchayats: gpd.GeoDataFrame,
                      max_fallback_km: float, points: gpd.GeoDataFrame) -> pd.DataFrame:
    """Intersection-area weights per (GP, cell); fallback as in center method."""
    buf = config.LAYER2_BBOX_BUF_DEG
    minx, miny, maxx, maxy = panchayats.total_bounds
    nearby = cells.cx[minx - buf:maxx + buf, miny - buf:maxy + buf].copy()
    print(f"[overlay] {len(nearby)} cells vs {len(panchayats)} Panchayats")
    overlay = gpd.overlay(
        nearby,
        panchayats[["gpcode", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    overlay = overlay.to_crs(METRIC_CRS)
    overlay["weight"] = overlay.geometry.area
    overlay = overlay[overlay["weight"] > 0].copy()

    grouped = overlay.groupby("gpcode")
    rows = []
    for gp, g in grouped:
        w = g.groupby("cell_id")["weight"].sum()
        w = w / w.sum()
        rows.append({
            "gpcode": gp,
            "n_cells": int(len(w)),
            "cell_ids": tuple(int(i) for i in w.index),
            "weights": tuple(float(x) for x in w.values),
            "mapping_method": "area_weighted",
            "fallback_distance_m": np.nan,
        })
    pass1 = pd.DataFrame(rows)
    matched = set(pass1["gpcode"].tolist()) if len(pass1) else set()
    print(f"[overlay] Pass 1 (area): {len(pass1)} Panchayats")

    # unmatched: reuse nearest-center fallback
    fb = assign_cells_center(points, panchayats, max_fallback_km)
    fb = fb[~fb["gpcode"].isin(matched)].copy()
    if "weights" not in fb.columns:
        fb["weights"] = fb["cell_ids"].map(
            lambda ids: tuple(1.0 / len(ids) for _ in ids) if ids else ()
        )
    return pd.concat([pass1, fb], ignore_index=True)


def aggregate_rainfall(field: Layer1Field, assignment: pd.DataFrame,
                       panchayats: gpd.GeoDataFrame,
                       wet_mm: float) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Mean rainfall per GP per date from precomputed cell lists."""
    rain = field.rainfall_mm  # (T, H, W)
    t, h, w = rain.shape
    flat = rain.reshape(t, h * w)

    recs = []
    for _, row in assignment.iterrows():
        ids = np.array(row["cell_ids"], dtype=int)
        if "weights" in assignment.columns and isinstance(row.get("weights"), tuple) and row["weights"]:
            wt = np.array(row["weights"], dtype="float64")
            vals = np.nansum(flat[:, ids] * wt[None, :], axis=1)
        else:
            vals = np.nanmean(flat[:, ids], axis=1)
        for di, date in enumerate(field.dates):
            recs.append({
                "date": date,
                "gpcode": row["gpcode"],
                "rainfall_mm": float(vals[di]) if np.isfinite(vals[di]) else np.nan,
                "n_cells": int(row["n_cells"]),
                "mapping_method": row["mapping_method"],
                "fallback_distance_m": row.get("fallback_distance_m", np.nan),
            })
    daily = pd.DataFrame(recs)

    admin = rename_admin(panchayats)
    daily = admin.drop(columns="geometry").merge(
        daily, left_on="panchayat_id", right_on="gpcode", how="left"
    )
    daily = daily.drop(columns=["gpcode"], errors="ignore")
    daily["mapping_method"] = daily["mapping_method"].fillna("unmapped")
    missing_dates = daily["date"].isna()
    if missing_dates.any():
        unmapped_ids = daily.loc[missing_dates, "panchayat_id"].unique()
        base = admin.drop(columns="geometry")
        base = base[base["panchayat_id"].isin(unmapped_ids)]
        extra = []
        for date in field.dates:
            chunk = base.copy()
            chunk["date"] = date
            chunk["rainfall_mm"] = np.nan
            chunk["n_cells"] = 0
            chunk["mapping_method"] = "unmapped"
            chunk["fallback_distance_m"] = np.nan
            extra.append(chunk)
        daily = pd.concat([daily[~missing_dates], *extra], ignore_index=True)
    daily["n_cells"] = daily["n_cells"].fillna(0).astype(int)

    daily["wet_day"] = daily["rainfall_mm"] >= wet_mm
    daily = daily.sort_values(["state", "district", "block_name",
                               "panchayat_name", "date"]).reset_index(drop=True)

    summary = (
        daily.groupby(CSV_ATTRS + ["mapping_method"], dropna=False, as_index=False)
        .agg(
            n_days=("date", "nunique"),
            rainfall_mean_mm=("rainfall_mm", "mean"),
            rainfall_max_mm=("rainfall_mm", "max"),
            rainfall_min_mm=("rainfall_mm", "min"),
            n_wet_days=("wet_day", "sum"),
            n_cells=("n_cells", "max"),
            fallback_distance_m=("fallback_distance_m", "mean"),
        )
    )
    # mapping_method can vary if we ever mixed; keep first non-unmapped
    geom = admin[["panchayat_id", "geometry"]]
    return daily, summary.merge(geom, on="panchayat_id", how="left")


def qc_report(field: Layer1Field, daily: pd.DataFrame, method: str,
              max_fallback_km: float, region_label: str) -> dict:
    one = daily.drop_duplicates("panchayat_id")
    n = len(one)
    n_direct = int(one["mapping_method"].isin(["direct_grid", "area_weighted"]).sum())
    n_fb = int((one["mapping_method"] == "nearest_fallback").sum())
    n_un = int((one["mapping_method"] == "unmapped").sum())
    fb = one.loc[one["mapping_method"] == "nearest_fallback", "fallback_distance_m"]
    rain = daily["rainfall_mm"]
    return {
        "layer1_source": str(field.source),
        "reconstructed_grid": field.reconstructed_grid,
        "dates": field.dates,
        "n_days": len(field.dates),
        "grid": {
            "n_lat": int(len(field.latitude)),
            "n_lon": int(len(field.longitude)),
            "lat_min": float(field.latitude.min()),
            "lat_max": float(field.latitude.max()),
            "lon_min": float(field.longitude.min()),
            "lon_max": float(field.longitude.max()),
            "resolution_deg": 0.05,
        },
        "region": region_label,
        "method": method,
        "max_fallback_km": max_fallback_km,
        "panchayats": {
            "total": n,
            "mapped": n - n_un,
            "direct_or_area": n_direct,
            "nearest_fallback": n_fb,
            "unmapped": n_un,
        },
        "fallback_distance_m": {
            "n": int(fb.notna().sum()),
            "min": float(fb.min()) if len(fb) else None,
            "median": float(fb.median()) if len(fb) else None,
            "max": float(fb.max()) if len(fb) else None,
        },
        "rainfall_mm": {
            "min": float(np.nanmin(rain)) if rain.notna().any() else None,
            "mean": float(np.nanmean(rain)) if rain.notna().any() else None,
            "max": float(np.nanmax(rain)) if rain.notna().any() else None,
        },
    }


def _save_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        alt = path.with_name(path.stem + "_updated" + path.suffix)
        df.to_csv(alt, index=False)
        print(f"[warn] {path.name} is locked; wrote {alt.name} instead")
        return alt


def _save_json(payload: dict, path: Path) -> Path:
    text = json.dumps(payload, indent=2)
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        alt = path.with_name(path.stem + "_updated" + path.suffix)
        alt.write_text(text, encoding="utf-8")
        print(f"[warn] {path.name} is locked; wrote {alt.name} instead")
        return alt


def _save_geojson(gdf: gpd.GeoDataFrame, path: Path) -> Path:
    try:
        gdf.to_file(path, driver="GeoJSON")
        return path
    except PermissionError:
        alt = path.with_name(path.stem + "_updated" + path.suffix)
        gdf.to_file(alt, driver="GeoJSON")
        print(f"[warn] {path.name} is locked; wrote {alt.name} instead")
        return alt


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_outputs(daily: pd.DataFrame, summary: gpd.GeoDataFrame,
                  field: Layer1Field, map_date: str, out_dir: Path,
                  region_label: str, qc: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_cols = ["date", *CSV_ATTRS, "rainfall_mm", "n_cells",
                "mapping_method", "fallback_distance_m"]
    daily_path = _save_csv(daily[csv_cols], out_dir / "panchayat_weather.csv")

    sum_cols = [*CSV_ATTRS, "n_days", "rainfall_mean_mm", "rainfall_min_mm",
                "rainfall_max_mm", "n_wet_days", "n_cells",
                "mapping_method", "fallback_distance_m"]
    summary_path = _save_csv(
        pd.DataFrame(summary.drop(columns="geometry", errors="ignore"))[sum_cols],
        out_dir / "panchayat_summary.csv")

    snap = daily[daily["date"] == map_date].copy()
    geo = summary[["panchayat_id", "geometry"]].drop_duplicates("panchayat_id")
    geo = geo.merge(snap[[*CSV_ATTRS, "rainfall_mm", "n_cells",
                          "mapping_method", "fallback_distance_m"]],
                    on="panchayat_id", how="left")
    geo = gpd.GeoDataFrame(geo, geometry="geometry", crs="EPSG:4326")
    geojson_path = _save_geojson(geo, out_dir / "panchayat_weather.geojson")

    qc["map_date"] = map_date
    qc["outputs"] = {
        "csv": str(daily_path),
        "summary_csv": str(summary_path),
        "geojson": str(geojson_path),
    }
    qc_path = _save_json(qc, out_dir / "layer2_qc.json")

    map_path = make_map(geo, out_dir, region_label, map_date)
    qc["outputs"]["map"] = str(map_path)
    qc_path = _save_json(qc, qc_path)
    return {"csv": daily_path, "summary": summary_path, "geojson": geojson_path,
            "qc": qc_path, "map": map_path}


def make_map(result: gpd.GeoDataFrame, out_dir: Path,
             region_label: str, map_date: str) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    mapped = result[result["rainfall_mm"].notna()].copy()
    unmapped = result[result["rainfall_mm"].isna()].copy()
    if not mapped.empty:
        mapped.plot(
            column="rainfall_mm", ax=ax, cmap="YlGnBu",
            vmin=mapped["rainfall_mm"].min(), vmax=mapped["rainfall_mm"].max(),
            linewidth=0.4, edgecolor="black", legend=True,
            legend_kwds={"label": "Rainfall (mm/day)", "orientation": "vertical"},
        )
    if not unmapped.empty:
        unmapped.plot(ax=ax, color="lightgrey", edgecolor="black",
                      linewidth=0.4, hatch="///")
    if len(result) <= 40:
        for _, row in result.iterrows():
            if row.geometry is None:
                continue
            cx = row.geometry.centroid
            name = str(row.get("panchayat_name", ""))[:12]
            rain = row["rainfall_mm"]
            label = f"{name}\n{rain:.1f}" if pd.notna(rain) else f"{name}\nN/A"
            ax.annotate(label, xy=(cx.x, cx.y), ha="center", va="center",
                        fontsize=5, color="black")
    ax.set_title(f"Panchayat daily rainfall - {region_label}\n"
                 f"{map_date}  |  Layer 1 0.05° zonal mean",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    fig.tight_layout()
    map_path = out_dir / "panchayat_weather_map.png"
    fig.savefig(map_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return map_path


def print_summary(qc: dict, paths: dict) -> None:
    p = qc["panchayats"]
    print()
    print("=" * 50)
    print("LAYER 2 RESULTS")
    print("=" * 50)
    print(f"Region:              {qc['region']}")
    print(f"Layer 1:             {qc['layer1_source']}")
    print(f"Dates:               {qc['n_days']} ({qc['dates'][0]} .. {qc['dates'][-1]})")
    print(f"Method:              {qc['method']}")
    print()
    print(f"Total Panchayats:    {p['total']}")
    print(f"Mapped:              {p['mapped']}")
    print(f"  - direct/area:     {p['direct_or_area']}")
    print(f"  - nearest_fallback:{p['nearest_fallback']}")
    print(f"Unmapped:            {p['unmapped']}")
    r = qc["rainfall_mm"]
    if r["mean"] is not None:
        print()
        print(f"Rainfall min/mean/max: {r['min']:.2f} / {r['mean']:.2f} / {r['max']:.2f} mm")
    print()
    for k, v in paths.items():
        print(f"{k:8s} {v}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "--npz", "--npy", dest="input", default=None,
                    help="Layer 1 .npz (preferred) or .npy")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--block", default=None, help="LGD blkname (optional)")
    group.add_argument("--state", default=None, help="LGD stname (optional)")
    ap.add_argument("--roi-clip", dest="roi_clip", action="store_true", default=True,
                    help="Keep only Panchayats intersecting the Layer 1 grid (default)")
    ap.add_argument("--no-roi-clip", dest="roi_clip", action="store_false")
    ap.add_argument("--method", choices=("center", "area"), default="center",
                    help="center = mean of cell centers in polygon; "
                         "area = intersection-area weighted cell mean")
    ap.add_argument("--dates", default=None,
                    help="Comma-separated YYYY-MM-DD subset of Layer 1 dates")
    ap.add_argument("--map-date", default=None, help="Date to draw / put in GeoJSON")
    ap.add_argument("--max-fallback-km", type=float, default=config.MAX_FALLBACK_KM)
    ap.add_argument("--wet-mm", type=float, default=config.WET_DAY_MM)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--panchayats", default=str(DEFAULT_PARQUET))
    args = ap.parse_args()

    explicit = Path(args.input) if args.input else None
    src = discover_layer1_path(explicit)
    field = load_layer1(src)
    wanted = [d.strip() for d in args.dates.split(",")] if args.dates else None
    field = select_dates(field, wanted)
    map_date = args.map_date or field.dates[0]
    if map_date not in field.dates:
        sys.exit(f"[ERROR] --map-date {map_date} not in selected dates")

    dlat = float(np.median(np.diff(field.latitude))) if len(field.latitude) > 1 else 0.05
    dlon = float(np.median(np.diff(field.longitude))) if len(field.longitude) > 1 else 0.05
    roi_box = box(
        float(field.longitude.min()) - dlon / 2,
        float(field.latitude.min()) - dlat / 2,
        float(field.longitude.max()) + dlon / 2,
        float(field.latitude.max()) + dlat / 2,
    ) if args.roi_clip else None

    if args.block:
        region_label = f"Block {args.block.strip().upper()}"
    elif args.state:
        region_label = f"State {args.state.strip().upper()}"
    else:
        region_label = "Layer 1 ROI"

    print(f"\n{'=' * 50}")
    print("Layer 2: Panchayat Weather Mapping")
    print(region_label)
    print(f"{'=' * 50}\n")

    panchayats = load_panchayats(
        Path(args.panchayats), args.block, args.state, roi_box)
    points = grid_to_points(field)
    lat_step = float(np.median(np.diff(field.latitude))) if len(field.latitude) > 1 else 0.05
    lon_step = float(np.median(np.diff(field.longitude))) if len(field.longitude) > 1 else 0.05
    if args.method == "area":
        cells = grid_to_cells(field)
        assignment = assign_cells_area(
            cells, panchayats, args.max_fallback_km, points)
    else:
        assignment = assign_cells_center(
            points, panchayats, args.max_fallback_km,
            lat_step=lat_step, lon_step=lon_step)

    daily, summary = aggregate_rainfall(
        field, assignment, panchayats, args.wet_mm)
    qc = qc_report(field, daily, args.method, args.max_fallback_km, region_label)
    paths = write_outputs(
        daily, gpd.GeoDataFrame(summary, geometry="geometry", crs="EPSG:4326"),
        field, map_date, Path(args.out), region_label, qc)
    print_summary(qc, paths)


if __name__ == "__main__":
    main()
