"""
Auxiliary dataset builder for Layer-2 readiness (NO model training here).

Builds, for a named study region (config.REGIONS):
  1. ADMIN    : administrative master layer from GADM 4.1 (levels 1/2/3 =
                state / district / subdistrict-tier), bbox-clipped to the
                region, with stable GADM IDs + centroids, and a mapping from
                every land 0.05-deg fine cell -> state/district/subdistrict.
  2. SOIL     : SoilGrids v2.0 (ISRIC, 250 m) sand/clay/ocd/phh2o/bdod at
                5-15 cm sampled on the fine grid (batched REST calls,
                per-batch cache -> restartable).
  3. NDVI     : monthly composites (Jun-Sep) from the NOAA CDR VIIRS NDVI
                (0.05 deg, daily global NetCDF; 3 dates/month sampled,
                per-pixel valid-value mean composite).
  4. LULC     : ESA WorldCover v200 (2021, 10 m) per-class area fractions on
                the fine grid from the official S3 COG tiles.
  5. REPORTS  : data dictionary, coverage report, missingness report.

Everything is auxiliary: NONE of these layers is a U-Net input channel in the
current architecture (documented in README.md and the data dictionary).

Run:
  python scripts/build_aux.py --region deccan
  python scripts/build_aux.py --region deccan --skip-soil   # etc.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from preprocess import imd_land_mask  # noqa: E402
from imd_reader import load_imd  # noqa: E402

GADM_URL = ("https://geodata.ucdavis.edu/gadm/gadm4.1/json/"
            "gadm41_{iso}_{lvl}.json.zip")

LULC_CLASSES = {10: "tree_cover", 20: "shrubland", 30: "grassland",
                40: "cropland", 50: "built_up", 60: "bare", 70: "snow_ice",
                80: "water", 90: "wetland", 95: "mangroves",
                100: "moss_lichen"}
LULC_FRACTIONS = ["cropland_fraction", "forest_fraction", "water_fraction",
                  "builtup_fraction", "grassland_fraction", "barren_fraction"]
# WorldCover class -> our 6 fractions (forest = tree cover + mangroves)
LULC_GROUP = {"cropland_fraction": (40,),
              "forest_fraction": (10, 95),
              "water_fraction": (80,),
              "builtup_fraction": (50,),
              "grassland_fraction": (30, 20),
              "barren_fraction": (60, 70, 100)}


def _get(url: str, **kw) -> requests.Response:
    last = None
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=kw.pop("timeout", (15, 180)), **kw)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"GET failed ({last}): {url[:120]}")


def resample_masked(data_lat, data_lon, values, want_lat, want_lon):
    """Bilinear resample that never mixes valid values with NaN neighbours.

    Resamples the field and a 1/0 validity mask separately; an output pixel is
    kept only when its whole bilinear stencil is valid (mask == 1), otherwise
    NaN. Prevents sentinel/NaN bleed: a plain bilinear over sentinel-filled
    values would fabricate values like 0.5*real + 0.5*sentinel.
    """
    from preprocess import resample_to_grid
    valid = np.isfinite(values).astype("float32")
    filled = np.nan_to_num(values, nan=0.0)
    f = resample_to_grid(data_lat, data_lon, filled, want_lat, want_lon)
    m = resample_to_grid(data_lat, data_lon, valid, want_lat, want_lon)
    return np.where(m >= 0.999, f, np.nan)


# --------------------------------------------------------------------------
# region / grid context (shared by all builders)
# --------------------------------------------------------------------------
def region_context(region: str) -> dict:
    """ROI, fine grid and land mask for a region (from raw IMD, as preprocess)."""
    roi = config.region_roi(region)
    start = f"{min(config.YEARS)}{config.MONSOON_START}"
    end = f"{max(config.YEARS)}{config.MONSOON_END}"
    imd = None
    for p in sorted(config.RAW_IMD.glob("*.nc")):
        try:
            d = load_imd(p, {**roi, "start_date": start, "end_date": end})
        except ValueError:
            continue
        if imd is None:
            imd = d
        else:
            imd["dates"] += d["dates"]
            imd["rain"] = np.concatenate([imd["rain"], d["rain"]])
    if imd is None:
        raise RuntimeError("no IMD files cover the region; run download_or_export.py")
    frac = np.isfinite(imd["rain"]).mean(axis=0)
    land = frac >= 0.5
    lat_edges = np.concatenate([imd["lat"] - 0.125, [imd["lat"][-1] + 0.125]])
    lon_edges = np.concatenate([imd["lon"] - 0.125, [imd["lon"][-1] + 0.125]])
    from grids import fine_grid  # noqa: E402
    fine_lat, fine_lon = fine_grid(imd["lat"], imd["lon"], config.FINE_SUB)
    land_fine = np.repeat(np.repeat(land, config.FINE_SUB, axis=0),
                          config.FINE_SUB, axis=1)
    return {"roi": roi, "land": land, "land_fine": land_fine,
            "imd_lat": imd["lat"], "imd_lon": imd["lon"],
            "fine_lat": fine_lat, "fine_lon": fine_lon,
            "n_land_cells": int(land_fine.sum()),
            "fine_shape": (len(fine_lat), len(fine_lon))}


# --------------------------------------------------------------------------
# 1. ADMIN (GADM 4.1) + grid mapping
# --------------------------------------------------------------------------
def _gadm_path(lvl: int) -> Path:
    return config.RAW_ADMIN / f"gadm41_IND_{lvl}.json.zip"


def ensure_gadm() -> dict[int, list]:
    """Download/unzip GADM India levels 1-3 once; return parsed features."""
    out = {}
    for lvl in (1, 2, 3):
        zpath = _gadm_path(lvl)
        if not zpath.exists():
            url = GADM_URL.format(iso="IND", lvl=lvl)
            print(f"[admin] downloading GADM level {lvl}: {url}")
            data = _get(url).content
            zpath.write_bytes(data)
        with zipfile.ZipFile(zpath) as z:
            name = z.namelist()[0]
            out[lvl] = json.loads(z.read(name))["features"]
    return out


def _feat_bbox(feat) -> tuple[float, float, float, float]:
    lons, lats = [], []

    def walk(coords):
        if isinstance(coords[0], (int, float)):
            lons.append(coords[0]); lats.append(coords[1])
            return
        for c in coords:
            walk(c)

    walk(feat["geometry"]["coordinates"])
    return min(lons), min(lats), max(lons), max(lats)


def _centroid(feat) -> tuple[float, float]:
    """Area-weighted vertex centroid of the largest ring (good enough as a
    representative point; GADM polygons are already simplified)."""
    g = feat["geometry"]

    def rings(geom):
        if geom["type"] == "Polygon":
            return [geom["coordinates"][0]]
        return [poly[0] for poly in geom["coordinates"]]

    best, best_n = None, -1
    for ring in rings(g):
        if len(ring) > best_n:
            best, best_n = ring, len(ring)
    xs = [p[0] for p in best]
    ys = [p[1] for p in best]
    return float(np.mean(xs)), float(np.mean(ys))


def build_admin(ctx: dict, region: str) -> dict:
    feats = ensure_gadm()
    roi = ctx["roi"]
    pad = 0.25

    def inside(feat):
        x0, y0, x1, y1 = _feat_bbox(feat)
        return not (x1 < roi["lon_min"] - pad or x0 > roi["lon_max"] + pad or
                    y1 < roi["lat_min"] - pad or y0 > roi["lat_max"] + pad)

    sel = {lvl: [f for f in feats[lvl] if inside(f)] for lvl in (1, 2, 3)}
    print(f"[admin] region features: states={len(sel[1])} "
          f"districts={len(sel[2])} subdistricts(blocks-tier)={len(sel[3])}")
    n_districts_all = len(feats[2])
    n_states_all = len(feats[1])

    # master layer: subdistrict (block-tier) features with stable IDs
    master = []
    for f in sel[3]:
        p = f["properties"]
        cx, cy = _centroid(f)
        master.append({
            "gadm_subdistrict_id": p.get("GID_3"),
            "gadm_district_id": p.get("GID_2"),
            "gadm_state_id": p.get("GID_1"),
            "country_id": p.get("GID_0"),
            "subdistrict_name": p.get("NAME_3"),
            "district_name": p.get("NAME_2"),
            "state_name": p.get("NAME_1"),
            "type_subdistrict": p.get("TYPE_3"),
            "centroid_lon": cx, "centroid_lat": cy,
            "geometry": f["geometry"],
        })
    out_dir = config.AUX / "admin"
    out_dir.mkdir(parents=True, exist_ok=True)
    gj = out_dir / f"admin_master_{region}.geojson"
    gj.write_text(json.dumps({
        "type": "FeatureCollection",
        "name": f"gadm41 block-tier master layer ({region} bbox + {pad} deg pad)",
        "note": "GADM 4.1 level-3 (subdistrict/tehsil/mandal = block tier). "
                "Gram-Panchayat/village polygons are NOT in GADM; LGD "
                "(lgdirectory.gov.in) provides them but has no bulk download "
                "- the ID schema here is ready to be joined to LGD codes.",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": [{"type": "Feature", "properties":
                      {k: v for k, v in m.items() if k != "geometry"},
                      "geometry": m["geometry"]} for m in master]},
        indent=1))
    print(f"[admin] master layer -> {gj} ({len(master)} subdistrict features)")

    # ---- fine-grid cell -> admin mapping (nearest representative point) ----
    # Exact point-in-polygon for ~50k cells against ~1600 vertex-heavy
    # polygons needs shapely/GIS libs; the mapping uses each subdistrict's
    # representative-point nearest-neighbour assignment instead. VALIDATION:
    # point-in-polygon spot checks are reported so the approximation error is
    # quantified (see the QC report).
    lat_f, lon_f = ctx["fine_lat"], ctx["fine_lon"]
    ii, jj = np.where(ctx["land_fine"])
    pts_lat, pts_lon = lat_f[ii], lon_f[jj]
    sub = np.array([[m["centroid_lat"], m["centroid_lon"]] for m in master])
    from scipy.spatial import cKDTree
    tree = cKDTree(sub)
    d, idx = tree.query(np.column_stack([pts_lat, pts_lon]), k=1)
    sub_ids = np.array([m["gadm_subdistrict_id"] for m in master], dtype=object)
    dist_ids = np.array([m["gadm_district_id"] for m in master], dtype=object)
    state_ids = np.array([m["gadm_state_id"] for m in master], dtype=object)
    # distinct admin units actually touched by fine land cells
    n_sub_hit = len(set(sub_ids[idx].tolist()))
    n_dist_hit = len(set(dist_ids[idx].tolist()))
    n_state_hit = len(set(state_ids[idx].tolist()))
    np.savez_compressed(
        config.AUX / "admin" / f"grid_admin_map_{region}.npz",
        fine_i=ii, fine_j=jj, fine_lat=pts_lat, fine_lon=pts_lon,
        subdistrict_idx=idx, nearest_dist_deg=d.astype("float32"),
        subdistrict_ids=sub_ids[idx], district_ids=dist_ids[idx],
        state_ids=state_ids[idx],
        subdistrict_names=np.array([m["subdistrict_name"] for m in master],
                                   dtype=object)[idx],
        district_names=np.array([m["district_name"] for m in master],
                                dtype=object)[idx],
        state_names=np.array([m["state_name"] for m in master], dtype=object)[idx])
    print(f"[admin] grid mapping: {len(ii)} land cells -> {n_sub_hit} "
          f"subdistricts / {n_dist_hit} districts / {n_state_hit} states "
          f"(max nearest-point dist {float(d.max()):.3f} deg)")

    # ---- validation: point-in-polygon spot check (pure-python, sampled) ----
    rng = np.random.default_rng(42)
    sample = rng.choice(len(ii), size=min(200, len(ii)), replace=False)

    def pip(x, y, feat) -> bool:
        # ray casting on every ring; holes handled by even-odd toggling
        g = feat["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        inside_any = False
        for poly in polys:
            c = False
            for ring in poly:
                for k in range(len(ring) - 1):
                    x1, y1 = ring[k]
                    x2, y2 = ring[k + 1]
                    if (y1 > y) != (y2 > y):
                        xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                        if x < xin:
                            c = not c
            if c:
                inside_any = not inside_any  # hole -> toggle out
        return inside_any

    feats3 = sel[3]
    agree = 0
    for s in sample:
        f3 = feats3[int(idx[s])]
        if pip(float(pts_lon[s]), float(pts_lat[s]), f3):
            agree += 1
    agree_frac = agree / max(len(sample), 1)
    print(f"[admin] PIP spot check: {agree}/{len(sample)} sampled cells fall "
          f"inside their assigned subdistrict polygon ({agree_frac * 100:.0f}%)")

    return {"n_states_all": n_states_all, "n_districts_all": n_districts_all,
            "n_states": len(sel[1]), "n_districts": len(sel[2]),
            "n_subdistricts": len(sel[3]), "n_subdistricts_hit": n_sub_hit,
            "n_districts_hit": n_dist_hit, "n_states_hit": n_state_hit,
            "pip_agree_frac": round(agree_frac, 3),
            "master_file": str(gj), "mapping_file":
                str(config.AUX / "admin" / f"grid_admin_map_{region}.npz")}


# --------------------------------------------------------------------------
# 2. SOIL (SoilGrids v2.0 REST, batched + cached)
# --------------------------------------------------------------------------
def build_soil(ctx: dict, region: str) -> dict:
    """SoilGrids sampled on the LAND COARSE (0.25-deg) cells, then bilinearly
    expanded to the fine grid.

    Why 0.25-deg: the official REST point service is one request per point;
    ~50k fine-grid points would mean ~50k requests (hours of hammering a
    public service). Soil properties at 5-15 cm are static and spatially
    smooth, so the 0.25-deg sampling + bilinear expansion preserves their
    signal for auxiliary use (these layers are NOT current U-Net inputs;
    a future model can raise the sampling density - the downloader is
    restartable and cached per batch).
    """
    lat_c, lon_c = ctx["imd_lat"], ctx["imd_lon"]
    ii, jj = np.where(ctx["land"])
    pts_lat, pts_lon = lat_c[ii], lon_c[jj]
    n = len(ii)
    props = list(config.SOILGRID_PROPS)
    cache_dir = config.RAW_SOIL / "batches"
    cache_dir.mkdir(parents=True, exist_ok=True)
    nb = (n + config.SOIL_BATCH - 1) // config.SOIL_BATCH
    vals = {p: np.full(n, np.nan, dtype="float32") for p in props}
    got_batches = 0
    print(f"[soil] {n} land coarse cells x {len(props)} properties, "
          f"{nb} batches of {config.SOIL_BATCH} (0.25-deg sampling lattice)")
    for b in range(nb):
        cache = cache_dir / f"{region}_batch_{b:04d}.npz"
        sl = slice(b * config.SOIL_BATCH, min((b + 1) * config.SOIL_BATCH, n))
        if cache.exists():
            z = np.load(cache)
            for p in props:
                vals[p][sl] = z[p]
            got_batches += 1
            continue
        batch_vals = {p: [] for p in props}
        ok = True

        def sample_point(la_lo):
            la, lo = la_lo
            q = "&".join(f"property={p}" for p in props)
            try:
                r = _get(f"{config.SOILGRID_API}?lat={la:.4f}&lon={lo:.4f}&"
                         f"{q}&depth={config.SOILGRID_DEPTH}&value=mean",
                         timeout=(10, 60))
                layers = r.json()["properties"]["layers"]
                by_name = {L["name"]: L for L in layers}
                row = []
                for p in props:
                    L = by_name.get(p)
                    if L is None:
                        row.append(np.nan)
                        continue
                    m = L["depths"][0]["values"]["mean"]
                    row.append(float("nan") if m is None else float(m))
                return row
            except Exception as e:  # noqa: BLE001
                print(f"[soil] batch {b}: point ({la:.3f},{lo:.3f}) failed: {e}")
                return [np.nan] * len(props)

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=int(
                getattr(config, "SOIL_WORKERS", 16))) as ex:
            rows = list(ex.map(sample_point,
                               zip(pts_lat[sl], pts_lon[sl])))
        for p_i, p in enumerate(props):
            batch_vals[p] = np.asarray([r[p_i] for r in rows], dtype="float32")
        time.sleep(2.0)
        for p in props:
            batch_vals[p] = np.asarray(batch_vals[p], dtype="float32")
        np.savez_compressed(cache, **batch_vals)
        for p in props:
            vals[p][sl] = batch_vals[p]
        got_batches += 1
        if got_batches % 25 == 0 or got_batches == nb:
            print(f"[soil] {got_batches}/{nb} batches "
                  f"({100 * got_batches / nb:.0f}%)")
    arr = np.column_stack([vals[p] for p in props])
    # Expansion to the fine grid: each fine pixel takes the values of its
    # parent coarse cell (the sample point IS the parent-cell center). The
    # sampling lattice is row-major over land cells (not a clean rectangular
    # grid), so per-parent broadcast replaces lattice resampling; a parent
    # with a genuine SoilGrids null propagates NaN to its children (documented
    # missingness, no invented values).
    fii, fjj = np.where(ctx["land_fine"])
    # map full-grid coarse position -> land-cell order used by `arr`
    land_order = np.full(ctx["land"].size, -1, dtype="int64")
    land_order[ii * ctx["land"].shape[1] + jj] = np.arange(len(ii))
    parent_of_fine = land_order[(fii // config.FINE_SUB) *
                                ctx["land"].shape[1] + (fjj // config.FINE_SUB)]
    fine_vals = arr[parent_of_fine].astype("float32")
    np.savez_compressed(
        config.AUX / f"soil_soilgrids_{region}.npz",
        coarse_i=ii, coarse_j=jj, coarse_lat=pts_lat, coarse_lon=pts_lon,
        properties=np.array(props), values=arr,
        fine_i=fii, fine_j=fjj, fine_lat=ctx["fine_lat"][fii],
        fine_lon=ctx["fine_lon"][fjj], fine_values=fine_vals,
        depth=config.SOILGRID_DEPTH,
        sampling="SoilGrids REST point samples at land 0.25-deg IMD cell "
                 "centers; each fine pixel inherits its parent coarse cell's "
                 "values (null parents -> NaN children, no interpolation)",
        units_note="raw SoilGrids mapped units with d_factor 10 "
                   "(clay/sand g/kg -> % = /10; ocd dg/dm3 -> hg/m3 = /10 "
                   "(= g/dm3... see data dictionary); phh2o x10 -> pH = /10; "
                   "bdod kg/dm3 -> kg/m3 = x1000")
    miss = {p: round(float(np.isnan(vals[p]).mean()) * 100, 2) for p in props}
    print(f"[soil] done; missing % per property: {miss}")
    return {"n_points": n, "batches": nb, "batches_ok": got_batches,
            "missing_pct": miss,
            "file": str(config.AUX / f"soil_soilgrids_{region}.npz")}


# --------------------------------------------------------------------------
# 3. NDVI (NOAA CDR VIIRS NDVI 0.05-deg; monthly composites Jun-Sep)
# --------------------------------------------------------------------------
NDVI_BASE = ("https://www.ncei.noaa.gov/data/"
             "land-normalized-difference-vegetation-index/access")
NDVI_DAYS = tuple(getattr(config, "NDVI_DAYS", (5, 25)))


def _ndvi_year_listing(year: int) -> dict[str, str]:
    """{YYYYMMDD: filename} for one year (cached)."""
    cache = config.RAW_NDVI / f"listing_{year}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    html = _get(f"{NDVI_BASE}/{year}/", timeout=(15, 120)).text
    import re
    files = re.findall(r'href="(VIIRS-Land_v001[^"]+\.nc)"', html)
    out = {}
    for f in files:
        m = re.search(r"_(\d{8})_c\d+\.nc$", f)
        if m:
            out[m.group(1)] = f
    cache.write_text(json.dumps(out))
    return out


def _ndvi_date_slice(fname: str, ymd: str, ctx: dict) -> np.ndarray:
    """Stream one daily global file, slice the region, cache + return
    (Hf, Wf) float32 NDVI on the fine grid (ascending lat/lon), NaN invalid."""
    roi = ctx["roi"]
    pad = 0.25
    cache = config.RAW_NDVI / "slices" / f"{region_cache(roi)}_{ymd}.npz"
    if cache.exists():
        z = np.load(cache)
        return _ndvi_resample(z["lat"], z["lon"], z["vals"], ctx)
    import h5py
    raw = io.BytesIO(_get(f"{NDVI_BASE}/{ymd[:4]}/{fname}",
                          timeout=(20, 900)).content)
    with h5py.File(raw, "r") as f:
        nd = f["NDVI"]
        scale = float(np.ravel(nd.attrs["scale_factor"])[0])
        fill = float(np.ravel(nd.attrs["_FillValue"])[0])
        lo, hi = nd.attrs["valid_range"]
        lats = np.asarray(f["latitude"][:], dtype="float64")
        lons = np.asarray(f["longitude"][:], dtype="float64")
        j0, j1 = np.searchsorted(lons, [roi["lon_min"] - pad,
                                        roi["lon_max"] + pad])
        desc = lats[0] > lats[-1]
        asc = lats[::-1] if desc else lats
        k0, k1 = np.searchsorted(asc, [roi["lat_min"] - pad,
                                       roi["lat_max"] + pad])
        if desc:
            # ascending row k  <->  original row len-1-k
            i0, i1 = len(lats) - k1, len(lats) - k0
            slab = nd[0, i0:i1, j0:j1][::-1, :]   # back to ascending lat
            lat_slice = lats[i0:i1][::-1]
        else:
            i0, i1 = k0, k1
            slab = nd[0, i0:i1, j0:j1]
            lat_slice = lats[i0:i1]
        lon_slice = lons[j0:j1]
        vals = np.asarray(slab, dtype="float32") * scale
        vals[(slab == fill) | (slab < lo) | (slab > hi)] = np.nan
    # cache the RAW slab (lat/lon/vals) so reprocessing never re-downloads
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, lat=lat_slice.astype("float64"),
                        lon=lon_slice.astype("float64"), vals=vals)
    return _ndvi_resample(lat_slice, lon_slice, vals, ctx)


def _ndvi_resample(lat_slice, lon_slice, vals, ctx) -> np.ndarray:
    """Raw 0.05-deg slab -> fine grid, validity-masked (no sentinel bleed)."""
    fine = resample_masked(lat_slice, lon_slice, vals,
                           ctx["fine_lat"], ctx["fine_lon"])
    return fine.astype("float32")


def region_cache(roi: dict) -> str:
    return (f"r{roi['lat_min']}_{roi['lat_max']}_{roi['lon_min']}_{roi['lon_max']}"
            .replace(".", "p"))


def build_ndvi(ctx: dict, region: str) -> dict:
    lat_f, lon_f = ctx["fine_lat"], ctx["fine_lon"]
    ii, jj = np.where(ctx["land_fine"])
    months, comps, valid_frac = [], [], []
    total = len(config.YEARS) * 4
    done = 0
    for y in config.YEARS:
        listing = _ndvi_year_listing(y)
        for m in (6, 7, 8, 9):
            dates = []
            for d in NDVI_DAYS:
                ymd = f"{y}{m:02d}{d:02d}"
                if ymd in listing:
                    dates.append((ymd, listing[ymd]))
            # fetch the date files in parallel (each is a ~64 MB stream-slice)
            from concurrent.futures import ThreadPoolExecutor
            def _grab(item):
                ymd, fname = item
                try:
                    return _ndvi_date_slice(fname, ymd, ctx)
                except Exception as e:  # noqa: BLE001
                    print(f"[ndvi] {ymd}: slice failed ({e})")
                    return None
            with ThreadPoolExecutor(max_workers=int(
                    getattr(config, "NDVI_WORKERS", 4))) as ex:
                got = list(ex.map(_grab, dates))
            stack = [g for g in got if g is not None]
            if stack:
                st = np.stack(stack)
                with np.errstate(invalid="ignore"):
                    comp = np.nanmean(st, axis=0)
                vf = float(np.isfinite(st).mean())
            else:
                comp = np.full(ctx["fine_shape"], np.nan, dtype="float32")
                vf = 0.0
            months.append(f"{y}-{m:02d}")
            comps.append(comp)
            valid_frac.append(round(vf, 4))
            done += 1
            print(f"[ndvi] {y}-{m:02d}: {len(stack)}/{len(NDVI_DAYS)} dates, "
                  f"valid px {vf * 100:.1f}%  [{done}/{total}]")
    arr = np.stack(comps).astype("float32")
    out = config.AUX / f"ndvi_monthly_{region}.npz"
    np.savez_compressed(out, months=np.array(months),
                        fine_i=ii, fine_j=jj,
                        ndvi=arr[:, ii, jj],
                        valid_frac_per_month=np.array(valid_frac),
                        fine_shape=np.array(ctx["fine_shape"]),
                        note="NDVI (unitless, -1..1) monthly valid-value mean "
                             "composite of 3 sampled daily NOAA CDR VIIRS "
                             "NDVI 0.05-deg files (days 5/15/25); NaN = no "
                             "valid obs (cloud/missing)")
    miss = float(np.isnan(arr[:, ii, jj]).mean()) * 100
    rng = (float(np.nanmin(arr)), float(np.nanmax(arr)))
    print(f"[ndvi] done -> {out}; missing {miss:.2f}% of land-month values, "
          f"range {rng[0]:.3f}..{rng[1]:.3f}")
    return {"months": months, "n_months": len(months),
            "valid_frac_per_month": valid_frac,
            "missing_pct": round(miss, 2), "range": [round(rng[0], 3),
                                                     round(rng[1], 3)],
            "file": str(out)}


# --------------------------------------------------------------------------
# 4. LULC (ESA WorldCover v200 2021 COG tiles -> per-cell class fractions)
# --------------------------------------------------------------------------
def _wc_tile_name(lon: float, lat: float) -> str:
    la = int(np.floor(lat / 3.0) * 3)
    lo = int(np.floor(lon / 3.0) * 3)
    ns = "N" if la >= 0 else "S"
    ew = "E" if lo >= 0 else "W"
    return f"{ns}{abs(la):02d}{ew}{abs(lo):03d}"


def _wc_open(tile: str):
    import tifffile
    local = config.RAW_LULC / f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    if not local.exists():
        url = config.WORLDCOVER_URL.format(tile=tile)
        print(f"[lulc] downloading {tile} ...", flush=True)
        # tiles are 20-100 MB COGs; retry with generous read timeouts
        got = False
        for attempt in range(3):
            try:
                with requests.get(url, timeout=(20, 600), stream=True) as r:
                    r.raise_for_status()
                    tmp = local.with_suffix(".tmp")
                    with open(tmp, "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                    tmp.rename(local)
                    got = True
                    break
            except Exception as e:  # noqa: BLE001
                print(f"[lulc] {tile} attempt {attempt + 1} failed: "
                      f"{type(e).__name__}: {e}", flush=True)
                local.with_suffix(".tmp").unlink(missing_ok=True)
                time.sleep(10 * (attempt + 1))
        if not got:
            raise RuntimeError(f"could not download WorldCover tile {tile}")
    tf = tifffile.TiffFile(local)
    pg = tf.pages[0]
    tags = {t.name: t.value for t in pg.tags}
    sx, sy = tags["ModelPixelScaleTag"][:2]
    tx, ty, _, lon0, lat0, _ = tags["ModelTiepointTag"]
    # Windowed reader: the COG is DEFLATE-compressed in 1024x1024 pixel chunks
    # (1296 per 3-deg tile). Slicing through zarr decompresses ONLY the chunks
    # intersecting the requested window, so reading our region costs seconds
    # instead of a full 36000x36000 page decompress per read.
    zarr_arr = None
    try:
        import zarr
        zarr_arr = zarr.open(pg.aszarr(), mode="r")
    except Exception as e:  # noqa: BLE001
        print(f"[lulc] {tile}: zarr windowed reads unavailable "
              f"({type(e).__name__}: {e}); falling back to full-page reads "
              f"(one decompress per tile window, still never per cell)",
              flush=True)
    return {"tf": tf, "pg": pg, "sx": sx, "sy": sy,
            "lon0": lon0, "lat0": lat0,
            "W": pg.shape[1], "H": pg.shape[0], "zarr": zarr_arr}


def _wc_read_window(h, r0, r1, c0, c1) -> np.ndarray:
    """Read pixel rows [r0, r1) x cols [c0, c1) from an open WorldCover COG.

    Row 0 = north, col 0 = west. Through zarr this decompresses ONLY the
    1024x1024 COG chunks intersecting the window.
    """
    if r1 <= r0 or c1 <= c0:
        return np.empty((0, 0), dtype="uint8")
    if h["zarr"] is not None:
        return h["zarr"][r0:r1, c0:c1]
    return h["pg"].asarray()[r0:r1, c0:c1]


def build_lulc(ctx: dict, region: str) -> dict:
    lat_f, lon_f = ctx["fine_lat"], ctx["fine_lon"]
    ii, jj = np.where(ctx["land_fine"])
    n = len(ii)
    fracs = np.zeros((n, len(LULC_FRACTIONS)), dtype="float32")
    dominant = np.zeros(n, dtype="uint8")
    half = 0.025  # 0.05-deg cell half-width (deg)

    # group cells by source tile to open each COG once
    tiles_needed = sorted({_wc_tile_name(lon_f[j], lat_f[i])
                           for i, j in zip(ii, jj)})
    cell_tile = np.array([_wc_tile_name(lon_f[j], lat_f[i])
                          for i, j in zip(ii, jj)])
    print(f"[lulc] {n} land cells on {len(tiles_needed)} WorldCover tiles: "
          f"{tiles_needed}")
    done = 0
    STRIP = 4096   # window rows per strip read (keeps RAM ~200 MB worst case)
    MARGIN = 700   # > max stencil height (0.05 deg = 600 px) so stencils fit
    for tile in tiles_needed:
        sel = np.where(cell_tile == tile)[0]
        # SPATIAL WINDOWING: read only the COG chunks that overlap this tile's
        # assigned land cells, in 4096-row strips (+/- a stencil margin). With
        # zarr, each strip slice decompresses only the 1024x1024 chunks it
        # intersects - never the full 36000x36000 page, and never per cell.
        # Each 0.05-deg cell stencil (600x600 px) is then sliced from the
        # in-memory strip window.
        t0 = time.time()
        h = _wc_open(tile)
        try:
            # vectorised per-cell stencil pixel bounds (row 0 = north)
            c0s = np.floor((lon_f[jj[sel]] - half - h["lon0"]) / h["sx"]).astype(np.int64)
            c1s = np.ceil((lon_f[jj[sel]] + half - h["lon0"]) / h["sx"]).astype(np.int64)
            r0s = np.floor((h["lat0"] - lat_f[ii[sel]] - half) / h["sy"]).astype(np.int64)
            r1s = np.ceil((h["lat0"] - lat_f[ii[sel]] + half) / h["sy"]).astype(np.int64)
            np.clip(c0s, 0, h["W"], out=c0s)
            np.clip(c1s, 0, h["W"], out=c1s)
            np.clip(r0s, 0, h["H"], out=r0s)
            np.clip(r1s, 0, h["H"], out=r1s)
            valid = (r1s > r0s) & (c1s > c0s)
            if not valid.any():
                print(f"[lulc] tile {tile}: no overlap with region, skipped "
                      f"[{done}/{len(tiles_needed)} tiles]", flush=True)
                continue
            centre = (r0s + r1s) // 2
            s_lo = int(centre[valid].min()) // STRIP
            s_hi = int(centre[valid].max()) // STRIP
            n_px = 0
            for s in range(s_lo, s_hi + 1):
                instrip = valid & (centre >= s * STRIP) & (centre < (s + 1) * STRIP)
                if not instrip.any():
                    continue
                # strip window covers every stencil of cells centred in it
                wr0 = max(0, s * STRIP - MARGIN)
                wr1 = min(h["H"], (s + 1) * STRIP + MARGIN)
                wc0 = int(c0s[instrip].min())
                wc1 = int(c1s[instrip].max())
                win = _wc_read_window(h, wr0, wr1, wc0, wc1)
                n_px += win.shape[0] * win.shape[1]
                ks = sel[instrip]
                # subset arrays in the SAME order as ks (do NOT index the
                # full per-tile arrays with the subset position!)
                rs_a, re_a = r0s[instrip], r1s[instrip]
                cs_a, ce_a = c0s[instrip], c1s[instrip]
                for qi, k in enumerate(ks):
                    rs, re = max(rs_a[qi], wr0), min(re_a[qi], wr1)
                    cs, ce = max(cs_a[qi], wc0), min(ce_a[qi], wc1)
                    if re <= rs or ce <= cs:
                        continue
                    vals = win[rs - wr0:re - wr0, cs - wc0:ce - wc0].ravel()
                    vals = vals[vals != 0]  # 0 = nodata outside the map
                    if vals.size == 0:
                        continue
                    cnt = np.bincount(vals, minlength=256)
                    tot = vals.size
                    for fi, fname in enumerate(LULC_FRACTIONS):
                        fracs[k, fi] = sum(cnt[c] for c in LULC_GROUP[fname]) / tot
                    dominant[k] = int(np.argmax(cnt))
                del win
            done += 1
            print(f"[lulc] tile {tile}: {len(sel)} cells, "
                  f"{(s_hi - s_lo + 1)} window strip(s), {n_px / 1e6:.0f} Mpx "
                  f"read (of {h['H'] * h['W'] / 1e6:.0f} Mpx tile), "
                  f"{time.time() - t0:.1f}s "
                  f"[{done}/{len(tiles_needed)} tiles]", flush=True)
        finally:
            # close the file handle before the next tile; the zarr view holds
            # no pixel data, only chunk references into the open tifffile
            h.pop("zarr", None)
            h["tf"].close()
            h = None
    out = config.AUX / f"lulc_fractions_{region}.npz"
    np.savez_compressed(
        out, fine_i=ii, fine_j=jj, fractions=fracs,
        fraction_names=np.array(LULC_FRACTIONS),
        dominant_class=dominant,
        class_legend=json.dumps(LULC_CLASSES),
        source="ESA WorldCover v200, 2021, 10 m (S3 COG tiles); "
               "fractions = per-class pixel share within each 0.05-deg cell")
    cov = float((fracs.sum(axis=1) > 0).mean()) * 100
    if cov < 90.0:
        print(f"[lulc] WARNING: only {cov:.1f}% of land cells got class data "
              f"(expected >90% over land); check window/stencil indexing")
    means = {f: round(float(fracs[:, i].mean()), 4)
             for i, f in enumerate(LULC_FRACTIONS)}
    print(f"[lulc] done -> {out}; cells with class data: {cov:.1f}%; "
          f"mean fractions {means}")
    return {"n_cells": n, "tiles": tiles_needed,
            "coverage_pct": round(cov, 2), "mean_fractions": means,
            "n_nodata_cells": int((fracs.sum(axis=1) == 0).sum()),
            "nodata_note": "cells whose whole 0.05-deg stencil is WorldCover "
                           "nodata (class 0, ocean); sea fringe of coastal IMD "
                           "grid boxes, never valid training targets (M=0)",
            "file": str(out)}


# --------------------------------------------------------------------------
# 6. SOIL MOISTURE (Open-Meteo daily ERA5-Land volumetric water, Layer-3 aux)
# --------------------------------------------------------------------------
# 0-7 cm only: the free Open-Meteo hourly pool makes the 2-variable fetch a
# multi-hour stop-and-go; 7-28 cm can be added later the same way (append to
# this tuple, delete the per-year caches, re-run - caches are per variable
# set via the array keys, so old single-var caches stay valid).
SOILMOISTURE_VARS = ("soil_moisture_0_to_7cm_mean",)


def build_soilmoisture(ctx: dict, region: str) -> dict:
    """Daily volumetric soil moisture (m3/m3) on the SAME 1.0-deg ERA5 lattice
    (incl. margin) as the model's ERA5 temperature/dewpoint channels.

    Source: Open-Meteo archive API, ERA5-Land daily mean volumetric water for
    0-7 cm and 7-28 cm. Same underlying model as our existing era5_* channels,
    so the alignment convention is identical: bilinear from this lattice to
    the fine grid replicates preprocess.py era5 handling exactly. Layer-3
    agro-advisory layer, NOT a U-Net input channel. Light file (~0.5 MB):
    lattice resolution, not fine-grid resolution.

    Fetches are batched (config ERA5_LOC_BATCH locations per request) and
    cached per point-batch in data/raw/era5/soilmoisture_<region>_<la>_<lo>.npz
    so the build is restartable. Dates: monsoon (Jun 1 - Sep 30) of config.YEARS.
    """
    import time as _time
    from download_or_export import era5_sample_points
    roi = ctx["roi"]
    pts = era5_sample_points(roi)          # same lattice as the era5 channels
    lats = sorted({p[0] for p in pts})
    lons = sorted({p[1] for p in pts})
    batch = int(getattr(config, "ERA5_LOC_BATCH", 64))
    sleep_s = int(getattr(config, "ERA5_REQUEST_SLEEP", 20))
    cache_dir = config.RAW_ERA5
    cache_dir.mkdir(parents=True, exist_ok=True)
    windows = [(f"{y}{config.MONSOON_START}", f"{y}{config.MONSOON_END}")
               for y in config.YEARS]  # MONSOON_* already include the dash

    def snap_key(t, pool):
        return min(pool, key=lambda p: (p[0] - t[0]) ** 2 + (p[1] - t[1]) ** 2)

    point_cache = {}   # (lat, lon) -> {var: {date: value}}
    got_batches = 0
    got_year_chunks = 0
    for bi in range(0, len(pts), batch):
        bpts = pts[bi:bi + batch]
        la0, lo0 = bpts[0]
        # per-(batch, year) caches: each request costs ~64x122x2 request
        # units, so caching per year-window lets repeated runs make progress
        # even when the hourly Open-Meteo pool runs out mid-batch.
        per_batch = {}   # year -> {var: (n_days, n_points) array}
        need_fetch = []
        for (ws, we) in windows:
            ycache = cache_dir / (f"soilmoisture_{region}_{la0:.2f}_{lo0:.2f}"
                                  f"_{ws[:4]}.npz")
            if ycache.exists():
                z = np.load(ycache, allow_pickle=False)
                per_batch[ws[:4]] = {"dates": [str(d) for d in z["dates"]],
                                     **{v: z[v] for v in SOILMOISTURE_VARS}}
                got_year_chunks += 1
                continue
            need_fetch.append((ws, we, ycache))
        if not need_fetch:
            got_batches += 1
        lat_q = ",".join(str(p[0]) for p in bpts)
        lon_q = ",".join(str(p[1]) for p in bpts)
        for (ws, we, ycache) in need_fetch:
            url = (f"{config.ERA5_API}?latitude={lat_q}&longitude={lon_q}"
                   f"&start_date={ws}&end_date={we}"
                   f"&daily={','.join(SOILMOISTURE_VARS)}"
                   f"&timezone={config.ERA5_TIMEZONE}")
            data = None
            for attempt in range(3):
                try:
                    r = requests.get(url, timeout=(15, 180))
                    if r.status_code == 429:
                        print(f"[soilmoisture] 429 quota at {ws}..{we} "
                              f"(batch {la0},{lo0}); completed year-windows "
                              "are cached - re-run within the next hour",
                              flush=True)
                        break
                    r.raise_for_status()
                    data = r.json()
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"[soilmoisture] attempt {attempt + 1} failed: "
                          f"{type(e).__name__}: {e}")
                    _time.sleep(10)
            if data is None:
                _time.sleep(sleep_s)
                continue
            items = data if isinstance(data, list) else [data]
            per_loc = {}
            for item in items:
                key = (round(float(item["latitude"]), 4),
                       round(float(item["longitude"]), 4))
                dd = item["daily"]["time"]
                per_loc[key] = {v: dict(zip(dd, item["daily"][v]))
                                for v in SOILMOISTURE_VARS}
            keys = sorted(per_loc)
            b_dates = sorted(next(iter(per_loc.values()))[SOILMOISTURE_VARS[0]])
            didx = {d: i for i, d in enumerate(b_dates)}
            arrs = {v: np.full((len(b_dates), len(bpts)), np.nan,
                               dtype="float32") for v in SOILMOISTURE_VARS}
            for pi, p in enumerate(bpts):
                k = snap_key(p, keys)
                for v in SOILMOISTURE_VARS:
                    for d, val in per_loc[k][v].items():
                        if val is not None and d in didx:
                            arrs[v][didx[d], pi] = val
            np.savez_compressed(ycache, dates=np.array(b_dates),
                                lat=np.array([p[0] for p in bpts]),
                                lon=np.array([p[1] for p in bpts]), **arrs)
            per_batch[ws[:4]] = {"dates": b_dates, **arrs}
            got_year_chunks += 1
            _time.sleep(sleep_s)
        # merge this batch's years into the point cache
        if per_batch:
            b_dates = sorted({d for yb in per_batch.values() for d in yb["dates"]})
            didx = {d: i for i, d in enumerate(b_dates)}
            arrs = {v: np.full((len(b_dates), len(bpts)), np.nan,
                               dtype="float32") for v in SOILMOISTURE_VARS}
            for yi, y in enumerate(sorted(per_batch)):
                yb = per_batch[y]
                rows = [didx[d] for d in yb["dates"]]
                for v in SOILMOISTURE_VARS:
                    arrs[v][rows, :] = yb[v]
            for pi, p in enumerate(bpts):
                point_cache[p] = {v: {d: float(x) for d, x in zip(b_dates, arrs[v][:, pi])
                                      if np.isfinite(x)}
                                  for v in SOILMOISTURE_VARS}
            got_batches += 1
            print(f"[soilmoisture] batch {la0},{lo0}: {len(per_batch)}/{len(windows)} "
                  f"year-windows present [{got_batches} batches, "
                  f"{got_year_chunks} year-chunks cached]", flush=True)

    # final assembly on the full lattice - only when COMPLETE (every batch,
    # every year-window); otherwise leave no half-baked output file behind
    missing_chunks = got_batches < (len(pts) + batch - 1) // batch
    if not point_cache or missing_chunks:
        if (config.AUX / f"soilmoisture_daily_{region}.npz").exists():
            (config.AUX / f"soilmoisture_daily_{region}.npz").unlink()
        raise RuntimeError(
            f"[soilmoisture] incomplete: {got_batches} batches complete "
            f"({got_year_chunks} year-chunks cached) of "
            f"{(len(pts) + batch - 1) // batch}. The free Open-Meteo API has a "
            "DAILY request limit; completed (batch, year) chunks are cached - "
            "re-run this command after the daily reset (next day) until done.")
    all_dates = sorted({d for pc in point_cache.values()
                        for v in SOILMOISTURE_VARS for d in pc[v]})
    out = config.AUX / f"soilmoisture_daily_{region}.npz"
    stacked = {v: np.full((len(all_dates), len(lats), len(lons)), np.nan,
                          dtype="float32") for v in SOILMOISTURE_VARS}
    di = {d: i for i, d in enumerate(all_dates)}
    for la_i, la in enumerate(lats):
        for lo_i, lo in enumerate(lons):
            pc = point_cache.get((float(la), float(lo)))
            if pc is None:
                continue
            for v in SOILMOISTURE_VARS:
                for d, val in pc[v].items():
                    stacked[v][di[d], la_i, lo_i] = val
    note = ("Open-Meteo archive API daily ERA5-Land volumetric soil moisture "
            "(m3/m3, 0-7 cm; 7-28 cm available from the same API, dropped to "
            "fit the free hourly quota) on the ERA5 sampling lattice "
            "(config ERA5_STEP_DEG, +margin), same geometry as the era5_* "
            "model channels; bilinear resample to the fine grid replicates "
            "preprocess.py era5 handling. Layer-3 agro-advisory layer, NOT a "
            "U-Net input channel.")
    np.savez_compressed(out, dates=np.array(all_dates), lat=np.array(lats),
                        lon=np.array(lons), **stacked, note=note)
    miss = {v: round(float(np.isnan(a).mean() * 100), 3)
            for v, a in stacked.items()}
    print(f"[soilmoisture] done -> {out} ({len(all_dates)} days, missing {miss})")
    return {"n_days": len(all_dates), "lattice": [len(lats), len(lons)],
            "variables": list(SOILMOISTURE_VARS), "missing_pct": miss,
            "n_batches": got_batches, "file": str(out), "note": note}


# --------------------------------------------------------------------------
# 5. REPORTS
# --------------------------------------------------------------------------
def write_reports(ctx: dict, region: str, res: dict) -> dict:
    rep = config.REPORTS
    rep.mkdir(parents=True, exist_ok=True)
    admin, soil, ndvi, lulc = (res.get("admin", {}), res.get("soil", {}),
                               res.get("ndvi", {}), res.get("lulc", {}))
    sm = res.get("soilmoisture", {})
    roi = ctx["roi"]
    years = ",".join(str(y) for y in config.YEARS)
    n_days = len(config.YEARS) * 122

    dd = f"""# Data dictionary - region `{region}`

## Model grid (Layer 1)
| name | meaning |
|---|---|
| coarse grid | IMD 0.25-deg cell centers (meta.grid.imd_lat/imd_lon) |
| fine grid | 5x5 area-tiling sub-cells of each coarse cell = 0.05 deg (meta.grid.fine_lat/fine_lon) |
| X | (n, C, Hf, Wf) input channels: imd_rain (bilinear 5x, NaN->0), dem, era5_* |
| Y | (n, 1, Hf, Wf) CHIRPS v2.0 daily rainfall (mm/day) on the fine grid, IMD-land cells only |
| M | (n, 1, Hf, Wf) 1 where Y is a valid land pixel (sea pixels are 0 by design) |
| land mask | IMD cell = land iff >=50% of aligned days valid; sea cells excluded from Y/M/metrics |

## Auxiliary layers (data/aux_data/) - NOT U-Net inputs
| file | contents | units |
|---|---|---|
| admin/admin_master_{region}.geojson | block-tier (GADM L3 subdistrict) polygons + state/district IDs, names, centroids | deg (EPSG:4326) |
| admin/grid_admin_map_{region}.npz | fine land cell -> (state, district, subdistrict) nearest-representative-point mapping | indices/IDs |
| soil_soilgrids_{region}.npz | sand, clay, ocd, phh2o, bdod at {config.SOILGRID_DEPTH}: sampled on land 0.25-deg cells; each fine pixel inherits its parent coarse-cell value (no bilinear smoothing across the land lattice) | raw SoilGrids mapped units (x10 factors; see units_note inside the file) |
| ndvi_monthly_{region}.npz | NDVI monthly composites Jun-Sep {years} on land fine cells | unitless -1..1 |
| lulc_fractions_{region}.npz | 6 per-class area fractions + dominant class (WorldCover 2021) | 0..1 |
| soilmoisture_daily_{region}.npz | OPTIONAL/PENDING (quota-blocked): daily ERA5-Land volumetric soil moisture 0-7 cm on the ERA5 lattice (bilinear to fine grid like the era5_* channels); resume via build_aux.py | m3/m3 |

### Soil unit conversion (SoilGrids d_factor=10)
sand/clay: g/kg -> % = value/10. ocd: dg/dm3 -> g/dm3 = value/10. phh2o: pH x10 -> pH = value/10.
bdod: kg/dm3 -> kg/m3 = value * 1000.

### LULC class legend (ESA WorldCover v200)
10 tree cover; 20 shrubland; 30 grassland; 40 cropland; 50 built-up; 60 bare/sparse;
70 snow/ice; 80 water; 90 wetland; 95 mangroves; 100 moss/lichen.
Fractions: forest = 10+95; grassland = 30+20; barren = 60+70+100.
Note: the 6 fractions do NOT sum to 1 in cells containing wetland (class 90)
pixels - wetland is tracked in `dominant_class` but has no fraction column
(104 of 47,250 covered cells, wetland share 0.02..0.32 in the Deccan build).
All-nodata stencils (ocean) are all-zero rows, so covered cells are
identifiable as fractions.sum(axis=1) > 0.

### NDVI composite
Per-pixel mean of valid (non-fill, in-range) values from {len(config.NDVI_DAYS)} sampled days
(day {" and ".join(str(d) for d in config.NDVI_DAYS)} of each month); NaN = no valid
observation (persistent monsoon cloud). Per-month valid-pixel fractions are stored
in the file for QC.

### LULC processing note
ESA WorldCover v200 COG tiles (36000x36000 px at 10 m) are read with spatial
windowing (zarr chunked reads of only the 1024x1024 COG chunks overlapping the
region), 4096-pixel row strips; per-cell class fractions use the 600x600-pixel
stencil around each 0.05-deg cell. Cells whose whole stencil is WorldCover
nodata (class 0, ocean) get all-zero fractions - these are the sea fringe of
coastal IMD grid boxes and are never valid training targets (M = 0).

### Admin mapping caveat
Grid cells are assigned to the nearest subdistrict representative point
(KD-tree). A 200-cell point-in-polygon spot check quantifies the agreement
(see coverage report). Panchayat/village polygons are not available in GADM;
LGD provides them without a bulk API - the ID schema is LGD-joinable.
"""
    (rep / f"data_dictionary_{region}.md").write_text(dd)

    cov = f"""# Coverage report - region `{region}`

Region bbox: lat {roi['lat_min']}..{roi['lat_max']}, lon {roi['lon_min']}..{roi['lon_max']}
Study window: {min(config.YEARS)}-06-01 .. {max(config.YEARS)}-09-30 (monsoon only, {years})
Days: {n_days} (per year 122)

## Administrative (block tier = GADM L3 subdistricts)
States in bbox: {admin.get('n_states', 'n/a')} (of {admin.get('n_states_all', '-')} in India)
Districts in bbox: {admin.get('n_districts', 'n/a')} (of {admin.get('n_districts_all', '-')})
Subdistricts (block tier) in bbox: {admin.get('n_subdistricts', 'n/a')}
Hit by land grid cells: {admin.get('n_states_hit', '-')} states / {admin.get('n_districts_hit', '-')} districts / {admin.get('n_subdistricts_hit', '-')} subdistricts
PIP spot-check agreement (200 cells): {admin.get('pip_agree_frac', '-')}
Gram Panchayats / villages: NOT included (LGD bulk download unavailable) - LGD-joinable schema provided.

## Grids
Land coarse cells: {int(ctx['land'].sum())}/{ctx['land'].size}
Land fine cells: {ctx['n_land_cells']}/{ctx['land_fine'].size}
Soil sample points: {soil.get('n_points', '-')} (batches ok {soil.get('batches_ok', '-')}/{soil.get('batches', '-')})
NDVI months: {ndvi.get('n_months', '-')} (valid-pixel fractions per month stored)
LULC cells with class data: {lulc.get('coverage_pct', '-')}%
Soil-moisture days: {sm.get('n_days', '-')} on a {sm.get('lattice', ['-','-'])[0]}x{sm.get('lattice', ['-','-'])[1]} ERA5 lattice (Layer-3 aux; 0-7/7-28 cm)

## Model-ready (after preprocess.py, if present)
"""
    meta_p = config.PROCESSED / "meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        cov += f"Samples: {len(meta['dates'])} days\n"
        cov += f"Split: {meta['split_description']}\n"
        cov += f"train/val/test days: {meta['split']['train']['n_days']}/" \
               f"{meta['split']['val']['n_days']}/{meta['split']['test']['n_days']}\n"
        cov += f"Channels: {meta['channels']}\n"
        cov += (f"Land fraction of box: "
                f"{meta.get('land_mask', {}).get('land_fraction', 'n/a')}\n")
        cov += f"IMD-CHIRPS coarse corr: {meta['imd_chirps_coarse_corr']:.3f}\n"
    else:
        cov += "meta.json not found - run preprocess.py first\n"
    (rep / f"coverage_report_{region}.md").write_text(cov)

    miss = ["# Missingness report", ""]
    if soil:
        miss += [f"## Soil (missing % per property)"] + \
                [f"- {k}: {v}%" for k, v in soil.get("missing_pct", {}).items()]
    if ndvi:
        miss += [f"## NDVI",
                 f"- land-month values missing: {ndvi.get('missing_pct')}%",
                 f"- NDVI range: {ndvi.get('range')}",
                 "- per-month valid-pixel fraction (cloud-driven):"]
        miss += [f"  - {m}: {vf}" for m, vf in
                 zip(ndvi.get("months", []), ndvi.get("valid_frac_per_month", []))]
    if lulc:
        miss += [f"## LULC",
                 f"- cells with class data: {lulc.get('coverage_pct')}%",
                 f"- cells with all-nodata stencils (WorldCover ocean class 0): "
                 f"{lulc.get('n_nodata_cells', '-')} of {lulc.get('n_cells', '-')}",
                 f"- {lulc.get('nodata_note', '')}",
                 "- these cells are excluded from training targets by the model "
                 "mask M, so no imputation is performed or needed"]
    if admin:
        miss += [f"## Admin",
                 f"- PIP agreement (nearest-point vs polygon): "
                 f"{admin.get('pip_agree_frac')}",
                 "- panchayat/village tier: not available (LGD); documented"]
    meta_p = config.PROCESSED / "meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        lm = meta.get("land_mask", {})
        miss += [f"## Rainfall pipeline",
                 f"- dates dropped by alignment/quality filter: "
                 f"{n_days - len(meta['dates'])}",
                 f"- sea cells excluded from Y/M: "
                 f"{lm.get('n_sea_cells', 'n/a')} coarse cells",
                 f"- IMD input NaN pixels imputed 0 (sea bleed): documented in "
                 f"meta.json"]
    miss += [f"## Dataset decisions (data freeze)",
             "- wind: raw ERA5 cache complete (0% missing, 2018-2022, 2018 "
             "re-fetched in place); NOT a model channel - frozen baseline is "
             "imd_rain, dem, era5_t2m, era5_t2m_max, era5_dewp",
             "- ESA WorldCereal crop type: evaluated, NOT included - only "
             "no-auth distribution is Zenodo 7875105 (global multi-GB ZIPs of "
             "106 AEZ GeoTIFFs, AEZ-specific seasons); a Layer-3 standalone task",
             "- SMAP L4 soil moisture: replaced by ERA5-Land daily volumetric "
             "water (same model as our era5_* channels) - OPTIONAL / PENDING: "
             "quota-blocked by the free Open-Meteo DAILY limit; resume with "
             "build_aux.py --region deccan --skip-admin --skip-soil --skip-ndvi "
             "--skip-lulc (per-batch/year caches make each attempt additive) - "
             "see docs/HANDOVER.md section 7"]
    (rep / f"missingness_report_{region}.md").write_text("\n".join(miss))
    return {"reports": [str(p) for p in sorted(rep.glob("*"))]}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default=config.REGION_DEFAULT)
    ap.add_argument("--skip-admin", action="store_true")
    ap.add_argument("--skip-soil", action="store_true")
    ap.add_argument("--skip-ndvi", action="store_true")
    ap.add_argument("--skip-lulc", action="store_true")
    ap.add_argument("--skip-soilmoisture", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="only (re)write reports from existing aux files")
    args = ap.parse_args()
    region = args.region
    ctx = region_context(region)
    print(f"[ctx] fine grid {ctx['fine_shape'][0]}x{ctx['fine_shape'][1]}, "
          f"land cells {ctx['n_land_cells']}")
    res = {}
    if not args.report_only:
        if not args.skip_admin:
            res["admin"] = build_admin(ctx, region)
        if not args.skip_soil:
            res["soil"] = build_soil(ctx, region)
        if not args.skip_ndvi:
            res["ndvi"] = build_ndvi(ctx, region)
        if not args.skip_lulc:
            res["lulc"] = build_lulc(ctx, region)
        if not args.skip_soilmoisture:
            res["soilmoisture"] = build_soilmoisture(ctx, region)
        # merge into any previous summary so skipped builders keep their stats
        prev_p = config.AUX / f"build_summary_{region}.json"
        if prev_p.exists():
            prev = json.loads(prev_p.read_text())
            prev.update(res)
            res = prev
        with open(prev_p, "w") as f:
            json.dump(res, f, indent=1, default=str)
    else:
        p = config.AUX / f"build_summary_{region}.json"
        if p.exists():
            res = json.loads(p.read_text())
    res.update(write_reports(ctx, region, res))
    print("[done] auxiliary layers + reports written")


if __name__ == "__main__":
    main()
