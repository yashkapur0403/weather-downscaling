"""
Automated preprocessing: raw IMD + CHIRPS + DEM + ERA5 -> aligned training pairs.

Pipeline (Layer-1 spec v3, region-aware):
  A. IMD      : read per-year NetCDFs, select region ROI, concatenate, drop
                duplicate dates, derive the LAND MASK (IMD is land-only; sea
                cells are structural NaN and must never enter Y/M/metrics)
  B. CHIRPS   : select same ROI+dates (auto-detects monthly HDF5 or GEE
                GeoTIFFs); bilinear to the exact fine grid
  C. DEM      : sanitize, resample SRTM elevation to the working fine grid
  D. ERA5     : daily T2m/T2mmax/dewpoint (ERA5-Land) + wind (ERA5) sampled on
                a 0.5-deg lattice, bilinearly resampled to the fine grid
  E. ALIGN    : date intersection only (no invention); quality checks fail loudly
  F. PAIRS    : X = [imd_rain, dem, era5_t2m, era5_t2m_max, era5_dewp, era5_wind]
                Y = CHIRPS on the fine grid (5x sub-points per IMD cell),
                restricted to IMD-land cells; M = valid-pixel mask
  G. SPLIT    : strict year-based holdout (train years < val year < test year)
  H. NORM     : rain/elev scale + ERA5 standardization from TRAIN years only

Outputs (data/processed/):
  X_{train,val,test}.npy   (n, C, H*5, W*5) float32  [channel order = meta.channels]
  Y_{train,val,test}.npy   (n, 1, H*5, W*5) float32  (mm/day; 0 outside land)
  M_{train,val,test}.npy   (n, 1, H*5, W*5) float32  (1 = valid CHIRPS land pixel)
  meta.json                provenance, region, grid, land mask, dates, split,
                           normalization, notes
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from grids import aggregate_to_coarse, bilinear_upsample, fine_grid  # noqa: E402
from imd_reader import load_imd  # noqa: E402
from quality import (check, check_alignment, check_coords, check_dates,  # noqa: E402
                     check_dem, check_era5, check_rain)


# --------------------------------------------------------------------------
# CHIRPS ingestion (HDF5/NetCDF-4 monthly files or per-day GeoTIFFs from GEE)
# --------------------------------------------------------------------------
def load_chirps_h5(path: Path, roi: dict):
    """Read daily CHIRPS over the ROI from a monthly HDF5/NetCDF-4 file."""
    import h5py
    with h5py.File(path, "r") as f:
        datasets = {}
        f.visititems(lambda n, o: datasets.__setitem__(n, o)
                     if isinstance(o, h5py.Dataset) else None)
        rain_name = next((n for n in datasets
                          if "precip" in n.lower() or "rain" in n.lower()), None)
        lat_name = next((n for n in datasets
                         if n.lower() in ("latitude", "lat")), None)
        lon_name = next((n for n in datasets
                         if n.lower() in ("longitude", "lon")), None)
        time_name = next((n for n in datasets if "time" in n.lower()), None)
        if rain_name is None:
            raise ValueError(f"no precipitation variable in {path.name}: "
                             f"{list(datasets)}")
        lat = np.asarray(datasets[lat_name][:], dtype="float64")
        lon = np.asarray(datasets[lon_name][:], dtype="float64")
        tv = np.asarray(datasets[time_name][:], dtype="float64")
        tunits = datasets[time_name].attrs.get("units", b"")
        if isinstance(tunits, bytes):
            tunits = tunits.decode()
        import re
        from datetime import datetime, timedelta
        m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", str(tunits))
        if not m:
            raise ValueError(f"cannot parse CHIRPS time units {tunits!r}")
        origin = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        dates = [(origin + timedelta(days=float(t))).date() for t in tv]
        rd = datasets[rain_name]
        scale = float(rd.attrs.get("scale_factor", 1.0))
        offset = float(rd.attrs.get("add_offset", 0.0))
        fill = rd.attrs.get("_FillValue", None)
        if fill is not None:
            fill = float(np.ravel(fill)[0])

        # ROI slice padded by half an IMD cell (fine grid extends that far)
        pad = 0.125
        lat_sel = np.where((lat >= roi["lat_min"] - pad) &
                           (lat <= roi["lat_max"] + pad))[0]
        lon_sel = np.where((lon >= roi["lon_min"] - pad) &
                           (lon <= roi["lon_max"] + pad))[0]
        check(len(lat_sel) >= 10 and len(lon_sel) >= 10,
              f"{path.name}: ROI selects only {len(lat_sel)}x{len(lon_sel)} "
              "CHIRPS points")
        t_sel = [i for i, d in enumerate(dates)
                 if roi["start_date"] <= str(d) <= roi["end_date"]]
        if not t_sel:
            raise ValueError(f"{path.name} covers {dates[0]}..{dates[-1]}: "
                             f"requested window not inside")
        slabs = []
        for i in t_sel:
            s = rd[i, lat_sel[0]:lat_sel[-1] + 1, lon_sel[0]:lon_sel[-1] + 1]
            slabs.append(s.astype("float64") * scale + offset)
        rain = np.stack(slabs).astype("float32")
        if fill is not None:
            rain[np.isclose(rain, fill * scale + offset)] = np.nan
    return {"dates": [dates[i] for i in t_sel],
            "lat": lat[lat_sel], "lon": lon[lon_sel], "rain": rain,
            "source_file": str(path)}


def load_chirps_tifs(directory: Path):
    """Read per-day GeoTIFFs exported from GEE (CHIRPS_YYYY-MM-DD.tif)."""
    try:
        import tifffile
    except ImportError:
        raise ImportError("pip install tifffile  (needed to read GEE GeoTIFF exports)")
    files = sorted(directory.glob("CHIRPS_*.tif"))
    if not files:
        raise FileNotFoundError(f"no CHIRPS_*.tif in {directory}")
    dates, slabs = [], []
    for p in files:
        slabs.append(tifffile.imread(p).astype("float32"))
        dates.append(date.fromisoformat(p.stem.split("_")[1]))
    return {"dates": dates, "lat": None, "lon": None,
            "rain": np.stack(slabs), "source_file": str(directory)}


# --------------------------------------------------------------------------
# DEM ingestion (region npz from download_or_export.py or GeoTIFF from GEE)
# --------------------------------------------------------------------------
def load_dem(roi: dict, region: str | None = None):
    """Load the DEM mosaic for a region (falls back to the pilot file)."""
    tag = region or getattr(config, "REGION_DEFAULT", None)
    npz = (config.RAW_DEM / f"dem_roi_{tag}.npz") if tag else None
    legacy = config.RAW_DEM / "dem_roi.npz"
    tif = config.RAW_DEM / "dem_roi.tif"
    if npz is not None and npz.exists():
        z = np.load(npz)
        return z["lat"], z["lon"], z["elev"].astype("float32")
    if legacy.exists():
        z = np.load(legacy)
        return z["lat"], z["lon"], z["elev"].astype("float32")
    if tif.exists():
        import tifffile
        elev = tifffile.imread(tif).astype("float32")
        lat = np.linspace(roi["lat_max"], roi["lat_min"], elev.shape[0])
        lon = np.linspace(roi["lon_min"], roi["lon_max"], elev.shape[1])
        return lat, lon, elev
    raise FileNotFoundError(
        f"No DEM found. Expected {npz} (built by download_or_export.py for the "
        f"selected region), {legacy} (pilot ROI) or {tif} (GEE export).")


def sanitize_dem(dem_lat, dem_lon, dem_hi):
    if dem_lat[0] > dem_lat[-1]:  # ensure ascending
        dem_lat, dem_hi = dem_lat[::-1].copy(), dem_hi[::-1].copy()
    if dem_lon[0] > dem_lon[-1]:
        dem_lon, dem_hi = dem_lon[::-1].copy(), dem_hi[::-1].copy()
    bad = (dem_hi < -100) | (dem_hi > 8850)
    if bad.any():
        print(f"[dem] masking {int(bad.sum())} impossible pixels "
              f"({100 * bad.mean():.4f}% of the ROI)")
        dem_hi[bad] = np.nan
    return dem_lat, dem_lon, np.nan_to_num(dem_hi, nan=0.0).astype("float32")


def resample_to_grid(data_lat, data_lon, data, want_lat, want_lon):
    """Bilinear resample of a (H, W) lat/lon-ascending field to target centers.

    Weights clamped to [0, 1] -> constant (edge-value) extrapolation outside
    the source domain, never linear extrapolation.
    """
    def interp_axis(vals, q):
        if len(vals) < 2 or np.any(np.diff(vals) <= 0):
            raise ValueError("resample_to_grid needs strictly ascending coords")
        idx = np.searchsorted(vals, q) - 1
        idx = np.clip(idx, 0, len(vals) - 2)
        w = (q - vals[idx]) / (vals[idx + 1] - vals[idx])
        return idx, np.clip(w, 0.0, 1.0)

    iy, wy = interp_axis(data_lat, want_lat)
    ix, wx = interp_axis(data_lon, want_lon)
    a = data[np.ix_(iy, ix)]
    b = data[np.ix_(iy, ix + 1)]
    c = data[np.ix_(iy + 1, ix)]
    d = data[np.ix_(iy + 1, ix + 1)]
    wyy = wy[:, None]
    wxx = wx[None, :]
    return (a * (1 - wxx) + b * wxx) * (1 - wyy) + \
           (c * (1 - wxx) + d * wxx) * wyy


def _fill_nearest(field):
    """Fill NaNs with the nearest valid value (iterative 4-neighbour dilation).

    Used only for structurally-missing ERA5-Land sea points on the sampling
    margin - never for rainfall. Returns None if no valid value exists.
    """
    f = np.array(field, dtype="float32")
    for _ in range(max(f.shape) + 1):
        m = np.isnan(f)
        if not m.any():
            return f
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            src = np.roll(f, shift, axis=axis)
            valid = np.roll(~np.isnan(f), shift, axis=axis)
            fill = m & valid
            if fill.any():
                f[fill] = src[fill]
    return None if np.isnan(f).any() else f


# --------------------------------------------------------------------------
# ERA5-Land / ERA5 daily context (from era5_daily_<region>.npz, downloader)
# --------------------------------------------------------------------------
def load_era5(start: str, end: str, region: str | None = None):
    """Load daily ERA5 arrays covering [start, end].

    Returns {var: (days, nlat, nlon) float32} with var in
    {t2m_mean, t2m_max, dewp, wind}, plus dates/lat/lon. Values are daily
    aggregates over Asia/Kolkata days computed server-side by Open-Meteo,
    sampled on a 0.5-deg lattice (see download_or_export.era5_sample_points).
    """
    tag = region or getattr(config, "REGION_DEFAULT", None)
    path = (config.RAW_ERA5 / f"era5_daily_{tag}.npz") if tag else None
    legacy = config.RAW_ERA5 / "era5_daily.npz"
    if path is None or not path.exists():
        path = legacy if legacy.exists() else path
    if path is None or not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    dates = [str(d) for d in z["dates"]]
    want = [d for d in dates if start <= d <= end]
    if not want:
        print(f"[era5] {path} covers {dates[0]}..{dates[-1]}: "
              f"window {start}..{end} not inside -> ignoring ERA5")
        return None
    sel = [dates.index(d) for d in want]
    out = {"t2m_mean": z["t2m_mean"][sel], "t2m_max": z["t2m_max"][sel],
           "dewp": z["dewp"][sel], "wind": z["wind"][sel]}
    return {"dates": want, "lat": z["lat"], "lon": z["lon"], **out}


def imd_land_mask(imd_rain: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """Boolean (H, W) mask of IMD cells that are land.

    IMD 0.25-deg gridded rainfall is land-only: sea cells are missing on every
    day (structural NaN). A cell counts as land when at least `thresh` of all
    days have a valid value. The mask restricts Y/M, the per-day quality
    filter and every downstream metric, so ocean cells can never silently
    enter training targets or scores.
    """
    frac = np.isfinite(imd_rain).mean(axis=0)
    return frac >= thresh


# --------------------------------------------------------------------------
def parse_split_years(s: str | None):
    if not s:
        return config.SPLIT_YEARS
    tr, va, te = s.split("|")
    return {"train": [int(y) for y in tr.split(",") if y],
            "val": [int(y) for y in va.split(",") if y],
            "test": [int(y) for y in te.split(",") if y]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default=None,
                    help="study region from config.REGIONS (default: "
                         "config.REGION_DEFAULT); individual --lat/--lon "
                         "flags override it")
    ap.add_argument("--lat-min", type=float, default=None)
    ap.add_argument("--lat-max", type=float, default=None)
    ap.add_argument("--lon-min", type=float, default=None)
    ap.add_argument("--lon-max", type=float, default=None)
    ap.add_argument("--start", default=config.START_DATE_DEFAULT)
    ap.add_argument("--end", default=config.END_DATE_DEFAULT)
    ap.add_argument("--split-years", default=None,
                    help="train|val|test year lists, e.g. '2018,2019,2020|2021|2022'")
    ap.add_argument("--split", default=None,
                    help="legacy fraction split 0.7,0.15,0.15 (overrides years)")
    ap.add_argument("--era5-mode", default="strict", choices=["strict", "optional"],
                    help="strict: fail if any aligned date lacks ERA5; "
                         "optional: drop dates lacking ERA5 (ablation fallback)")
    args = ap.parse_args()
    region = args.region or config.REGION_DEFAULT
    roi = config.clip_roi(vars(args))
    roi = {**roi, "start_date": args.start, "end_date": args.end}
    split_years = parse_split_years(args.split_years)

    print(f"Region: {region}")
    print(f"ROI: {roi}")
    print(f"Split years: {split_years}")

    # ---------------- A. IMD (multi-year) ----------------
    imd_files = sorted(config.RAW_IMD.glob("*.nc"))
    check(len(imd_files) > 0, f"no IMD NetCDF in {config.RAW_IMD} - "
          "run scripts/download_or_export.py")
    imd = None
    for p in imd_files:
        try:
            d = load_imd(p, roi)
            check_rain(d["rain"], f"IMD {p.name}")
            print(f"[imd] {p.name}: {len(d['dates'])} days, grid {d['rain'].shape[1:]}")
            if imd is None:
                imd = d
            else:
                imd["dates"] = imd["dates"] + d["dates"]
                imd["rain"] = np.concatenate([imd["rain"], d["rain"]])
        except ValueError as e:
            print(f"[imd] skipping {p.name}: {e}")
    check(imd is not None and len(imd["dates"]) > 0,
          "IMD files do not cover the requested window")
    # de-duplicate + sort (defensive; downloader should never produce dupes)
    uniq = {}
    for i, d in enumerate(imd["dates"]):
        uniq[str(d)] = i
    if len(uniq) != len(imd["dates"]):
        print(f"[imd] dropped {len(imd['dates']) - len(uniq)} duplicate dates")
    order = sorted(uniq, key=lambda s: uniq[s])
    dates_sorted = sorted(uniq)
    imd_r = np.stack([imd["rain"][uniq[d]] for d in dates_sorted])
    imd = {"dates": [date.fromisoformat(d) for d in dates_sorted],
           "lat": imd["lat"], "lon": imd["lon"], "rain": imd_r,
           "source_file": [str(p) for p in imd_files]}
    check_dates([str(d) for d in imd["dates"]], "IMD")
    check_coords(imd["lat"], imd["lon"], "IMD")

    # ---- LAND MASK (IMD is land-only; sea cells are structural NaN) ----
    land = imd_land_mask(imd["rain"])
    print(f"[land] IMD land cells: {int(land.sum())}/{land.size} "
          f"({100 * land.mean():.1f}% of the box); "
          f"{int((~land).sum())} sea/edge cells excluded from Y/M/metrics")
    check(land.sum() >= 100, "land mask too small - check the ROI (all sea?)")

    # ---------------- B. CHIRPS ----------------
    chirps = None
    try:
        parts = []
        for p in sorted(config.RAW_CHIRPS.glob("*.nc")):
            try:
                parts.append(load_chirps_h5(p, roi))
            except ValueError as e:
                # e.g. a monthly file that does not overlap the requested window
                print(f"[chirps] skipping {p.name}: {e}")
        if parts:
            chirps = {"dates": sum((p["dates"] for p in parts), []),
                      "lat": parts[0]["lat"], "lon": parts[0]["lon"],
                      "rain": np.concatenate([p["rain"] for p in parts]),
                      "source_file": [p["source_file"] for p in parts]}
            print(f"[chirps] {len(chirps['dates'])} daily files from "
                  f"{len(parts)} monthly HDF5 files")
    except Exception as e:
        print(f"[chirps] HDF5 path failed ({type(e).__name__}: {e}); "
              "trying GEE GeoTIFF exports ...")
    if chirps is None:
        chirps = load_chirps_tifs(config.RAW_CHIRPS)
    if chirps["lat"] is not None:
        check_coords(chirps["lat"], chirps["lon"], "CHIRPS")
    check_rain(chirps["rain"], "CHIRPS")
    print(f"[chirps] rain min={np.nanmin(chirps['rain']):.1f} "
          f"max={np.nanmax(chirps['rain']):.1f} mm, grid {chirps['rain'].shape[1:]}")

    # ---------------- E. ALIGN (date intersection) ----------------
    imd_dates = [str(d) for d in imd["dates"]]
    ch_dates = [str(d) for d in chirps["dates"]]
    common = np.array(sorted(set(imd_dates) & set(ch_dates)), dtype="U10")
    dropped = sorted(set(imd_dates) ^ set(ch_dates))
    if dropped:
        print(f"[align] {len(dropped)} dates missing from one source -> dropped "
              f"(first: {dropped[:5]})")
    check_alignment(len(imd_dates), len(ch_dates), len(common))
    imd_r = np.stack([imd["rain"][imd_dates.index(d)] for d in common])
    ch_r = np.stack([chirps["rain"][ch_dates.index(d)] for d in common])

    # ---------------- C. DEM ----------------
    dem_lat, dem_lon, dem_hi = sanitize_dem(*load_dem(roi, region))
    check_dem(dem_hi)

    # ---------------- grids + channels ----------------
    fine_lat, fine_lon = fine_grid(imd["lat"], imd["lon"], config.FINE_SUB)
    check_coords(fine_lat, fine_lon, "fine grid")

    # dem on fine grid (documented: bilinear from the SRTM mosaic, weights
    # clamped); DEM is real land elevation everywhere incl. the sea margin
    dem_fine = resample_to_grid(dem_lat, dem_lon, dem_hi,
                                fine_lat, fine_lon).astype("float32")

    # CHIRPS -> exact fine grid (documented: bilinear, half-cell stagger noted)
    ch_fine = np.stack([resample_to_grid(chirps["lat"], chirps["lon"], ch_r[i],
                                         fine_lat, fine_lon)
                        for i in range(len(common))]).astype("float32")

    # IMD -> fine grid (bilinear; also channel 0 and the bilinear baseline)
    imd_fine = np.stack([bilinear_upsample(imd_r[i], config.FINE_SUB)
                         for i in range(len(common))])

    # land_fine[i, j] = land[i // 5, j // 5]: the fine grid tiles each coarse
    # cell exactly, so the coarse land mask maps 1:1 onto fine pixels.
    land_fine = np.repeat(np.repeat(land, config.FINE_SUB, axis=0),
                          config.FINE_SUB, axis=1)

    # TARGET validity: a fine pixel enters Y/M iff it is IMD-land AND CHIRPS
    # provides a value on EVERY aligned day. Coastal pixels within one CHIRPS
    # cell of CHIRPS-ocean are structurally missing (bilinear stencil touches
    # sea NaN) - they are excluded, never imputed, so Y holds only real
    # reference values. Sea fine cells are excluded for the same reason as in
    # the pilot: IMD has no value there, so the coarse input would be a
    # filled 0.
    y_valid = ~(np.isnan(ch_fine).any(axis=0)) & land_fine
    n_excl = int((land_fine & ~y_valid).sum())
    print(f"[land] target-valid fine pixels: {int(y_valid.sum())}/"
          f"{int(land_fine.sum())} land pixels "
          f"({100 * n_excl / max(land_fine.sum(), 1):.1f}% coastal/sea strip "
          "excluded from Y/M)")
    check(int(y_valid.sum()) >= 5000,
          f"only {int(y_valid.sum())} target-valid pixels - ROI mis-set?")
    ch_fine = np.where(y_valid[None, :, :], ch_fine, np.nan)

    # per-day quality filter: IMD missingness on all land pixels, CHIRPS
    # missingness on target-valid pixels (0 by construction there)
    def bad_frac(arr, mask):
        a = arr[:, mask]
        return np.isnan(a).reshape(len(a), -1).mean(axis=1)

    nan_imd_day = bad_frac(imd_fine, land_fine)
    nan_ch_day = bad_frac(ch_fine, y_valid)
    keep = (nan_imd_day <= config.MISSING_TOL) & (nan_ch_day <= config.MISSING_TOL)
    if int((~keep).sum()):
        print(f"[quality] dropping {int((~keep).sum())}/{len(common)} samples "
              f"with >{config.MISSING_TOL * 100:.0f}% missing land pixels")
    dates_ok = [common[i] for i in range(len(common)) if keep[i]]
    imd_fine, ch_fine = imd_fine[keep], ch_fine[keep]
    check(len(dates_ok) >= 30, f"only {len(dates_ok)} valid samples after the "
          "quality filter; widen the date range/ROI")

    # remaining NaNs in the INPUT channel are structural (sea/edge bleed):
    # impute 0 mm, exactly like the pilot pipeline, and document it
    n_imputed = int(np.isnan(imd_fine).sum())
    if n_imputed:
        print(f"[warn] {n_imputed / imd_fine.size * 100:.2f}% of upsampled IMD "
              "pixels missing (ocean/edge cells) -> imputed 0 mm "
              "(Y/M/metrics never use those cells)")
    imd_fine = np.nan_to_num(imd_fine, nan=0.0)

    # ---------------- D. ERA5 channels ----------------
    era5 = load_era5(args.start, args.end, region)
    era5_fine = {}
    common_e = common[keep]
    if era5 is not None:
        e_dates = era5["dates"]
        have = {d: i for i, d in enumerate(e_dates)}
        missing_era5 = [d for d in common_e if d not in have]
        if args.era5_mode == "optional":
            # ablation mode: keep dates with ERA5, drop the rest (documented;
            # year-based split integrity is preserved, only n changes)
            common_e = [d for d in common_e if d in have]
            print(f"[era5] optional mode: {len(missing_era5)} dates lack ERA5 "
                  "-> excluded from the ERA5-stack dataset")
        else:
            check(len(missing_era5) == 0,
                  f"ERA5 missing {len(missing_era5)} of the aligned dates "
                  f"(e.g. {missing_era5[:3]}) - re-run download_or_export.py "
                  "with the same start/end, or pass --era5-mode optional")
        # ERA5 is sampled on a 0.5-deg lattice (+margin); bilinearly resample
        # each daily field directly onto the fine grid (clamped weights ->
        # constant extrapolation at the edges)
        check(era5["lat"][0] < era5["lat"][-1] and
              era5["lon"][0] < era5["lon"][-1],
              "ERA5 sampling grid not strictly ascending")
        filled = {}
        for var in ("t2m_mean", "t2m_max", "dewp", "wind"):
            daily = era5[var][[have[d] for d in common_e]].copy()  # (n,nlat,nlon)
            nan_frac = float(np.isnan(daily).mean())
            day_ok = np.isfinite(daily).any(axis=(1, 2))
            if nan_frac > 0.5 or not bool(day_ok.all()):
                # incomplete year cache (e.g. a model's request quota-blocked
                # in some year): unusable for training -> skip the channel,
                # never impute it (era5_wind is not in CHANNELS_ALL anyway)
                print(f"[era5] WARNING: {var} is {nan_frac * 100:.0f}% missing "
                      f"with {int((~day_ok).sum())} fully-missing days "
                      "(incomplete year cache) -> channel skipped")
                continue
            # ERA5-Land is land-only: sea points on the sampling margin are
            # structurally missing. Documented fill = nearest valid value per
            # day, so interior gradients are untouched.
            for i in range(len(daily)):
                f = daily[i]
                if np.isnan(f).any():
                    f_filled = _fill_nearest(f)
                    check(f_filled is not None and bool(np.isfinite(f_filled).all()),
                          f"ERA5 {var} {common_e[i]}: sea-fill failed - "
                          "field has no valid values on the ROI")
                    daily[i] = f_filled
            era5_fine[var] = np.stack(
                [resample_to_grid(era5["lat"], era5["lon"], daily[i],
                                  fine_lat, fine_lon)
                 for i in range(len(common_e))]).astype("float32")
            filled[var] = daily
        # quality checks on the fields that actually enter the pipeline
        # (post sea-fill; the raw sea NaNs are structural and documented)
        check_era5(filled, list(common_e))
        print(f"[era5] channels ready on fine grid: {list(era5_fine)}")
    else:
        print("[era5] not available -> building 2-channel dataset "
              "(imd_rain, dem) fallback")

    if era5_fine and len(common_e) != len(dates_ok):
        # ERA5 optional mode dropped some dates -> restrict the whole dataset
        check(len(common_e) >= 30, f"only {len(common_e)} dates have ERA5; "
              "too few for an ERA5-stack ablation - widen the window")
        cmask = np.array([d in set(common_e) for d in dates_ok])
        dates_ok = [dates_ok[i] for i in range(len(dates_ok)) if cmask[i]]
        imd_fine, ch_fine = imd_fine[cmask], ch_fine[cmask]

    # ---------------- coherence check ----------------
    ch_coarse = aggregate_to_coarse(ch_fine, config.FINE_SUB)
    imd_coarse = aggregate_to_coarse(imd_fine, config.FINE_SUB)
    a, b = imd_coarse.ravel(), ch_coarse.ravel()
    okm = np.isfinite(a) & np.isfinite(b)
    corr = np.corrcoef(a[okm], b[okm])[0, 1] if okm.sum() > 10 else float("nan")
    print(f"[check] IMD vs CHIRPS (coarse-mean, land cells, all days): "
          f"corr={corr:.3f} mean IMD={np.nanmean(imd_coarse):.2f} "
          f"mean CHIRPS={np.nanmean(ch_coarse):.2f}")

    # ---------------- G. strict year-based split ----------------
    if args.split:
        fr = dict(zip(("train", "val", "test"),
                      (float(x) for x in args.split.split(","))))
        n = len(dates_ok)
        n_tr, n_va = int(n * fr["train"]), int(n * fr["val"])
        idx = {"train": np.arange(0, n_tr), "val": np.arange(n_tr, n_tr + n_va),
               "test": np.arange(n_tr + n_va, n)}
        split_desc = f"fractions {fr} (legacy)"
    else:
        idx = {k: np.array([i for i, d in enumerate(dates_ok)
                            if int(d[:4]) in ys])
               for k, ys in split_years.items()}
        split_desc = f"years {split_years}"
        for k in ("train", "val", "test"):
            check(len(idx[k]) > 0, f"split '{k}' has no dates; years="
                  f"{split_years[k]} vs data {dates_ok[0]}..{dates_ok[-1]}")
        # strict ordering: max(train) < min(val) < min(test) - no leakage
        check(max(dates_ok[i] for i in idx["train"]) <
              min(dates_ok[i] for i in idx["val"]),
              "train dates overlap/after val dates - temporal leakage")
        check(max(dates_ok[i] for i in idx["val"]) <
              min(dates_ok[i] for i in idx["test"]),
              "val dates overlap/after test dates - temporal leakage")
    for k in ("train", "val", "test"):
        print(f"[split:{k}] {len(idx[k])} days "
              f"({dates_ok[idx[k][0]]} .. {dates_ok[idx[k][-1]]})")

    # ---------------- H. normalization (TRAIN years only) ----------------
    tr = idx["train"]
    rain_scale = float(np.nanpercentile(imd_fine[tr], 99.9))
    rain_scale = float(np.clip(rain_scale, 20.0, config.RAIN_MAX))
    elev_scale = float(np.clip(float(np.nanpercentile(dem_fine, 99.9)),
                               500.0, config.ELEV_MAX))
    era5_stats = {}
    for var, arr in era5_fine.items():
        mu = float(np.nanmean(arr[tr]))
        sd = float(np.nanstd(arr[tr]))
        check(sd > 1e-3, f"ERA5 {var}: degenerate std ({sd}) in training years")
        era5_stats[var] = {"mean": mu, "std": sd}
    print(f"[norm] rain_scale={rain_scale:.1f} mm  elev_scale={elev_scale:.1f} m "
          f"era5_vars={list(era5_stats)}")

    # ---------------- assemble channel stack ----------------
    chans = []
    chan_names = []
    for name in config.CHANNELS_ALL:
        if name == "imd_rain":
            chans.append(imd_fine / rain_scale)
            chan_names.append(name)
        elif name == "dem":
            chans.append(np.broadcast_to(dem_fine / elev_scale, imd_fine.shape))
            chan_names.append(name)
        elif name.startswith("era5_"):
            var = name.replace("era5_", "", 1)
            key = {"t2m": "t2m_mean"}.get(var, var)
            if key not in era5_fine:
                continue
            st = era5_stats[key]
            chans.append((era5_fine[key] - st["mean"]) / st["std"])
            chan_names.append(name)
        else:
            raise ValueError(f"unknown channel {name}")
    X_all = np.stack(chans, axis=1).astype("float32")
    X_all = np.nan_to_num(X_all, nan=0.0)
    check(bool(np.isfinite(X_all).all()), "non-finite values in final X stack")
    Y_all = np.nan_to_num(ch_fine, nan=0.0).astype("float32")[:, None]
    M_all = (~np.isnan(ch_fine)).astype("float32")[:, None]

    for k in ("train", "val", "test"):
        np.save(config.PROCESSED / f"X_{k}.npy", X_all[idx[k]])
        np.save(config.PROCESSED / f"Y_{k}.npy", Y_all[idx[k]])
        np.save(config.PROCESSED / f"M_{k}.npy", M_all[idx[k]])

    # ---------------- provenance ----------------
    era5_path = (config.RAW_ERA5 / f"era5_daily_{region}.npz")
    if not era5_path.exists():
        era5_path = config.RAW_ERA5 / "era5_daily.npz"
    meta = {
        "region": region,
        "roi": {k: roi[k] for k in ("lat_min", "lat_max", "lon_min", "lon_max")},
        "start_date": args.start, "end_date": args.end,
        "dates": dates_ok,
        "split": {k: {"indices": [int(idx[k][0]), int(idx[k][-1])],
                      "n_days": int(len(idx[k])),
                      "dates": [dates_ok[i] for i in idx[k]]}
                  for k in ("train", "val", "test")},
        "split_description": split_desc,
        "channels": chan_names,
        "channel_semantics": "X = stacked input channels on the fine grid "
                             "(imd_rain = bilinear-upsampled IMD, missing->0; "
                             "dem = resampled elevation; era5_* = daily "
                             "aggregates, sea-filled). Y = CHIRPS daily "
                             "rainfall on the same fine grid restricted to "
                             "IMD-land cells; M = 1 where Y is a valid land "
                             "pixel.",
        "land_mask": {
            "method": "IMD 0.25-deg cell is land iff >=50% of all aligned "
                      "days have a valid (non-missing) rainfall value",
            "coarse_shape": [int(land.shape[0]), int(land.shape[1])],
            "n_land_cells": int(land.sum()),
            "n_sea_cells": int((~land).sum()),
            "land_fraction": round(float(land.mean()), 4),
            "n_target_valid_fine_cells": int(y_valid.sum()),
            "target_validity": "fine pixel enters Y/M iff IMD-land AND CHIRPS "
                               "valid on every aligned day (coastal strip "
                               "within one CHIRPS cell of CHIRPS-ocean is "
                               "structurally missing -> excluded, not imputed)",
            "applied_to": ["Y", "M", "per-day quality filter",
                           "coherence corr", "all downstream metrics"],
            "note": "IMD is land-only; sea cells are structural NaN. X "
                    "channels keep filled values over sea (IMD->0, DEM real, "
                    "ERA5 sea-filled) so convolutions stay finite, but no "
                    "sample day, target pixel or metric ever uses sea or "
                    "non-valid coastal cells.",
        },
        "grid": {"imd_lat": imd["lat"].tolist(), "imd_lon": imd["lon"].tolist(),
                 "fine_lat": fine_lat.tolist(), "fine_lon": fine_lon.tolist(),
                 "fine_sub": config.FINE_SUB,
                 "H_fine": int(imd_fine.shape[1]), "W_fine": int(imd_fine.shape[2]),
                 "note": "fine grid = 5x5 sub-cell centers of the IMD grid "
                         "(area-tiling, 0.05 deg); fine extent = ROI +/- 0.125 deg"},
        "normalization": {"rain_scale_mm": rain_scale,
                          "elev_scale_m": elev_scale,
                          "era5_standardization": era5_stats,
                          "computed_from": "train-split years only"},
        "resampling_operations": {
            "imd_to_fine": "bilinear from 0.25-deg cell centers to 5x5 "
                           "sub-cell centers (grids.bilinear_upsample)",
            "chirps_to_fine": "bilinear from native 0.05-deg lattice; lattice is "
                              "staggered half a cell vs our area-tiling grid, so "
                              "Y is a half-cell bilinear sample of CHIRPS "
                              "(identical for baseline and model -> fair)",
            "dem_to_fine": "bilinear from the SRTM terrarium mosaic (zoom 11 "
                           "pilot / zoom 10 deccan, ~76 m/px); sea/glitch "
                           "pixels sanitized to 0 m",
            "era5_to_fine": "sampled on a 0.5-deg lattice at download, then "
                            "bilinear resample onto the fine grid "
                            "(clamped-edge weights)",
        },
        "era5_aggregation": "hourly ERA5-Land aggregated to daily mean/max over "
                            "Asia/Kolkata days server-side (Open-Meteo archive "
                            "API, models=era5_land; wind from models=era5 "
                            "because ERA5-Land has no 10m wind)",
        "provenance": {
            "imd": {"files": imd.get("source_file"),
                    "url": "https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html",
                    "units": "mm/day", "missing_value": -999.0},
            "chirps": {"files": chirps.get("source_file"),
                       "url": "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/by_month/",
                       "units": "mm/day", "version": "v2.0"},
            "dem": {"file": str(config.RAW_DEM / f"dem_roi_{region}.npz"),
                    "source": "SRTM 30 m via AWS Terrarium tiles (Mapzen/Nextzen)"},
            "era5": {"file": str(era5_path),
                     "api": config.ERA5_API,
                     "variables": list(era5_stats)},
        },
        "imputation": "remaining NaN IMD input pixels set to 0 (sea/edge bleed; "
                      "counted in meta.land_mask notes); missing CHIRPS land "
                      "pixels set to 0 in Y with M=0 marking them invalid for "
                      "masked loss/metrics; Y/M are 0/0 over sea by design",
        "target_note": "Y is CHIRPS bilinearly resampled onto the exact 5x "
                       "IMD-area fine grid, restricted to IMD-land cells. "
                       "CHIRPS is a REFERENCE product, not ground truth.",
        "imd_chirps_coarse_corr": float(corr),
    }
    with open(config.PROCESSED / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[done] region={region} channels={chan_names}")
    for k in ("train", "val", "test"):
        print(f"       {k}: X {X_all[idx[k]].shape}  Y {Y_all[idx[k]].shape}")
    print(f"       wrote X/Y/M_*.npy + meta.json to {config.PROCESSED}")


if __name__ == "__main__":
    main()
