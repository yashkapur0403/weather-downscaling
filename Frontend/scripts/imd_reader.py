"""
Reader for IMD 0.25-deg daily gridded rainfall NetCDF (classic NetCDF-3).

Auto-detects coordinate/variable names (per project rule: never hard-code
assumptions). Returns rainfall as float32 with missing values -> NaN.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

import numpy as np

from netcdf3 import (Nc3File, read_fixed_subset, read_record_subset,
                     read_record_var_1d)

COORD_HINTS = {
    "lat": ("latitude", "lat", "y", "latitude0", "nav_lat"),
    "lon": ("longitude", "lon", "x", "longitude0", "nav_lon"),
    "time": ("time", "t", "date", "julian", "TIME"),
}
RAIN_HINTS = ("rainfall", "precip", "rain", "pr", "apcp", "rf")
FILL_VALUES = (-999.0, -9999.0, -99.9, -99.0, 9.96921e36)


def _detect(candidates: list[str], hints: tuple[str, ...], default: str | None = None):
    """Find the variable whose name best matches the hints (substring match)."""
    names = {n.lower(): n for n in candidates}
    for h in hints:
        for low, orig in names.items():
            if h in low:
                return orig
    return default


def detect_imd_structure(nc: Nc3File) -> dict:
    lat_name = _detect(list(nc.variables), COORD_HINTS["lat"])
    lon_name = _detect(list(nc.variables), COORD_HINTS["lon"])
    time_name = _detect(list(nc.variables), COORD_HINTS["time"])
    rain_name = _detect([n for n, v in nc.variables.items() if len(v.dims) == 3],
                        RAIN_HINTS)
    if not all([lat_name, lon_name, time_name, rain_name]):
        raise ValueError(
            f"Could not auto-detect variables. Present: {list(nc.variables)}")
    rain = nc.variables[rain_name]
    fill = None
    for key in ("missing_value", "_FillValue", "missing"):
        if key in rain.attrs:
            fill = float(np.ravel(rain.attrs[key])[0])
            break
    units = rain.attrs.get("units", "mm")
    return {"lat": lat_name, "lon": lon_name, "time": time_name,
            "rain": rain_name, "fill": fill, "units": units,
            "lat_size": nc.dims[lat_name].size, "lon_size": nc.dims[lon_name].size,
            "time_size": nc.numrecs}


def load_imd(path, roi: dict):
    """
    Load IMD daily rainfall for the ROI.

    Returns dict with:
      dates     (n,) list of datetime.date
      lat       (nlat,) ascending
      lon       (nlon,) ascending
      rain      (n, nlat, nlon) float32, mm/day, missing -> NaN
    """
    with open(path, "rb") as f:
        nc = Nc3File(f)

        def fetch(a, b):
            f.seek(a)
            return f.read(b - a)

        info = detect_imd_structure(nc)
        lat = read_fixed_subset(nc, info["lat"], 0, info["lat_size"], fetch)
        lon = read_fixed_subset(nc, info["lon"], 0, info["lon_size"], fetch)

        # time -> dates (TIME is a 1-D record variable in IMD files)
        time_name = info["time"]
        tvar = nc.variables[time_name]
        if tvar.is_record:
            tvals_arr = read_record_var_1d(nc, time_name, 0, nc.numrecs, fetch)
        else:
            tvals_arr = read_fixed_subset(nc, time_name, 0, tvar._sizes[0], fetch)
        units = str(nc.variables[time_name].attrs.get("units", "days since 1900-12-31"))
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", units)
        if not m:
            raise ValueError(f"Cannot parse time units: {units!r}")
        origin = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        dates = [(origin + timedelta(days=float(tv))).date() for tv in tvals_arr]

        # ROI slicing
        lat0, lat1 = float(roi["lat_min"]), float(roi["lat_max"])
        lon0, lon1 = float(roi["lon_min"]), float(roi["lon_max"])
        lat_sel = np.where((lat >= lat0) & (lat <= lat1))[0]
        lon_sel = np.where((lon >= lon0) & (lon <= lon1))[0]
        if len(lat_sel) < 2 or len(lon_sel) < 2:
            raise ValueError(
                f"ROI {roi} selects fewer than 2 IMD grid points "
                f"({len(lat_sel)} lat x {len(lon_sel)} lon). "
                "Increase the ROI or check the bounds.")

        # restrict to the requested date window
        want0 = datetime.strptime(str(roi["start_date"]), "%Y-%m-%d").date()
        want1 = datetime.strptime(str(roi["end_date"]), "%Y-%m-%d").date()
        rain_dates = [(origin + timedelta(days=float(tv))).date() for tv in tvals_arr]
        idx = [i for i, d in enumerate(rain_dates) if want0 <= d <= want1]
        if not idx:
            raise ValueError(
                f"IMD file {path.name} covers {rain_dates[0]} .. {rain_dates[-1]}; "
                f"requested window not inside. Download the matching year(s).")
        slab = read_record_subset(nc, info["rain"], (idx[0], idx[-1] + 1),
                                  (int(lat_sel[0]), int(lat_sel[-1]) + 1),
                                  (int(lon_sel[0]), int(lon_sel[-1]) + 1), fetch)

    rain = slab.astype("float32")
    rain[rain == info["fill"]] = np.nan
    for fv in FILL_VALUES:
        rain[rain == fv] = np.nan
    rain = np.clip(rain, 0.0, None)

    if not idx:
        raise ValueError(
            f"IMD file {path.name} covers {rain_dates[0]} .. {rain_dates[-1]}; "
            f"requested window {want0}..{want1} not inside. "
            "Download the matching year(s) via download_or_export.py.")

    return {
        "dates": [rain_dates[i] for i in idx],
        "lat": lat[lat_sel].astype("float64"),
        "lon": lon[lon_sel].astype("float64"),
        "rain": rain,
        "info": info,
        "source_file": str(path),
    }
