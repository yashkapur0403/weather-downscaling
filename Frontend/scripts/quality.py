"""
Automated data-quality checks, run inside preprocess.py before training.

Philosophy: fail loudly on scientifically important errors (date misalignment,
wrong latitude ordering, unit ambiguity, excessive NaNs, invalid DEM, broken
ERA5 aggregation). Do NOT silently repair those.
"""
from __future__ import annotations

import numpy as np


def check(condition: bool, msg: str, *, warn_only: bool = False) -> bool:
    """Raise ValueError (or warn) when `condition` is False."""
    if not condition:
        if warn_only:
            print(f"[quality WARNING] {msg}")
            return False
        raise ValueError(f"DATA QUALITY FAILURE: {msg}")
    return True


def check_dates(dates: list[str], source: str) -> None:
    check(len(dates) > 0, f"{source}: no dates extracted")
    check(len(set(dates)) == len(dates), f"{source}: duplicate timestamps found")
    check(all(len(d) == 10 and d[4] == "-" and d[7] == "-" for d in dates),
          f"{source}: dates not in YYYY-MM-DD format: {dates[:3]}")
    sd = sorted(dates)
    check(dates == sd or dates == sorted(dates, reverse=True),
          f"{source}: dates are not in a consistent order")


def check_coords(lat, lon, source: str, *, expect_lat_asc: bool = True,
                 expect_lon_asc: bool = True) -> None:
    lat, lon = np.asarray(lat), np.asarray(lon)
    check(len(lat) >= 2 and len(lon) >= 2, f"{source}: degenerate coordinate axes")
    check(bool(np.all(np.diff(lat) > 0)) if expect_lat_asc
          else bool(np.all(np.diff(lat) < 0)),
          f"{source}: latitude not strictly "
          f"{'ascending' if expect_lat_asc else 'descending'} "
          f"({lat[0]}..{lat[-1]})")
    check(bool(np.all(np.diff(lon) > 0)) if expect_lon_asc
          else bool(np.all(np.diff(lon) < 0)),
          f"{source}: longitude not strictly "
          f"{'ascending' if expect_lon_asc else 'descending'}")
    check(bool(np.max(lat) <= 90.5 and np.min(lat) >= -90.5),
          f"{source}: latitude out of physical range")
    check(bool(np.max(lon) <= 180.5 and np.min(lon) >= -180.5),
          f"{source}: longitude out of physical range")


def check_rain(rain: np.ndarray, source: str, *, max_mm: float = 2000.0) -> None:
    check(rain.dtype.kind == "f", f"{source}: rainfall is not a float array")
    fin = np.isfinite(rain)
    check(float(fin.mean()) > 0.5,
          f"{source}: only {fin.mean()*100:.1f}% finite values - data broken")
    vals = rain[fin]
    check(float(vals.min()) >= -1e-3,
          f"{source}: negative rainfall present (min={vals.min():.2f} mm) - "
          "check missing-value handling/units")
    check(float(vals.max()) <= max_mm,
          f"{source}: implausible rainfall max={vals.max():.1f} mm/day "
          "(check units: IMD/CHIRPS are mm/day)")
    nan_frac = float((~fin).mean())
    if nan_frac > 0.30:
        print(f"[quality WARNING] {source}: {nan_frac*100:.1f}% missing pixels "
              "(expected for land-only products near coasts)")


def check_dem(elev: np.ndarray) -> None:
    fin = np.isfinite(elev)
    check(float(fin.mean()) > 0.99, "DEM: too many non-finite values")
    vals = elev[fin]
    check(float(vals.min()) >= -120.0,
          f"DEM: invalid elevations below -120 m (min={vals.min():.1f} m) - "
          "tile decoding glitch or wrong source")
    check(float(vals.max()) <= 9000.0,
          f"DEM: invalid elevations above 9000 m (max={vals.max():.1f} m)")
    check(float(vals.std()) > 5.0,
          "DEM: near-constant elevation - DEM carries no information "
          "(check ROI/resampling)")


def check_era5(arrs: dict, dates: list[str]) -> None:
    """arrs: {name: (days, lat, lon) array}; dates: YYYY-MM-DD list."""
    for name, a in arrs.items():
        check(a.shape[0] == len(dates),
              f"ERA5 {name}: {a.shape[0]} days vs {len(dates)} target days")
        fin = np.isfinite(a)
        check(float(fin.mean()) > 0.95,
              f"ERA5 {name}: only {fin.mean()*100:.1f}% finite values")
        vals = a[fin]
        if "t2m" in name or "dewp" in name:
            check(float(vals.min()) > -60 and float(vals.max()) < 60,
                  f"ERA5 {name}: implausible temperature {vals.min():.1f}.."
                  f"{vals.max():.1f} C (daily aggregates broken?)")
        if name == "wind":
            check(float(vals.min()) >= 0 and float(vals.max()) < 60,
                  f"ERA5 wind: implausible speeds {vals.min():.1f}.."
                  f"{vals.max():.1f} km/h")
    # physical sanity: Tmax >= Tmean >= dewpoint (allow small rounding slack)
    if "t2m_max" in arrs and "t2m_mean" in arrs:
        diff = np.nanmean(arrs["t2m_max"] - arrs["t2m_mean"])
        check(diff > -0.5, f"ERA5: mean daily T2m_max - T2m_mean = {diff:.2f} C "
              "(should be positive; aggregation broken?)")
    if "t2m_mean" in arrs and "dewp" in arrs:
        diff = np.nanmean(arrs["t2m_mean"] - arrs["dewp"])
        check(diff > -0.5, f"ERA5: mean T2m_mean - dewpoint = {diff:.2f} C "
              "(dewpoint should not exceed temperature; aggregation broken?)")


def check_alignment(n_imd: int, n_chirps: int, n_common: int) -> None:
    check(n_common > 0, "IMD/CHIRPS: zero overlapping dates")
    check(n_common >= min(n_imd, n_chirps) * 0.90,
          f"IMD/CHIRPS: only {n_common} of min({n_imd},{n_chirps}) dates overlap "
          "- systematic misalignment suspected")
