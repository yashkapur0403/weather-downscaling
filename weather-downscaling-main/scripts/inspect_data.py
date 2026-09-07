"""
Inspect every raw data source and print its structure.

Per the project rule, nothing is hard-coded: IMD variables/coordinates are
auto-detected, and CHIRPS structure is read from the file itself.
Run:  python scripts/inspect_data.py [--lat-min ... --start ... --end ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from imd_reader import detect_imd_structure, load_imd  # noqa: E402
from netcdf3 import Nc3File  # noqa: E402


def describe_nc_header(path: Path) -> dict:
    """Read just the header (first 64 KB) of any NetCDF/HDF5 file."""
    with open(path, "rb") as f:
        magic = f.read(8)
    kind = ("NetCDF-3" if magic[:3] == b"CDF"
            else "HDF5/NetCDF-4" if magic[:4] == b"\x89HDF"
            else "unknown")
    out = {"file": path.name, "kind": kind, "size_mb": path.stat().st_size / 1e6}
    if kind == "NetCDF-3":
        with open(path, "rb") as f:
            nc = Nc3File(f)
            out["dims"] = {k: v.size for k, v in nc.dims.items()}
            out["vars"] = {n: {"dims": v.dims, "dtype": v.dtype,
                               "attrs": dict(v.attrs)}
                           for n, v in nc.variables.items()}
            out["gattrs"] = dict(nc.attrs)
    else:
        out["note"] = ("HDF5 file - read with h5py in preprocess.py; "
                       "structure is auto-detected there")
    return out


def describe_chirps_h5(path: Path) -> dict:
    import h5py
    with h5py.File(path, "r") as f:
        out = {"file": path.name, "kind": "HDF5/NetCDF-4",
               "size_mb": path.stat().st_size / 1e6,
               "vars": {}, "gattrs": {k: v for k, v in f.attrs.items()}}
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out["vars"][name] = {"shape": obj.shape, "dtype": str(obj.dtype),
                                     "attrs": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                               for k, v in obj.attrs.items()}}
        f.visititems(visit)
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lat-min", type=float, default=None)
    ap.add_argument("--lat-max", type=float, default=None)
    ap.add_argument("--lon-min", type=float, default=None)
    ap.add_argument("--lon-max", type=float, default=None)
    ap.add_argument("--start", default=config.START_DATE_DEFAULT)
    ap.add_argument("--end", default=config.END_DATE_DEFAULT)
    args = ap.parse_args()
    roi = config.clip_roi(vars(args))
    print(f"ROI: {roi}   period: {args.start} .. {args.end}\n")

    # ---------------- IMD ----------------
    imd_files = sorted(config.RAW_IMD.glob("*.nc"))
    if not imd_files:
        print(f"!! No IMD files in {config.RAW_IMD}. Run download_or_export.py")
    for p in imd_files:
        info = describe_nc_header(p)
        print("=" * 70)
        print(f"IMD: {p}")
        for k, v in info.items():
            if k == "vars":
                for vn, vd in v.items():
                    print(f"  var {vn}: dims={vd['dims']} dtype={vd['dtype']}")
                    for ak, av in vd["attrs"].items():
                        print(f"      {ak} = {av}")
            elif k == "dims":
                print(f"  dims: {v}")
            elif k == "gattrs":
                print(f"  global attrs: {v}")
            else:
                print(f"  {k}: {v}")
        if info["kind"] == "NetCDF-3":
            try:
                with open(p, "rb") as f:
                    nc = Nc3File(f)
                    det = detect_imd_structure(nc)
                print(f"  AUTO-DETECTED: rain={det['rain']} lat={det['lat']} "
                      f"lon={det['lon']} time={det['time']} "
                      f"fill={det['fill']} units={det['units']}")
                # quick smoke: load first 3 days over the ROI
                d = load_imd(p, {**roi, "start_date": args.start, "end_date": args.end})
                r = d["rain"]
                print(f"  ROI slice: {r.shape[0]} days, lat {d['lat'][0]:.2f}..{d['lat'][-1]:.2f} "
                      f"({len(d['lat'])}), lon {d['lon'][0]:.2f}..{d['lon'][-1]:.2f} ({len(d['lon'])})")
                print(f"  rainfall: min={np.nanmin(r):.1f} max={np.nanmax(r):.1f} "
                      f"mean={np.nanmean(r):.2f} mm  nan_frac={np.isnan(r).mean():.3f}")
                print(f"  first dates: {[str(x) for x in d['dates'][:3]]} ... last: {d['dates'][-1]}")
            except Exception as e:
                print(f"  ! load test failed: {type(e).__name__}: {e}")

    # ---------------- CHIRPS ----------------
    print()
    chirps_files = sorted(config.RAW_CHIRPS.glob("*.nc")) + \
        sorted(config.RAW_CHIRPS.glob("*.tif"))
    if not chirps_files:
        print(f"!! No CHIRPS files in {config.RAW_CHIRPS}. Run download_or_export.py")
    for p in chirps_files:
        print("=" * 70)
        if p.suffix == ".nc":
            info = describe_chirps_h5(p) if p.read_bytes()[:4] == b"\x89HDF" \
                else describe_nc_header(p)
        else:
            info = {"file": p.name, "kind": "GeoTIFF", "note": "handled in preprocess.py"}
        for k, v in info.items():
            if k == "vars":
                for vn, vd in v.items():
                    print(f"  var {vn}: shape={vd.get('shape', vd.get('dims'))} "
                          f"dtype={vd.get('dtype', '')}")
                    for ak, av in (vd.get("attrs") or {}).items():
                        print(f"      {ak} = {av}")
            else:
                print(f"  {k}: {v}")

    # ---------------- DEM ----------------
    print()
    dem_files = sorted(config.RAW_DEM.glob("*"))
    if not dem_files:
        print(f"!! No DEM files in {config.RAW_DEM}. Run download_or_export.py")
    for p in dem_files:
        print("=" * 70)
        if p.suffix == ".npz":
            z = np.load(p)
            print(f"DEM: {p}")
            for k in z.files:
                a = z[k]
                print(f"  {k}: shape={a.shape} dtype={a.dtype}"
                      + (f" min={a.min():.1f} max={a.max():.1f}"
                         if np.issubdtype(a.dtype, np.number) and a.size > 1 else f" = {a}"))
        else:
            print(f"DEM: {p} ({p.stat().st_size/1e6:.1f} MB) - handled in preprocess.py")


if __name__ == "__main__":
    main()
