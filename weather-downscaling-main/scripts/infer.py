"""
Inference: new IMD rainfall (+DEM, +ERA5 depending on the checkpoint) ->
fine-resolution rainfall map, without retraining.

Loads any checkpoint saved by train.py, reads the channel list it was trained
with, builds exactly those channels, and writes:
  - a PNG map per date (outputs/maps/infer_<date>.png)
  - a machine-readable .npz per run (prediction/infer_<date>.npz) with
    date / latitude / longitude / rainfall_mm -- the Layer-2 contract.

Examples:
  python scripts/infer.py --imd data/raw/imd/ind2022_rfp25.nc --date 2022-07-10
  python scripts/infer.py --imd ind2023_rfp25.nc --start 2023-06-01 --end 2023-06-30
  python scripts/infer.py --imd ... --checkpoint models/model_c.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from grids import bilinear_upsample, fine_grid  # noqa: E402
from imd_reader import load_imd  # noqa: E402
from preprocess import (_fill_nearest, load_dem, resample_to_grid,  # noqa: E402
                        sanitize_dem)
from train import SmallUNet  # noqa: E402


def _fill_sea(lat, lon, field):
    """Era5-Land sea points are structurally NaN; fill nearest (documented)."""
    if np.isnan(field).any():
        f = _fill_nearest(field)
        if f is None:
            raise ValueError("ERA5 field has no valid values on the ROI")
        return lat, lon, f
    return lat, lon, field


def build_channels(channels, imd, dem_fine, era5_day, era5_stats, meta,
                   rain_scale, elev_scale, sub):
    """Assemble the requested channel stack on the fine grid for all days."""
    fine_lat, fine_lon = fine_grid(imd["lat"], imd["lon"], sub)
    n = len(imd["rain"])
    out = []
    for name in channels:
        if name == "imd_rain":
            out.append(np.stack([bilinear_upsample(imd["rain"][i], sub)
                                 for i in range(n)]) / rain_scale)
        elif name == "dem":
            out.append(np.broadcast_to(dem_fine / elev_scale, (n, *dem_fine.shape)))
        elif name.startswith("era5_"):
            key = {"t2m": "t2m_mean"}.get(name.replace("era5_", "", 1),
                                          name.replace("era5_", "", 1))
            if era5_day is None or key not in era5_day:
                raise ValueError(
                    f"channel '{name}' needs ERA5 data for the requested "
                    "date(s); provide data/raw/era5/era5_daily.npz covering "
                    "them (re-run download_or_export.py) or use a checkpoint "
                    "trained without ERA5 channels (models/model_c.pt)")
            st = era5_stats[key]
            field = era5_day[key]  # (n, nlat, nlon) on the 0.5-deg lattice
            fine = np.stack([resample_to_grid(
                *_fill_sea(era5_day["lat"], era5_day["lon"], field[i]),
                fine_lat, fine_lon) for i in range(n)])
            out.append((fine - st["mean"]) / st["std"])
        else:
            raise ValueError(f"unknown channel {name}")
    X = np.stack(out, axis=1).astype("float32")
    return X, fine_lat, fine_lon


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imd", required=True,
                    help="IMD 0.25-deg NetCDF (same format as data/raw/imd/*.nc)")
    ap.add_argument("--date", default=None, help="single YYYY-MM-DD to predict")
    ap.add_argument("--start", default=None, help="or an inclusive date range")
    ap.add_argument("--end", default=None)
    ap.add_argument("--dem", default=None, help="optional DEM npz/tif "
                    "(default: data/raw/dem/dem_roi.npz)")
    ap.add_argument("--checkpoint", default=str(config.MODELS / "best_model.pt"))
    ap.add_argument("--output", default=None,
                    help="output PNG (single-day runs only)")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    channels = ckpt.get("channels") or ckpt["args"]["channels"]
    rain_scale = float(ckpt.get("rain_scale", 100.0))
    width = int(ckpt.get("width", 16))
    residual = bool(ckpt.get("residual", True))
    meta_path = config.PROCESSED / "meta.json"
    if not meta_path.exists():
        sys.exit(f"missing {meta_path}: run scripts/preprocess.py once so "
                 "normalization statistics exist")
    with open(meta_path) as f:
        meta = json.load(f)
    elev_scale = float(meta["normalization"]["elev_scale_m"])
    era5_stats = meta["normalization"].get("era5_standardization", {})
    model = SmallUNet(cin=len(channels), width=width, residual=residual)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.checkpoint}: channels={channels} residual={residual}")

    # ---------------- inputs ----------------
    imd_path = Path(args.imd)
    if not imd_path.exists():
        sys.exit(f"MISSING FILE: {imd_path}\nGet IMD files from "
                 "https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html "
                 "or run scripts/download_or_export.py")
    start = args.date or args.start or config.START_DATE_DEFAULT
    end = args.date or args.end or start
    roi = {**config.ROI_DEFAULT, "start_date": start, "end_date": end}
    imd = load_imd(imd_path, roi)
    print(f"[imd] {len(imd['dates'])} day(s): {imd['dates'][0]} .. {imd['dates'][-1]}")

    if args.dem:
        p = Path(args.dem)
        if p.suffix == ".npz":
            z = np.load(p)
            dlat, dlon, delev = z["lat"], z["lon"], z["elev"].astype("float32")
        else:
            import tifffile
            delev = tifffile.imread(p).astype("float32")
            r0 = config.ROI_DEFAULT
            dlat = np.linspace(max(r0["lat_min"] - 0.25, -89),
                               min(r0["lat_max"] + 0.25, 89), delev.shape[0])
            dlon = np.linspace(r0["lon_min"] - 0.25, r0["lon_max"] + 0.25,
                               delev.shape[1])
    else:
        dlat, dlon, delev = load_dem(roi)
    dlat, dlon, delev = sanitize_dem(dlat, dlon, delev)
    fine_lat, fine_lon = fine_grid(imd["lat"], imd["lon"], config.FINE_SUB)
    dem_fine = resample_to_grid(dlat, dlon, delev,
                                fine_lat, fine_lon).astype("float32")

    # ERA5 day(s) if any checkpoint channel needs it
    era5_day = None
    if any(c.startswith("era5_") for c in channels):
        e5 = config.RAW_IMD.parent / "era5" / "era5_daily.npz"
        if not e5.exists():
            sys.exit(f"checkpoint needs ERA5 channels but {e5} is missing - "
                     "run scripts/download_or_export.py, or use a checkpoint "
                     "trained without ERA5 (models/model_c.pt)")
        z = np.load(e5, allow_pickle=False)
        e_dates = [str(d) for d in z["dates"]]
        want = [str(d) for d in imd["dates"]]
        missing = [d for d in want if d not in e_dates]
        if missing:
            sys.exit(f"ERA5 data does not cover {missing[:3]} - re-run "
                     "download_or_export.py, or use models/model_c.pt")
        sel = [e_dates.index(d) for d in want]
        era5_day = {"lat": z["lat"], "lon": z["lon"]}
        for v in ("t2m_mean", "t2m_max", "dewp", "wind"):
            era5_day[v] = z[v][sel]

    X, fine_lat, fine_lon = build_channels(
        channels, imd, dem_fine, era5_day, era5_stats, meta,
        rain_scale, elev_scale, config.FINE_SUB)
    if bool(np.isfinite(X).all()) is False:
        sys.exit("non-finite values in inference input - check input data")
    print(f"[input] {X.shape} on the {config.FINE_SUB}x fine grid")

    # ---------------- predict ----------------
    base = None
    if residual:
        i0 = channels.index("imd_rain")
        base = torch.from_numpy(X[:, i0:i0 + 1].copy())
    with torch.no_grad():
        out = model(torch.from_numpy(X), baseline=base).numpy()
    pred_mm = np.clip(out[:, 0] * rain_scale, 0.0, None)

    # ---------------- outputs ----------------
    pred_dir = config.ROOT / "prediction"
    pred_dir.mkdir(exist_ok=True)
    dates = [str(d) for d in imd["dates"]]
    slug = "_".join(dates[0].split("-")) if len(dates) > 1 else dates[0]
    npz_path = pred_dir / f"infer_{slug}.npz"
    np.savez_compressed(npz_path, dates=np.array(dates),
                        latitude=fine_lat, longitude=fine_lon,
                        rainfall_mm=pred_mm.astype("float32"))
    with open(pred_dir / f"infer_{slug}_schema.json", "w") as f:
        json.dump({
            "file": npz_path.name,
            "checkpoint": Path(args.checkpoint).name,
            "channels": channels,
            "fields": {"dates": "YYYY-MM-DD per sample",
                       "latitude": "fine-grid lat centers (deg N, ascending)",
                       "longitude": "fine-grid lon centers (deg E, ascending)",
                       "rainfall_mm": "(n_days, lat, lon) daily rainfall mm/day"},
            "grid": {"resolution_deg": 0.05, "roi": meta["roi"]},
            "note": "Layer-1 output for Panchayat-polygon intersection (Layer 2)",
        }, f, indent=1)
    print(f"predictions -> {npz_path} (+ schema json)")

    for i, d in enumerate(dates):
        out_png = (Path(args.output) if args.output and len(dates) == 1
                   else config.OUT_MAPS / f"infer_{d}.png")
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        im = ax.imshow(pred_mm[i], origin="lower",
                       extent=[fine_lon[0], fine_lon[-1], fine_lat[0], fine_lat[-1]],
                       cmap="YlGnBu", interpolation="nearest")
        ax.set_title(f"Downscaled rainfall ({Path(args.checkpoint).name}) - {d}")
        ax.set_xlabel("lon [deg E]")
        ax.set_ylabel("lat [deg N]")
        fig.colorbar(im, ax=ax, label="mm/day")
        fig.savefig(out_png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        npy_path = config.OUT_MAPS / f"infer_{d}.npy"
        np.save(npy_path, pred_mm[i].astype("float32"))
        print(f"  {d}: max {pred_mm[i].max():.1f} mm -> {out_png} + {npy_path.name}")


if __name__ == "__main__":
    main()
