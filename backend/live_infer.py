"""
Issue 5 - real, on-demand U-Net inference for a location that is not in the pre-computed CSV.

Wraps the repo's own scripts/infer.py building blocks (same channel order, same normalisation,
same residual-U-Net call) so there is exactly ONE definition of "how inputs are built".

    runner = ModelRunner(REPO_ROOT)                     # once, at server start (loads checkpoint)
    out = runner.rainfall_at(lat, lon, "2022-07-10")    # -> {"rainfall_mm": .., "cell_lat": .., ...}

The whole 285x200 fine grid is predicted once per date (a few hundred ms on CPU) and cached, so
clicking many panchayats on the same date costs one forward pass.

Raises DataUnavailable (with a message the API can show) instead of returning a fake number.
NOT covered by the unit tests here: the torch/data path needs the checkpoint + data/raw + torch,
which are not in git. Run `python live_infer.py --check` on your machine to validate it end-to-end.
"""
from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np


class DataUnavailable(RuntimeError):
    """A required input (IMD file, ERA5 day, DEM) is missing for the requested date."""


class OutsideDomain(ValueError):
    """The requested point is outside the trained study region / fine grid."""


def sample_grid(field: np.ndarray, fine_lat: np.ndarray, fine_lon: np.ndarray,
                lat: float, lon: float, tol_deg: float = 0.05) -> dict:
    """Nearest fine-grid cell to (lat, lon). Pure numpy; rejects points outside the grid."""
    step = 0.05
    if not (fine_lat[0] - step / 2 - tol_deg <= lat <= fine_lat[-1] + step / 2 + tol_deg and
            fine_lon[0] - step / 2 - tol_deg <= lon <= fine_lon[-1] + step / 2 + tol_deg):
        raise OutsideDomain(f"({lat:.3f}, {lon:.3f}) is outside the model domain")
    i = int(np.abs(fine_lat - lat).argmin())
    j = int(np.abs(fine_lon - lon).argmin())
    v = float(field[i, j])
    if not np.isfinite(v):
        raise DataUnavailable("model output is not finite at this cell (sea / excluded cell)")
    return {"rainfall_mm": round(v, 2), "fine_i": i, "fine_j": j,
            "cell_lat": float(fine_lat[i]), "cell_lon": float(fine_lon[j]), "n_cells": 1}


class ModelRunner:
    def __init__(self, repo_root: str | Path, checkpoint: Optional[str] = None,
                 region: Optional[str] = None, cache_days: int = 8):
        import torch
        self.root = Path(repo_root)
        sys.path.insert(0, str(self.root / "scripts"))
        import config
        from infer import build_channels            # noqa: F401  (reused, not re-implemented)
        # ONE definition of "what the deployed model is": scripts/ensemble.py
        # resolves models/ensemble.json (the weighted-member declaration written
        # by scripts/ablation.py) and falls back to models/best_model.pt. Using
        # it here too means live inference and outputs/prediction_test.npz can
        # never be produced by different models.
        from ensemble import ensemble_predict_mm, load_members, load_single
        self._torch, self._config, self._build = torch, config, build_channels
        self.region = region

        scripts_dir = self.root / "scripts"
        if checkpoint:
            self.nets, self.spec, self.manifest = load_single(checkpoint, scripts_dir)
        else:
            self.nets, self.spec, self.manifest = load_members(config.MODELS, scripts_dir)
        self.channels = self.spec["channels"]
        self.rain_scale = float(self.spec["rain_scale"])
        self.residual = bool(self.spec["residual"])
        self.checkpoint_name = " + ".join(self.spec["members"])
        self.n_parameters = int(self.spec["n_parameters"] or 0)

        meta_path = config.PROCESSED / "meta.json"
        if not meta_path.exists():
            raise DataUnavailable(f"{meta_path} missing: run scripts/preprocess.py (normalisation stats)")
        self.meta = json.loads(meta_path.read_text())
        self.elev_scale = float(self.meta["normalization"]["elev_scale_m"])
        self.era5_stats = self.meta["normalization"].get("era5_standardization", {})
        self._cache: "OrderedDict[str, tuple]" = OrderedDict()
        self._cache_days = cache_days
        self._dem_fine = None

    # ---- inputs -----------------------------------------------------------
    def _imd_path(self, date: str) -> Path:
        p = self._config.RAW_IMD / f"ind{date[:4]}_rfp25.nc"
        if not p.exists():
            raise DataUnavailable(f"IMD file for {date[:4]} not found at {p}")
        return p

    def _era5_day(self, dates: list[str]):
        if not any(c.startswith("era5_") for c in self.channels):
            return None
        cfg = self._config
        e5 = cfg.RAW_ERA5 / f"era5_daily_{self.region or cfg.REGION_DEFAULT}.npz"
        if not e5.exists():
            e5 = cfg.RAW_ERA5 / "era5_daily.npz"
        if not e5.exists():
            raise DataUnavailable(f"ERA5 cache missing ({e5}); needed by channels {self.channels}")
        z = np.load(e5, allow_pickle=False)
        have = [str(d) for d in z["dates"]]
        if any(d not in have for d in dates):
            raise DataUnavailable(f"ERA5 cache does not cover {dates[0]}")
        sel = [have.index(d) for d in dates]
        out = {"lat": z["lat"], "lon": z["lon"]}
        for v in ("t2m_mean", "t2m_max", "dewp", "wind"):
            out[v] = z[v][sel]
        return out

    def _dem_on_fine_grid(self, imd):
        if self._dem_fine is None:
            from grids import fine_grid
            from preprocess import load_dem, resample_to_grid, sanitize_dem
            cfg = self._config
            roi = {**cfg.region_roi(self.region)}
            dlat, dlon, delev = sanitize_dem(*load_dem(roi, self.region))
            fl, fo = fine_grid(imd["lat"], imd["lon"], cfg.FINE_SUB)
            self._dem_fine = resample_to_grid(dlat, dlon, delev, fl, fo).astype("float32")
        return self._dem_fine

    # ---- prediction -------------------------------------------------------
    def predict_day(self, date: str):
        if date in self._cache:
            self._cache.move_to_end(date)
            return self._cache[date]
        from imd_reader import load_imd
        cfg, torch = self._config, self._torch
        roi = {**cfg.region_roi(self.region), "start_date": date, "end_date": date}
        try:
            imd = load_imd(self._imd_path(date), roi)
        except DataUnavailable:
            raise
        except Exception as e:
            raise DataUnavailable(f"could not read IMD data for {date}: {e}") from e
        if len(imd["dates"]) != 1:
            raise DataUnavailable(f"IMD has no data for {date}")
        X, fine_lat, fine_lon = self._build(
            self.channels, imd, self._dem_on_fine_grid(imd), self._era5_day([date]),
            self.era5_stats, self.meta, self.rain_scale, self.elev_scale, cfg.FINE_SUB)
        if not np.isfinite(X).all():
            raise DataUnavailable(f"non-finite model input for {date} (missing IMD/ERA5 values)")
        i0 = self.channels.index("imd_rain")
        base = torch.from_numpy(X[:, i0:i0 + 1].copy()) if self.residual else None
        # same helper every other serving path uses -> live inference cannot
        # disagree with outputs/prediction_test.npz
        pred = ensemble_predict_mm(
            self.nets, self.spec, torch.from_numpy(X), base
        ).numpy()[0].astype("float32")
        self._cache[date] = (pred, fine_lat, fine_lon)
        while len(self._cache) > self._cache_days:
            self._cache.popitem(last=False)
        return self._cache[date]

    def rainfall_at(self, lat: float, lon: float, date: str) -> dict:
        pred, fine_lat, fine_lon = self.predict_day(date)
        out = sample_grid(pred, fine_lat, fine_lon, lat, lon)
        out.update(date=date, checkpoint=self.checkpoint_name, channels=self.channels)
        return out


if __name__ == "__main__":       # python live_infer.py --check  (run from anywhere)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--date", default="2022-07-10")
    ap.add_argument("--lat", type=float, default=17.4)
    ap.add_argument("--lon", type=float, default=78.4)
    a = ap.parse_args()
    r = ModelRunner(a.repo)
    print(f"{r.checkpoint_name}: {r.n_parameters:,} params, channels={r.channels}")
    print(r.rainfall_at(a.lat, a.lon, a.date))
