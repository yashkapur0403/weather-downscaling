"""Shared fixtures.

`tiny_repo` writes the minimum set of artefacts `data_store.Store` needs. Its
grid deliberately contains the exact rainfall values the advisory contract tests
assert on, so those tests run through the REAL verification path (the server
checks a supplied rainfall against the stored field) instead of bypassing it.

    panchayat_id  cell   stored rainfall
    11            (16,17)  70.0 mm     -> 'alert' tier
    12            (17,18)  30.0 mm     -> 'warning' tier
    13            (18,19)  12.0 mm     -> 'info' tier
    14            (19,20)   2.0 mm
    15            (20,21)  NaN         -> masked cell (must refuse)
    16            (21,22)   0.0 mm     -> a real, falsy-in-JS answer
    99            absent               -> not in the coordinate index

Grid: 40x40 at 0.05 deg, lat 12.025.., lon 74.025.., single date 2022-07-10.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
os.environ.setdefault("GROQ_API_KEY", "x")
os.environ.setdefault("SARVAM_API_KEY", "x")

DATE = "2022-07-10"
N = 40
LAT0, LON0, STEP = 12.025, 74.025, 0.05
CELLS = {
    11: (16, 17, 70.0),
    12: (17, 18, 30.0),
    13: (18, 19, 12.0),
    14: (19, 20, 2.0),
    15: (20, 21, None),
    16: (21, 22, 0.0),
}
ID_ALERT, ID_WARNING, ID_INFO, ID_TINY, ID_MASKED, ID_ZERO = 11, 12, 13, 14, 15, 16
ID_UNKNOWN = 99


@pytest.fixture(scope="session")
def tiny_repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("repo")
    out = root / "outputs"
    l2 = out / "layer2"
    l2.mkdir(parents=True)
    metrics = out / "metrics"
    metrics.mkdir()

    lat = LAT0 + STEP * np.arange(N)
    lon = LON0 + STEP * np.arange(N)
    rain = np.full((1, N, N), 21.0, dtype="float32")

    rows, index = [], []
    for pid, (i, j, mm) in CELLS.items():
        rain[0, i, j] = np.nan if mm is None else mm
        rows.append(dict(state="KARNATAKA", district="D", block_id=1, block_name="B",
                         panchayat_id=pid, panchayat_name=f"GP{pid}", n_days=1,
                         rainfall_mean_mm=(np.nan if mm is None else mm),
                         rainfall_min_mm=0.0,
                         rainfall_max_mm=(np.nan if mm is None else mm),
                         n_wet_days=0, n_cells=1,
                         mapping_method="direct_grid", fallback_distance_m=None))
        index.append(dict(panchayat_id=pid, lat=float(lat[i]), lon=float(lon[j])))

    pd.DataFrame(rows).to_csv(l2 / "panchayat_summary.csv", index=False)
    pd.DataFrame(index).to_csv(l2 / "panchayat_index.csv", index=False)
    np.savez(out / "prediction_test.npz", dates=np.array([DATE]),
             latitude=lat, longitude=lon, rainfall_mm=rain)

    json.dump({"rows": {"A": {"label": "Bilinear IMD baseline"},
                        "D": {"label": "U-Net D"}},
               "selection": {"row": "D", "label": "U-Net D"},
               "results": {"val": {"A": dict(MAE=9, RMSE=10, corr=.3),
                                   "D": dict(MAE=8, RMSE=9, corr=.4)},
                           "test": {"A": dict(MAE=10, RMSE=11, corr=.3),
                                    "D": dict(MAE=8, RMSE=9, corr=.4)}}},
              open(metrics / "ablation.json", "w"))
    json.dump({"meta": {"channels": ["imd_rain"], "reference": "ref"}},
              open(metrics / "metrics.json", "w"))
    return root


@pytest.fixture
def app_with_data(tiny_repo, monkeypatch):
    """main.app with the data layer wired to `tiny_repo`.

    This is what `backend/app.py` does in production (routes_data.register
    publishes the resolver on app.state); here it is patched per-test so no
    global state leaks between modules.
    """
    import data_store
    import main

    store = data_store.store_for_root(tiny_repo)

    def resolver(panchayat_id: int, date: str) -> dict:
        return store.panchayat_grid_value(int(panchayat_id), date)

    monkeypatch.setattr(main.app.state, "rainfall_resolver", resolver, raising=False)
    return main.app


@pytest.fixture
def app_without_data(monkeypatch):
    """main.app with NO data layer mounted (text-only deployment)."""
    import main

    monkeypatch.delattr(main.app.state, "rainfall_resolver", raising=False)
    return main.app
