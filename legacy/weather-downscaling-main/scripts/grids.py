"""Grid helpers: coarse<->fine alignment and bilinear upsampling.

The fine grid is defined as exactly FINE_SUB x FINE_SUB sub-points inside each
IMD 0.25-deg cell, so the downscaling factor is FINE_SUB in each dimension
(default 5 -> 0.05 deg, matching CHIRPS).
"""
from __future__ import annotations

import numpy as np


def fine_grid(imd_lat: np.ndarray, imd_lon: np.ndarray, sub: int = 5):
    """
    Fine-grid cell centers covering exactly the same area as the coarse grid.

    IMD cells are 0.25-deg boxes centered on the given coordinates; each box is
    divided into sub x sub sub-boxes and we return the sub-box centers.
    For IMD lat [12, 12.25] with sub=5 -> fine lat centers
    [12.025, 12.075, 12.125, 12.175, 12.225].

    Returns (fine_lat, fine_lon), both ascending.
    """
    step = 0.25 / sub
    # left/right edges of every coarse cell
    lat_edges = np.concatenate([imd_lat - 0.125, [imd_lat[-1] + 0.125]])
    lon_edges = np.concatenate([imd_lon - 0.125, [imd_lon[-1] + 0.125]])
    fine_lat = np.concatenate(
        [lat_edges[i] + step * (np.arange(sub) + 0.5) for i in range(len(imd_lat))])
    fine_lon = np.concatenate(
        [lon_edges[j] + step * (np.arange(sub) + 0.5) for j in range(len(imd_lon))])
    return fine_lat.astype("float64"), fine_lon.astype("float64")


def bilinear_upsample(field: np.ndarray, sub: int = 5, fill_value: float | None = None):
    """
    Bilinearly upsample a (..., H, W) coarse field to (..., H*sub, W*sub).

    NaN values propagate (used by the baseline so missing IMD stays missing).
    Vectorized implementation; no scipy dependency.
    """
    *lead, H, W = field.shape
    # place values at cell centers on a unit grid, interpolate to sub-grid
    gy = (np.arange(H * sub) + 0.5) / sub - 0.5     # positions in coarse index space
    gx = (np.arange(W * sub) + 0.5) / sub - 0.5
    gy = np.clip(gy, 0, H - 1)
    gx = np.clip(gx, 0, W - 1)
    y0 = np.floor(gy).astype(int)
    x0 = np.floor(gx).astype(int)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)
    wy = (gy - y0)[:, None]                          # (H*sub, 1)
    wx = (gx - x0)[None, :]                          # (1, W*sub)

    f = field.astype("float32")
    a = f[..., y0, :][..., :, x0]                    # (..., H*sub, W*sub)
    b = f[..., y0, :][..., :, x1]
    c = f[..., y1, :][..., :, x0]
    d = f[..., y1, :][..., :, x1]
    top = a + (b - a) * wx
    bot = c + (d - c) * wx
    out = top + (bot - top) * wy

    if fill_value is not None:
        out = np.where(np.isnan(field[..., y0, :][..., :, x0]), fill_value, out)
    return out.astype("float32")


def aggregate_to_coarse(field_fine: np.ndarray, sub: int = 5) -> np.ndarray:
    """Block-average a (..., H*sub, W*sub) fine field back to (..., H, W).

    NaN-aware: averages the valid sub-pixels; a coarse cell becomes NaN only if
    every sub-pixel is missing."""
    *lead, Hf, Wf = field_fine.shape
    H, W = Hf // sub, Wf // sub
    v = field_fine.reshape(*lead, H, sub, W, sub)
    m = np.isnan(v)
    if not m.any():
        return v.mean(axis=(-3, -1))
    cnt = (~m).sum(axis=(-3, -1))
    tot = np.where(m, 0.0, v).sum(axis=(-3, -1))
    out = tot / np.maximum(cnt, 1)
    return np.where(cnt == 0, np.nan, out)
