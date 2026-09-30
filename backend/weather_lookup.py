"""
Issue 1 - replace _synthetic_context (hash-of-place-name fake numbers) with real sources.

  temperature / humidity : Open-Meteo archive (ERA5-Land family, keyless)
  elevation              : the SAME SRTM DEM the model was trained with, if data/raw/dem/dem_roi.npz
                           is present (preferred, offline); otherwise Open-Topo-Data (SRTM 90 m)

Every value carries its source string, and a failure returns None + an error message.
It never fabricates a number.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_TOPO = "https://api.opentopodata.org/v1/srtm90m"


@dataclass
class WeatherContext:
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    elevation_m: Optional[float] = None
    temperature_source: Optional[str] = None
    humidity_source: Optional[str] = None
    elevation_source: Optional[str] = None
    error: Optional[str] = None
    _errs: list[str] = field(default_factory=list, repr=False)

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("temperature_c", "humidity_pct", "elevation_m", "temperature_source",
                                           "humidity_source", "elevation_source")}
        d["error"] = "; ".join(self._errs) or None
        return d


class LocalDEM:
    """Nearest-cell lookup in the training DEM (npz with lat, lon, elev)."""

    def __init__(self, npz_path: str | Path):
        z = np.load(npz_path)
        self.lat, self.lon, self.elev = z["lat"].astype("float64"), z["lon"].astype("float64"), z["elev"]

    def at(self, lat: float, lon: float) -> Optional[float]:
        if not (self.lat.min() <= lat <= self.lat.max() and self.lon.min() <= lon <= self.lon.max()):
            return None
        i, j = int(np.abs(self.lat - lat).argmin()), int(np.abs(self.lon - lon).argmin())
        v = float(self.elev[i, j])
        return None if not np.isfinite(v) else round(v, 1)


async def fetch_weather(lat: float, lon: float, date: str, client: httpx.AsyncClient,
                        dem: Optional[LocalDEM] = None) -> WeatherContext:
    ctx = WeatherContext()

    async def temp_hum():
        try:
            r = await client.get(OPEN_METEO_ARCHIVE, params={
                "latitude": lat, "longitude": lon, "start_date": date, "end_date": date,
                "daily": "temperature_2m_mean,relative_humidity_2m_mean", "timezone": "Asia/Kolkata"}, timeout=15)
            r.raise_for_status()
            daily = r.json().get("daily", {})
            t = (daily.get("temperature_2m_mean") or [None])[0]
            h = (daily.get("relative_humidity_2m_mean") or [None])[0]
            if t is not None:
                ctx.temperature_c, ctx.temperature_source = round(t, 1), "Open-Meteo Archive (ERA5 reanalysis)"
            if h is not None:
                ctx.humidity_pct, ctx.humidity_source = round(h), "Open-Meteo Archive (ERA5 reanalysis)"
            if t is None and h is None:
                ctx._errs.append("Open-Meteo returned no data for this date")
        except Exception as e:
            ctx._errs.append(f"weather lookup failed: {type(e).__name__}")

    async def elevation():
        if dem is not None:
            v = dem.at(lat, lon)
            if v is not None:
                ctx.elevation_m, ctx.elevation_source = v, "SRTM DEM (model training grid)"
                return
        try:
            r = await client.get(OPEN_TOPO, params={"locations": f"{lat},{lon}"}, timeout=15)
            r.raise_for_status()
            v = r.json()["results"][0]["elevation"]
            if v is not None:
                ctx.elevation_m, ctx.elevation_source = round(v, 1), "Open-Topo-Data (SRTM 90 m)"
            else:
                ctx._errs.append("elevation unavailable")
        except Exception as e:
            ctx._errs.append(f"elevation lookup failed: {type(e).__name__}")

    await asyncio.gather(temp_hum(), elevation())
    return ctx
