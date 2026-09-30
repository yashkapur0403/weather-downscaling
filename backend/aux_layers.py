"""
Auxiliary Layer-2 data access (soil / NDVI / land cover) for the advisory engine.

These committed arrays were previously NOT used anywhere in the product. This
module maps a Panchayat (or lat/lon) onto the fine 0.05-degree grid and returns
the static soil properties, the NDVI composite nearest to the requested date and
the land-cover fractions for that cell.

Nothing is invented: if a cell has no aux value (sea / outside the sampled land
lattice) the corresponding field is simply absent.

  data/aux_data/soil_soilgrids_deccan.npz      fine_values (47250,5) sand/clay/ocd/phh2o/bdod
  data/aux_data/ndvi_monthly_deccan.npz        ndvi (20, 47250) monthly composites + months
  data/aux_data/lulc_fractions_deccan.npz      fractions (47250,6) + fraction_names/dominant_class
  data/processed/meta.json                     fine grid lat/lon (285 x 200)
  outputs/layer2/panchayat_index.csv           panchayat_id -> lat/lon (real polygon point)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class AuxLayers:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.available = False
        aux = self.root / "data" / "aux_data"
        meta_p = self.root / "data" / "processed" / "meta.json"
        self._coords: dict[int, tuple[float, float]] = {}

        try:
            meta = json.loads(meta_p.read_text())
            self.fine_lat = np.asarray(meta["grid"]["fine_lat"], dtype=float)
            self.fine_lon = np.asarray(meta["grid"]["fine_lon"], dtype=float)
        except Exception:
            return

        idx = self.root / "outputs" / "layer2" / "panchayat_index.csv"
        if idx.exists():
            c = pd.read_csv(idx)
            self._coords = {int(r.panchayat_id): (float(r.lat), float(r.lon)) for r in c.itertuples()}

        self._soil: dict[tuple[int, int], dict] = {}
        self._lulc: dict[tuple[int, int], dict] = {}
        self._ndvi: dict[tuple[int, int], int] = {}       # cell -> row index in ndvi array
        self._ndvi_arr = None
        self._ndvi_months: list[str] = []

        try:
            s = np.load(aux / "soil_soilgrids_deccan.npz", allow_pickle=False)
            props = [str(x) for x in s["properties"]]
            vals = s["fine_values"]
            for k, (i, j) in enumerate(zip(s["fine_i"], s["fine_j"])):
                if np.isfinite(vals[k]).any():
                    self._soil[(int(i), int(j))] = dict(zip(props, (float(x) for x in vals[k])))
        except Exception:
            pass

        try:
            l = np.load(aux / "lulc_fractions_deccan.npz", allow_pickle=True)
            fnames = [str(x) for x in l["fraction_names"]]
            dom = l["dominant_class"]
            fr = l["fractions"]
            legend = {}
            try:
                legend = {int(k): str(v) for k, v in json.loads(str(l["class_legend"])).items()}
            except Exception:
                legend = {}
            for k, (i, j) in enumerate(zip(l["fine_i"], l["fine_j"])):
                d = dict(zip(fnames, (float(x) for x in fr[k])))
                cell = (int(i), int(j))
                try:
                    code = int(dom[k])
                except Exception:
                    code = None
                name = legend.get(code, str(dom[k]))
                self._lulc[cell] = {"fractions": d, "dominant": name}
        except Exception:
            pass

        try:
            n = np.load(aux / "ndvi_monthly_deccan.npz", allow_pickle=False)
            self._ndvi_months = [str(m) for m in n["months"]]
            self._ndvi_arr = n["ndvi"]
            for k, (i, j) in enumerate(zip(n["fine_i"], n["fine_j"])):
                self._ndvi[(int(i), int(j))] = k
        except Exception:
            pass

        self.available = bool(self._soil or self._lulc or self._ndvi)

    # ------------------------------------------------------------------ geometry
    def cell(self, lat: float, lon: float) -> tuple[int, int]:
        i = int(np.abs(self.fine_lat - lat).argmin())
        j = int(np.abs(self.fine_lon - lon).argmin())
        return i, j

    def _nearest_month_col(self, date: Optional[str]) -> int:
        if not self._ndvi_months:
            return -1
        if not date or len(date) < 7:
            return len(self._ndvi_months) - 1          # latest composite
        ym = date[:7]
        if ym in self._ndvi_months:
            return self._ndvi_months.index(ym)
        # nearest month by absolute month distance
        y, m = int(ym[:4]), int(ym[5:7])
        target = y * 12 + m
        best, best_d = len(self._ndvi_months) - 1, 10 ** 9
        for k, mm in enumerate(self._ndvi_months):
            d = abs((int(mm[:4]) * 12 + int(mm[5:7])) - target)
            if d < best_d:
                best, best_d = k, d
        return best

    def lookup_cell(self, lat: float, lon: float, date: Optional[str] = None) -> Optional[dict]:
        i, j = self.cell(lat, lon)
        key = (i, j)
        out: dict = {"cell": [i, j]}
        if key in self._soil:
            pr = self._soil[key]
            out["soil"] = {
                "depth": "5-15cm",
                "sand_g_per_kg": pr.get("sand"),
                "clay_g_per_kg": pr.get("clay"),
                "ocd_dg_per_dm3": pr.get("ocd"),
                "ph": (pr.get("phh2o") / 10.0) if pr.get("phh2o") is not None else None,
                "bdod": pr.get("bdod"),
            }
        if key in self._lulc:
            out["lulc"] = self._lulc[key]
        if key in self._ndvi and self._ndvi_arr is not None:
            k = self._ndvi[key]
            col = self._nearest_month_col(date)
            v = float(self._ndvi_arr[col, k])
            if np.isfinite(v):
                out["ndvi"] = {"value": round(v, 4), "month": self._ndvi_months[col]}
        return out if len(out) > 1 else None

    def lookup_panchayat(self, panchayat_id: int, date: Optional[str] = None) -> Optional[dict]:
        if panchayat_id not in self._coords:
            return None
        lat, lon = self._coords[panchayat_id]
        return self.lookup_cell(lat, lon, date)
