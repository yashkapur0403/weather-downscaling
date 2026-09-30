"""
Read-only access to the repo's REAL artefacts. Nothing here invents a value.

  outputs/layer2/panchayat_summary.csv     one row per panchayat (name, block, district, mapping_method, n_cells ...)
  outputs/layer2/panchayat_index.csv       OPTIONAL  panchayat_id, lat, lon  (built by build_panchayat_index.py)
  outputs/layer2/panchayat_weather.csv     OPTIONAL  per-date Layer-2 output (date, panchayat_id, rainfall_mm)
  outputs/prediction_test.npz              U-Net fine-grid rainfall for 2022-06-01..2022-09-30
  prediction/infer_*.npz                   more U-Net runs written by scripts/infer.py
  data/aux_data/admin/grid_admin_map_deccan.npz   used ONLY to find a district/block centre when no exact
                                                   coordinates exist (flagged via `location_precision`)
"""
from __future__ import annotations

import difflib
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from live_infer import DataUnavailable, OutsideDomain, sample_grid

DEFAULT_DATE = "2022-07-10"


def norm(s: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


class Store:
    def __init__(self, repo_root: str | Path):
        self.root = Path(repo_root)
        l2 = self.root / "outputs" / "layer2"
        path = l2 / "panchayat_summary.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run scripts/layer2_panchayat_mapping.py")
        df = pd.read_csv(path)
        df = df[df["mapping_method"].isin(["direct_grid", "area_weighted", "nearest_fallback"])].copy()
        df["_n_name"] = df["panchayat_name"].map(norm)
        df["_n_block"] = df["block_name"].map(norm)
        df["_n_dist"] = df["district"].map(norm)
        df["_n_state"] = df["state"].map(norm)
        df["_hay"] = df["_n_name"] + " " + df["_n_block"] + " " + df["_n_dist"] + " " + df["_n_state"]
        self.df = df.reset_index(drop=True)
        self.n_unmapped_dropped = int(pd.read_csv(path, usecols=["mapping_method"])["mapping_method"].eq("unmapped").sum())

        self.coords: dict[int, tuple[float, float]] = {}
        idx = l2 / "panchayat_index.csv"
        if idx.exists():
            c = pd.read_csv(idx)
            self.coords = {int(r.panchayat_id): (float(r.lat), float(r.lon)) for r in c.itertuples()}

        self.weather_csv = l2 / "panchayat_weather.csv"
        self._admin = None
        self._grids: list[dict] = []
        self._load_grids()

    # ------------------------------------------------------------- search
    def search(self, q: str, limit: int = 20) -> list[dict]:
        toks = [t for t in norm(q).split() if t]
        if not toks:
            return []
        m = np.ones(len(self.df), dtype=bool)
        for t in toks:
            m &= self.df["_hay"].str.contains(re.escape(t), regex=True).to_numpy()
        hit = self.df[m].copy()
        if hit.empty:
            return []
        first = toks[0]
        hit["_rank"] = np.where(hit["_n_name"] == norm(q), 0,
                        np.where(hit["_n_name"].str.startswith(first), 1,
                        np.where(hit["_n_name"].str.contains(first, regex=False), 2, 3)))
        hit = hit.sort_values(["_rank", "panchayat_name"]).head(limit)
        return [self._row_to_panchayat(r) for r in hit.itertuples()]

    def _row_to_panchayat(self, r) -> dict:
        lat, lon, prec = self.locate_row(r)
        fb = None if pd.isna(r.fallback_distance_m) else float(r.fallback_distance_m)
        return {
            "panchayat_id": int(r.panchayat_id), "panchayat_name": r.panchayat_name,
            "block_name": r.block_name, "block_id": int(r.block_id), "district": r.district, "state": r.state,
            "date": DEFAULT_DATE,
            # Season mean of the model's 2022 monsoon output for this panchayat (a placeholder until the user
            # runs a projection, which replaces it with the value for the chosen date).
            "rainfall_mm": round(float(r.rainfall_mean_mm), 2), "rainfall_basis": "season_mean_2022",
            "temperature_c": None, "humidity_pct": None, "elevation_m": None,   # filled by /auth/ from real sources
            "n_cells": int(r.n_cells), "mapping_method": r.mapping_method, "fallback_distance_m": fb,
            "lat": lat, "lon": lon, "location_precision": prec,
        }

    def matches(self, name: Optional[str], block: Optional[str] = None, district: Optional[str] = None,
                state: Optional[str] = None, panchayat_id: Optional[int] = None) -> pd.DataFrame:
        """Every row this lookup could mean. More than one => the caller MUST disambiguate
        (a bare name is NOT enough when several Panchayats share it - see find())."""
        d = self.df
        if panchayat_id is not None:
            return d[d["panchayat_id"] == panchayat_id]
        if not name:
            return d.iloc[0:0]
        m = d["_n_name"] == norm(name)
        if not m.any():
            return d.iloc[0:0]
        for col, val in (("_n_block", block), ("_n_dist", district), ("_n_state", state)):
            if val and (m & (d[col] == norm(val))).any():
                m &= d[col] == norm(val)
        return d[m]

    def find(self, name: Optional[str], block: Optional[str] = None, district: Optional[str] = None,
             state: Optional[str] = None, panchayat_id: Optional[int] = None):
        """Single best row (first match). Callers that may receive an ambiguous name should
        use matches() and refuse when len > 1 rather than silently taking a different Panchayat."""
        h = self.matches(name, block, district, state, panchayat_id)
        return h.iloc[0] if len(h) else None

    @staticmethod
    def candidate_dicts(h: pd.DataFrame, limit: int = 10) -> list[dict]:
        return [{"panchayat_id": int(r.panchayat_id), "panchayat_name": r.panchayat_name,
                 "block_name": r.block_name, "district": r.district, "state": r.state}
                for r in h.head(limit).itertuples()]

    # ---------------------------------------------------------- coordinates
    def _admin_centres(self):
        if self._admin is None:
            p = self.root / "data" / "aux_data" / "admin" / "grid_admin_map_deccan.npz"
            self._admin = {"district": {}, "block": {}}
            if p.exists():
                z = np.load(p, allow_pickle=True)
                t = pd.DataFrame({"s": [norm(x) for x in z["state_names"]], "d": [norm(x) for x in z["district_names"]],
                                  "b": [norm(x) for x in z["subdistrict_names"]],
                                  "lat": z["fine_lat"], "lon": z["fine_lon"]})
                for (s, d), g in t.groupby(["s", "d"]):
                    self._admin["district"][(s, d)] = (float(g.lat.mean()), float(g.lon.mean()))
                for (s, d, b), g in t.groupby(["s", "d", "b"]):
                    self._admin["block"][(s, d, b)] = (float(g.lat.mean()), float(g.lon.mean()))
        return self._admin

    def locate_row(self, r) -> tuple[Optional[float], Optional[float], str]:
        """(lat, lon, precision). precision: exact | block | district | unknown. Never guesses beyond that."""
        pid = int(r.panchayat_id)
        if pid in self.coords:
            return (*self.coords[pid], "exact")
        a = self._admin_centres()
        s, d, b = norm(r.state), norm(getattr(r, "district")), norm(r.block_name)
        # The admin NPZ uses CamelCase state names (e.g. "AndhraPradesh") which norm() collapses
        # to a single token without spaces, while CSV state names ("ANDHRA PRADESH") keep spaces.
        # Try both the spaced form and the space-stripped form so both match.
        s_nospace = s.replace(" ", "")
        dists = [k for k in a["district"] if k[0] in (s, s_nospace)]
        dm = difflib.get_close_matches(d, [k[1] for k in dists], n=1, cutoff=0.85)
        if dm:
            # Determine which state key form the admin dict actually uses
            s_key = dists[0][0] if dists else s
            bl = [k for k in a["block"] if k[0] == s_key and k[1] == dm[0]]
            bm = difflib.get_close_matches(b, [k[2] for k in bl], n=1, cutoff=0.85)
            if bm:
                return (*a["block"][(s_key, dm[0], bm[0])], "block")
            return (*a["district"][(s_key, dm[0])], "district")
        return (None, None, "unknown")

    # --------------------------------------------------------------- rainfall
    def _load_grids(self):
        cands = [self.root / "outputs" / "prediction_test.npz", *sorted((self.root / "prediction").glob("infer_*.npz"))]
        for p in cands:
            if not p.exists():
                continue
            z = np.load(p, allow_pickle=False)
            self._grids.append({"path": p, "dates": [str(x) for x in z["dates"]], "lat": z["latitude"],
                                "lon": z["longitude"], "rain": z["rainfall_mm"]})

    @property
    def grid_dates(self) -> list[str]:
        return sorted({d for g in self._grids for d in g["dates"]})

    def layer2_value(self, panchayat_id: int, date: str) -> Optional[float]:
        """Exact per-date Layer-2 value if panchayat_weather.csv exists (cached per date)."""
        if not self.weather_csv.exists():
            return None
        return self._weather_day(date).get(panchayat_id)

    @lru_cache(maxsize=6)
    def _weather_day(self, date: str) -> dict[int, float]:
        out: dict[int, float] = {}
        for ch in pd.read_csv(self.weather_csv, usecols=["date", "panchayat_id", "rainfall_mm"], chunksize=500_000):
            s = ch[ch["date"] == date]
            out.update(dict(zip(s["panchayat_id"].astype(int), s["rainfall_mm"].astype(float))))
        return out

    def grid_rainfall(self, lat: float, lon: float, date: str) -> Optional[dict]:
        for g in self._grids:
            if date in g["dates"]:
                r = sample_grid(g["rain"][g["dates"].index(date)], g["lat"], g["lon"], lat, lon)
                r["source"] = f"U-Net prediction grid ({g['path'].name})"
                return r
        return None

    def panchayat_grid_value(self, panchayat_id: int, date: str) -> dict:
        """The AUTHORITATIVE Layer-1 rainfall for one Panchayat and date.

        This is the value a caller's claim is checked against before it is allowed
        to reach Layer 3. It never invents a number: when the answer is not
        obtainable, `rainfall_mm` is None and `reason` names the missing piece so
        the caller can say exactly why it could not verify.

        reason: None | "unknown_panchayat" | "date_unavailable" | "masked_cell"
        """
        latlon = self.coords.get(int(panchayat_id))
        if latlon is None:
            return {"rainfall_mm": None, "reason": "unknown_panchayat", "source": None,
                    "cell": None, "lat": None, "lon": None, "location_precision": None}
        lat, lon = latlon
        if date not in self.grid_dates:
            return {"rainfall_mm": None, "reason": "date_unavailable", "source": None,
                    "cell": None, "lat": lat, "lon": lon, "location_precision": "exact"}
        try:
            r = self.grid_rainfall(lat, lon, date)
        except (DataUnavailable, OutsideDomain):
            return {"rainfall_mm": None, "reason": "masked_cell", "source": None,
                    "cell": None, "lat": lat, "lon": lon, "location_precision": "exact"}
        return {"rainfall_mm": float(r["rainfall_mm"]), "reason": None,
                "source": r["source"], "cell": [r["fine_i"], r["fine_j"]],
                "lat": lat, "lon": lon, "location_precision": "exact"}


# --------------------------------------------------------------------------
# Cached access (one Store per resolved repo root)
# --------------------------------------------------------------------------
_STORES: dict[str, Store] = {}


def store_for_root(repo_root: str | Path) -> Store:
    """Reuse one Store per repo root. The artefacts are read-only, so sharing is
    safe and avoids re-reading the summary CSV on every request."""
    key = str(Path(repo_root).resolve())
    if key not in _STORES:
        _STORES[key] = Store(key)
    return _STORES[key]
