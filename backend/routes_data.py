"""
Data routes the Next.js frontend calls (Frontend/src/api/backend.ts). Registered on the XAI app by app.py.

  GET  /                   health
  GET  /api/panchayats     search (real CSV rows: mapping_method / n_cells are the pipeline's own)
  GET  /api/geocode        coordinates + how precise they are
  POST /auth/              projection: rainfall (real model output) + real temperature/humidity/elevation
  GET  /api/metrics        real numbers from outputs/metrics/
  GET  /api/weather        real temperature/humidity/elevation for lat/lon/date

If a value cannot be obtained from a real source the route says so (HTTP 4xx/5xx or null) - it never invents one.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from data_store import DEFAULT_DATE, Store
from live_infer import DataUnavailable, OutsideDomain
from metrics_loader import load_metrics
from weather_lookup import LocalDEM, fetch_weather

log = logging.getLogger("data-routes")


class QueryRequest(BaseModel):
    state: Optional[str] = None
    district: Optional[str] = None
    block_name: Optional[str] = None
    panchayat_name: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    date: Optional[str] = None
    requested_metrics: list[str] = []
    raw_location_text: Optional[str] = None


def unet_param_count(cin: int, width: int) -> int:
    """Parameter count of scripts/train.py::SmallUNet (checked by a test against the formula in the code)."""
    conv = lambda a, b, k=3: k * k * a * b + b
    block = lambda a, b: conv(a, b) + conv(b, b)
    w = width
    return (block(cin, w) + block(w, 2 * w) + block(2 * w, 4 * w)
            + (2 * 2 * 4 * w * 2 * w + 2 * w) + block(4 * w, 2 * w)
            + (2 * 2 * 2 * w * w + w) + block(2 * w, w) + conv(w, 1, 1))


class State:
    store: Optional[Store] = None
    runner = None
    runner_error: Optional[str] = None
    dem: Optional[LocalDEM] = None
    client: Optional[httpx.AsyncClient] = None


def register(app: FastAPI, repo_root: str | Path | None = None) -> None:
    root = Path(repo_root or os.getenv("REPO_ROOT") or Path(__file__).resolve().parents[1]).resolve()
    S = State()

    def store() -> Store:
        if S.store is None:
            try:
                S.store = Store(root)
            except FileNotFoundError as e:
                raise HTTPException(503, str(e))
        return S.store

    def resolve_panchayat_rainfall(panchayat_id: int, date: str) -> dict:
        """Authoritative Layer-1 rainfall for one Panchayat and date.

        Published on `app.state` so routes that do not own the artefacts (the
        Layer-3 advisory in main.py) can VERIFY a caller's claim instead of
        trusting it. Raised as 503 when the artefacts are absent, because then
        nothing can be verified.
        """
        return store().panchayat_grid_value(int(panchayat_id), date)

    # The data layer is the only place that knows where the artefacts live; the
    # advisory route asks for it through this hook (see main._rainfall_resolver).
    app.state.rainfall_resolver = resolve_panchayat_rainfall

    def client() -> httpx.AsyncClient:
        if S.client is None:
            S.client = httpx.AsyncClient()
        return S.client

    def dem() -> Optional[LocalDEM]:
        p = root / "data" / "raw" / "dem" / "dem_roi.npz"
        if S.dem is None and p.exists():
            S.dem = LocalDEM(p)
        return S.dem

    def runner():
        """Optional live U-Net inference (needs torch + data/raw). Off unless ENABLE_LIVE_INFERENCE=1."""
        if os.getenv("ENABLE_LIVE_INFERENCE", "0") != "1" or S.runner_error:
            return None
        if S.runner is None:
            try:
                from live_infer import ModelRunner
                S.runner = ModelRunner(root)
            except Exception as e:                       # torch / data missing: report once, keep serving
                S.runner_error = f"{type(e).__name__}: {e}"
                log.warning("live inference disabled: %s", S.runner_error)
                return None
        return S.runner

    @app.get("/")
    async def health():
        try:
            n = len(store().df)
        except HTTPException:
            n = None                    # data files missing: report null rather than crash
        runner()                        # surface a requested-but-failed live-inference init
        return {"message": "Panchayat downscaling API", "repo_root": str(root),
                "panchayats_loaded": n,
                "live_inference": bool(S.runner), "live_inference_error": S.runner_error}

    @app.get("/api/panchayats")
    async def search_panchayats(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=50)):
        return store().search(q, limit)

    @app.get("/api/geocode")
    async def geocode(panchayat_name: str, block_name: str = "", district: str = "", state: str = ""):
        st = store()
        h = st.matches(panchayat_name, block_name, district, state)
        if len(h) == 0:
            raise HTTPException(404, "panchayat not found")
        if len(h) > 1:
            raise HTTPException(409, {
                "error": f"{len(h)} panchayats share the name '{panchayat_name}'; "
                         "add district / block_name (or state) to disambiguate",
                "candidates": Store.candidate_dicts(h)})
        lat, lon, prec = st.locate_row(h.iloc[0])
        if lat is None:
            raise HTTPException(404, "no coordinates for this panchayat. Run build_panchayat_index.py to add them")
        return {"lat": lat, "lon": lon, "location_precision": prec}

    @app.get("/api/weather")
    async def weather(lat: float, lon: float, date: str = DEFAULT_DATE):
        return (await fetch_weather(lat, lon, date, client(), dem())).as_dict()

    @app.get("/api/metrics")
    async def metrics():
        try:
            m = load_metrics(root / "outputs" / "metrics")
        except FileNotFoundError as e:
            raise HTTPException(503, str(e))
        hist = root / "models" / "best_model_history.json"
        if hist.exists():
            import json
            a = json.loads(hist.read_text()).get("args", {})
            m["n_parameters"] = f"{unet_param_count(len(a.get('channels', m['channels'])), int(a.get('width', 16))):,}"
        return m

    @app.post("/auth/")
    @app.post("/auth")
    async def query(req: QueryRequest):
        st = store()
        date = req.date or DEFAULT_DATE
        h = st.matches(req.panchayat_name, req.block_name, req.district, req.state)
        if len(h) > 1 and req.lat is None and req.lon is None:
            raise HTTPException(409, {
                "error": f"{len(h)} panchayats share the name '{req.panchayat_name}'; "
                         "add district / block_name (or state), or pass lat/lon, to disambiguate",
                "candidates": Store.candidate_dicts(h)})
        row = h.iloc[0] if len(h) else None
        lat, lon = req.lat, req.lon
        precision = "given"
        if row is not None:
            lat0, lon0, prec0 = st.locate_row(row)
            if lat is None or lon is None:
                lat, lon, precision = lat0, lon0, prec0
            elif lat0 is not None and abs(lat - lat0) < 1e-6 and abs(lon - lon0) < 1e-6:
                precision = prec0            # the frontend echoed our own geocode result back: keep its precision
        if precision in ("block", "district") and os.getenv("REQUIRE_EXACT_LOCATION", "0") == "1":
            raise HTTPException(422, f"only a {precision}-level location is known for this panchayat; "
                                     "run build_panchayat_index.py (needs the LGD parquet) for exact placement")

        rain: Optional[float] = None
        source = ""
        if row is not None:                                  # 1) exact Layer-2 value for this panchayat + date
            v = st.layer2_value(int(row.panchayat_id), date)
            if v is not None:
                rain, source = round(v, 2), "Layer-2 panchayat_weather.csv (area-weighted)"
        if rain is None and lat is not None and lon is not None:
            try:
                g = st.grid_rainfall(lat, lon, date)         # 2) real U-Net output grid
                if g is None and runner() is not None:       # 3) live inference for other dates
                    g = runner().rainfall_at(lat, lon, date)
                    g["source"] = f"live U-Net inference ({g['checkpoint']})"
                if g is not None:
                    rain, source = g["rainfall_mm"], g["source"]
            except OutsideDomain as e:
                raise HTTPException(422, str(e))
            except DataUnavailable as e:
                raise HTTPException(422, str(e))
        if rain is None:
            why = ("no coordinates for this panchayat (run build_panchayat_index.py)" if lat is None
                   else f"no model output for {date}")
            dates = st.grid_dates
            avail = f" Available dates: {dates[0]} to {dates[-1]}." if dates else ""
            raise HTTPException(422, f"Cannot produce a rainfall value: {why}.{avail}")

        w = (await fetch_weather(lat, lon, date, client(), dem())).as_dict() if lat is not None else {}
        who = req.panchayat_name or req.raw_location_text or "this location"
        caveat = "" if precision in ("exact", "given") else (
            f" Location is approximate ({precision} centre): this is the model value at that point, not for the "
            f"panchayat polygon, so panchayats in the same {precision} get the same number.")
        return {
            "params": req.model_dump(),
            "prediction": {"rainfall_mm": rain, "temperature_c": w.get("temperature_c"),
                           "humidity_pct": w.get("humidity_pct"), "elevation_m": w.get("elevation_m")},
            "answer": f"Estimated rainfall for {who} on {date}: {rain:.1f} mm/day (source: {source}).{caveat}",
            "model_status": "ready",
            "sources": {"rainfall": source, "temperature": w.get("temperature_source"),
                        "humidity": w.get("humidity_source"), "elevation": w.get("elevation_source"),
                        "location_precision": precision, "weather_error": w.get("error")},
        }
