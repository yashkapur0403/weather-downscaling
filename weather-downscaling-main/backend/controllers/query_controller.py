from __future__ import annotations

import logging
import os
from fastapi import HTTPException
from pydantic import BaseModel, Field

from backend.utils.model_loader import ModelLoader
from backend.utils.predictor import Predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("panchayat-weather-api")

ALL_METRICS = ["rainfall", "temperature", "humidity", "elevation"]

# Use the singleton instance from app.py
model_loader = ModelLoader.get_instance()
predictor = Predictor(
    model_loader,
    # Environment variables will be used inside Predictor with defaults
)


class QueryRequest(BaseModel):
    state: str | None = Field(None, description="e.g. 'KARNATAKA' (upper case)")
    district: str | None = None
    block_name: str | None = Field(None, description="Administrative block / taluk")
    panchayat_name: str | None = Field(None, description="Gram Panchayat name")
    lat: float | None = Field(None, description="Explicit latitude, if no admin name is given")
    lon: float | None = Field(None, description="Explicit longitude, if no admin name is given")
    date: str | None = Field(None, description="ISO 'YYYY-MM-DD'")
    requested_metrics: list[str] = Field(
        default_factory=lambda: list(ALL_METRICS),
        description="Any of: rainfall, temperature, humidity, elevation",
    )
    raw_location_text: str | None = Field(
        None, description="Optional free-text location label, used only for display/logging"
    )


class QueryResponse(BaseModel):
    params: dict
    prediction: dict
    answer: str
    model_status: str


def _build_answer(params: dict, prediction: dict) -> str:
    location = (
        params.get("panchayat_name")
        or params.get("block_name")
        or params.get("district")
        or params.get("state")
        or params.get("raw_location_text")
        or "the requested location"
    )
    date = params.get("date") or "the requested date"

    rainfall = prediction.get("rainfall_mm")
    if rainfall is None:
        rainfall_line = f"Rainfall data is not available for {location} on {date}."
    else:
        if rainfall < 2.5:
            level = "light"
        elif rainfall < 15:
            level = "moderate"
        else:
            level = "heavy"
        rainfall_line = f"Expected rainfall for {location} on {date} is {rainfall:.1f} mm/day ({level})."

    requested = set(params.get("requested_metrics") or [])
    extra_lines = []
    if "temperature" in requested:
        t = prediction.get("temperature_c")
        extra_lines.append(f"Temperature: {t}°C" if t is not None else "Temperature: not available")
    if "humidity" in requested:
        h = prediction.get("humidity_pct")
        extra_lines.append(f"Humidity: {h}%" if h is not None else "Humidity: not available")
    if "elevation" in requested:
        e = prediction.get("elevation_m")
        extra_lines.append(f"Elevation: {e} m" if e is not None else "Elevation: not available")

    return " ".join([rainfall_line, *extra_lines])


async def handle_query(req: QueryRequest):
    params = req.model_dump()

    try:
        prediction = predictor.predict(params)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    answer = _build_answer(params, prediction)

    if not model_loader.ready.is_set():
        model_status = "loading"
    elif model_loader.model is not None:
        model_status = "ready"
    else:
        model_status = "unavailable"

    return QueryResponse(
        params=params,
        prediction=prediction,
        answer=answer,
        model_status=model_status,
    )