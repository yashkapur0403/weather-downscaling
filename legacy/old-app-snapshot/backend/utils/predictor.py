"""
utils/predictor.py
-------------------
Turns the structured params passed in directly by the caller into a numeric
prediction dict: {rainfall_mm, temperature_c, humidity_pct, elevation_m, ...}.

Resolution order for rainfall:
  1. Look up a cached Layer-2 Panchayat value (outputs/layer2/panchayat_weather.csv,
     or the .pkl snapshot format shown in panchayat_weather.pkl) for the
     requested location + date, if one already exists.
  2. Otherwise, fall back to live Layer-1 model inference via ModelLoader
     (stubbed here -- see `_infer_from_model`, wire up grids.py's fine-grid
     builder from the original repo to make this real).

Temperature / humidity / elevation are context features (ERA5-Land inputs /
DEM), not model outputs. Until the raw stores are deployed, stable
location-seeded demo values are returned for these fields.
"""

from __future__ import annotations

import logging
import os
import pickle
import hashlib
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .model_loader import ModelLoader

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(
        self,
        model_loader: ModelLoader,
        layer2_csv: str = None,
        layer2_pickle: Optional[str] = None,
        model_wait_timeout: float = 15.0,
    ):
        self.model_loader = model_loader
        # Use passed parameters or environment variables with fallbacks
        self.layer2_csv_path = Path(layer2_csv or os.environ.get("LAYER2_CSV_PATH", "outputs/layer2/panchayat_weather.csv"))
        pickle_env = os.environ.get("LAYER2_PICKLE_PATH")
        self.layer2_pickle_path = Path(layer2_pickle or pickle_env) if (layer2_pickle or pickle_env) else None
        self.model_wait_timeout = model_wait_timeout
        self._df_cache: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def predict(self, params: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "rainfall_mm": None,
            "temperature_c": None,
            "humidity_pct": None,
            "elevation_m": None,
            "source": None,          # "layer2_cache" | "model_inference" | "unavailable"
            "mapping_method": None,
            "matched_location": None,
        }

        df = self._load_layer2()
        row = self._match_row(df, params) if df is not None else None

        if row is not None:
            result["rainfall_mm"] = _safe_float(row.get("rainfall_mm"))
            result["mapping_method"] = row.get("mapping_method")
            result["matched_location"] = row.get("panchayat_name") or row.get("block_name")
            result["source"] = "layer2_cache"
        else:
            inferred = self._infer_from_model(params)
            if inferred is not None:
                result["rainfall_mm"] = inferred
                result["source"] = "model_inference"
            else:
                result["source"] = "unavailable"

        requested = set(params.get("requested_metrics") or [])
        context_key = "|".join(
            str(params.get(field) or "")
            for field in ("state", "district", "panchayat_name")
        )
        if "temperature" in requested:
            result["temperature_c"] = self._lookup_era5_temperature(params, context_key)
        if "humidity" in requested:
            result["humidity_pct"] = self._lookup_era5_humidity(params, context_key)
        if "elevation" in requested:
            result["elevation_m"] = self._lookup_dem_elevation(params, context_key)

        return result

    # ------------------------------------------------------------------
    def _load_layer2(self) -> Optional[pd.DataFrame]:
        if self._df_cache is not None:
            return self._df_cache

        if self.layer2_csv_path.exists():
            # Check if 'date' column exists before trying to parse it
            try:
                self._df_cache = pd.read_csv(self.layer2_csv_path, parse_dates=["date"])
            except ValueError:
                # If 'date' column doesn't exist, load without date parsing
                self._df_cache = pd.read_csv(self.layer2_csv_path)
            return self._df_cache

        if self.layer2_pickle_path and self.layer2_pickle_path.exists():
            with open(self.layer2_pickle_path, "rb") as f:
                payload = pickle.load(f)
            df = payload["dataframe"] if isinstance(payload, dict) and "dataframe" in payload else payload
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            self._df_cache = df
            return self._df_cache

        logger.warning(
            "No Layer-2 data found at %s or %s; predictions will rely on live model inference.",
            self.layer2_csv_path, self.layer2_pickle_path,
        )
        return None

    @staticmethod
    def _match_row(df: pd.DataFrame, params: dict[str, Any]):
        filtered = df
        field_map = [
            ("state", "state"),
            ("district", "district"),
            ("block_name", "block_name"),
            ("panchayat_name", "panchayat_name"),
        ]
        for param_key, col in field_map:
            value = params.get(param_key)
            if value and col in filtered.columns:
                filtered = filtered[filtered[col].astype(str).str.upper() == str(value).upper()]

        # Only filter by date if the column exists
        if params.get("date") and "date" in filtered.columns:
            target_date = pd.to_datetime(params["date"])
            filtered = filtered[filtered["date"] == target_date]

        if filtered.empty:
            return None
        return filtered.iloc[0]

    # ------------------------------------------------------------------
    def _infer_from_model(self, params: dict[str, Any]) -> Optional[float]:
        loader = self.model_loader
        loader.wait_until_ready(timeout=self.model_wait_timeout)
        if loader.model is None:
            logger.info("Model not available (load_error=%s); cannot run live inference.",
                        loader.load_error)
            return None

        # TODO: build the fine-grid input tensor for the requested lat/lon + date
        # using the original repo's `grids.py` helpers (bilinear-upsampled IMD
        # rainfall, DEM, and any ERA5-Land channels the checkpoint expects, in
        # `loader.channels` order), then run:
        #   with torch.no_grad():
        #       residual = loader.model(input_tensor)
        #       rainfall_field = baseline_channel + residual
        # and sample the pixel(s) covering the requested panchayat.
        logger.info("Live model inference is stubbed -- wire up grids.py to enable it.")
        return None

    # ------------------------------------------------------------------
    def _lookup_era5_temperature(self, params: dict[str, Any], context_key: str) -> float:
        # Replace the seeded fallback with a nearest-grid lookup when the
        # ERA5-Land raster store is included in the deployment.
        return _synthetic_context(context_key)[0]

    def _lookup_era5_humidity(self, params: dict[str, Any], context_key: str) -> int:
        # Replace the seeded fallback with ERA5-Land humidity when available.
        return _synthetic_context(context_key)[1]

    def _lookup_dem_elevation(self, params: dict[str, Any], context_key: str) -> int:
        # Replace the seeded fallback with a DEM sample when available.
        return _synthetic_context(context_key)[2]


def _safe_float(value) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _synthetic_context(key: str) -> tuple[float, int, int]:
    """Return stable demo context values until source rasters are deployed."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    temperature = round(18 + (digest[0] / 255) * 20, 1)
    humidity = 45 + round((digest[1] / 255) * 50)
    elevation = 20 + round((digest[2] / 255) * 1780)
    return temperature, humidity, elevation