"""Compatibility Layer-2 configuration.

This file reconstructs the Layer-2 config variables expected by the team's
scripts from the current repo's canonical settings in scripts/config.py.
It keeps the same names and paths used by the Layer-2 mapping workflow while
remaining compatible with the shipped project layout.
"""

from __future__ import annotations

from pathlib import Path

import config as _base_config  # type: ignore

ROOT: Path = _base_config.ROOT

# Canonical project paths
RAW_ADMIN: Path = _base_config.RAW_ADMIN
RAW_PANCHAYAT: Path = (_base_config.RAW_ADMIN.parent / "administrative" / "panchayat")
AUX_ADMIN: Path = _base_config.AUX / "admin"
OUT_LAYER2: Path = (_base_config.ROOT / "outputs" / "layer2")
PRED_DIR: Path = (_base_config.ROOT / "prediction")

# Admin / block mapping
ADMIN_MASTER_PATH: Path = AUX_ADMIN / "admin_master_deccan.geojson"
GRID_ADMIN_MAP_PATH: Path = AUX_ADMIN / "grid_admin_map_deccan.npz"

# Default geographic context used by the Layer-2 workflow
ROI_DEFAULT: dict[str, float] = _base_config.region_roi(_base_config.REGION_DEFAULT)
LAYER2_BBOX_BUF_DEG: float = 0.1
METRIC_CRS: str = "EPSG:32643"

# Fallback / QC thresholds
MAX_FALLBACK_KM: float = 20.0
WET_DAY_MM: float = 2.5

# Common aliases expected by downstream scripts
RAW_GADM: Path = RAW_ADMIN
OUTPUT_DIR: Path = OUT_LAYER2
DEFAULT_OUT: Path = OUT_LAYER2
DEFAULT_PARQUET: Path = RAW_PANCHAYAT / "LGD_Panchayats.parquet"

__all__ = [
    "ROOT",
    "RAW_ADMIN",
    "RAW_GADM",
    "RAW_PANCHAYAT",
    "AUX_ADMIN",
    "ADMIN_MASTER_PATH",
    "GRID_ADMIN_MAP_PATH",
    "OUT_LAYER2",
    "OUTPUT_DIR",
    "DEFAULT_OUT",
    "DEFAULT_PARQUET",
    "PRED_DIR",
    "ROI_DEFAULT",
    "LAYER2_BBOX_BUF_DEG",
    "METRIC_CRS",
    "MAX_FALLBACK_KM",
    "WET_DAY_MM",
]
