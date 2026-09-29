"""Layer-2 config shim: ACTIVE (deccan) Layer-1 pipeline + app output paths.

Why this exists
---------------
This folder (`weather-downscaling-main/scripts/`) is the app copy of the OLD
Western-Ghats *pilot* (85x85, 122-day) code. The pilot `config.py` that sits
beside this file is preserved as-is for historical reference.

The three Layer-2 scripts in this folder are the ONLY scripts here that are
still part of the live pipeline:

    layer2_panchayat_mapping.py
    check_blocks.py
    export_pickle.py

They used to `import config` (-> the pilot config). Instead they now do
`import layer2_config as config`, so they run against the CURRENT root pipeline
(repo-root `scripts/config.py`, region "deccan", 285x200 fine grid,
`data/processed`, `prediction/`, `outputs/maps/`) while writing their Layer-2
results back into THIS app folder, where the backend reads them.

Inputs  -> repo root (`weather-downscaling/`)          : current Deccan run
Outputs -> this app folder (`weather-downscaling-main/`) : backend contract
           * OUT_LAYER2 = .../outputs/layer2/panchayat_weather.csv  <- backend/app.py
           (`layer1_model.pkl` / `prediction/*.pkl` have no consumer; they
            follow the root pipeline, i.e. `MODELS`/`PRED_DIR`.)

LGD note: `RAW_PANCHAYAT` (LGD_Panchayats.parquet) is NOT present yet — the LGD
panchayat tier is documented future work (see HANDOVER.md section 4.2). Layer 2
therefore still needs that file (or `--panchayats`/`--input`) to produce a CSV.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]   # weather-downscaling-main/
REPO_ROOT = APP_ROOT.parent                      # weather-downscaling/
_ROOT_SCRIPTS = REPO_ROOT / "scripts"            # ACTIVE pipeline (deccan)

if not (_ROOT_SCRIPTS / "config.py").exists():
    raise ImportError(
        f"active root config not found at {_ROOT_SCRIPTS / 'config.py'}; "
        "weather-downscaling-main/ must stay inside the weather-downscaling repo"
    )

# Load the root config by file path so it is never shadowed by the pilot
# config.py sitting next to this module.
_spec = importlib.util.spec_from_file_location("_root_config",
                                               _ROOT_SCRIPTS / "config.py")
_root = importlib.util.module_from_spec(_spec)
sys.modules["_root_config"] = _root
_spec.loader.exec_module(_root)

# Re-export the whole active config (REGIONS, REGION_DEFAULT, FINE_SUB, IMD_STEP,
# PROCESSED, RAW_*, CHANNELS_*, MODELS, OUT_MAPS, region_roi, clip_roi, ...).
globals().update({k: v for k, v in vars(_root).items() if not k.startswith("__")})

# Make the ACTIVE scripts dir importable FIRST, so `from grids import fine_grid`
# resolves to the current Deccan grids.py (not the pilot one in this folder).
if str(_ROOT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_ROOT_SCRIPTS))

# ---- Layer-2 only symbols (the root Layer-1 config does not define these) ---
ROI_DEFAULT = _root.region_roi(_root.REGION_DEFAULT)   # deccan ROI (11.5-25.5N)
OUT_LAYER2 = APP_ROOT / "outputs" / "layer2"           # backend contract path
RAW_PANCHAYAT = REPO_ROOT / "data" / "raw" / "administrative" / "panchayat"
PRED_DIR = REPO_ROOT / "prediction"                    # root infer.py output
WET_DAY_MM = 1.0            # mm/day threshold for a "wet" Panchayat-day
MAX_FALLBACK_KM = 15.0      # nearest-centroid GPs farther than this stay unmapped
LAYER2_BBOX_BUF_DEG = 0.15  # ~16 km search buffer around Panchayat bounds

for _d in (OUT_LAYER2, RAW_PANCHAYAT, PRED_DIR):
    _d.mkdir(parents=True, exist_ok=True)
