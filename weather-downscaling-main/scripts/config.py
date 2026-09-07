"""Central configuration for the Layer-1 rainfall downscaling prototype."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- Study region ----------------------------------------------------------
# Default: small ROI over the Western Ghats (Maharashtra-Goa-Karnataka).
# Strong orographic rainfall gradient -> elevation carries real signal.
# NOTE: IMD is land-only (Arabian Sea cells are missing), so the default ROI
# starts at 13N to stay on land; a 12N box would be ~13% missing every day.
# Override with --lat-min etc. on any script.
ROI_DEFAULT = {
    "lat_min": 13.0, "lat_max": 17.0,
    "lon_min": 74.25, "lon_max": 78.25,
}

# ---- Time period -----------------------------------------------------------
# Multi-year monsoon window (Jun 1 - Sep 30 each year). IMD + CHIRPS + ERA5
# all cover these years; strict time-based split: earlier years train,
# 2021 val, 2022 test (never shuffled across time).
YEARS = [2019, 2020, 2021, 2022]
MONSOON_START = "-06-01"   # appended to each year
MONSOON_END = "-09-30"
START_DATE_DEFAULT = "2019-06-01"
END_DATE_DEFAULT = "2022-09-30"
# Months of the year to download CHIRPS for (monsoon study design). Keeps the
# 4-year window at 16 monthly files (~1.6 GB) instead of 40 (~4 GB).
CHIRPS_MONTHS = [6, 7, 8, 9]

# ---- Time-based split ------------------------------------------------------
# Year-based: strict temporal holdout, no leakage. Override with --split-years
# "2019,2020,2021,2022" (train years, val years, test years) on preprocess.py.
SPLIT_YEARS = {"train": [2019, 2020], "val": [2021], "test": [2022]}
# Legacy fraction-based split (only used if --split given explicitly)
SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15}

# ---- Input channels --------------------------------------------------------
# Model input = stack of these channels on the fine grid. Order matters for
# checkpoint compatibility. Ablations select a subset by name.
#   imd_rain      : IMD 0.25 bilinearly upsampled 5x (also the bilinear baseline)
#   dem           : SRTM elevation resampled to the fine grid
#   era5_t2m      : ERA5-Land daily mean 2m temperature (C, coarse grid upsampled)
#   era5_t2m_max  : ERA5-Land daily max 2m temperature (C)
#   era5_dewp     : ERA5-Land daily mean 2m dewpoint (C) -> humidity proxy
#   era5_wind     : ERA5 daily mean 10m wind speed (km/h; ERA5-Land has no wind)
CHANNELS_ALL = ["imd_rain", "dem", "era5_t2m", "era5_t2m_max",
                "era5_dewp"]
# Default channels used by train.py/evaluate.py if not overridden:
CHANNELS_DEFAULT = CHANNELS_ALL
# NOTE: era5_wind is intentionally dropped for now - the 2019/2021 wind fetches
# hit the Open-Meteo quota and are 100% missing in the combined file. Once a
# re-run of download_or_export.py back-fills those year caches, the channel can
# be re-added here and trained as an extra ablation variant.

# ---- ERA5-Land auxiliary data (via Open-Meteo archive API, no auth) --------
# Hourly ERA5-Land aggregated to daily means/max server-side (physically
# sensible aggregation, Asia/Kolkata days); sampled nearest-neighbour at IMD
# coarse cell centers. Wind comes from ERA5 (not Land) - ERA5-Land lacks 10m wind.
ERA5_API = "https://archive-api.open-meteo.com/v1/archive"
ERA5_DAILY_VARS = {
    "era5_land": "temperature_2m_mean,temperature_2m_max,dew_point_2m_mean",
    "era5": "wind_speed_10m_mean",
}
ERA5_TIMEZONE = "Asia/Kolkata"
ERA5_LOC_BATCH = 64  # coordinates per request (URL-length safety)

# ---- Grids -----------------------------------------------------------------
IMD_STEP = 0.25
FINE_SUB = 5              # fine grid = 5x5 sub-points per IMD cell (0.05 deg)
                          # -> scale factor 5 in each dimension, as specified
MISSING = -999.0          # IMD missing value
MISSING_TOL = 0.05        # drop a sample if >5% of pixels are missing

# ---- Data locations --------------------------------------------------------
RAW_IMD = ROOT / "data" / "raw" / "imd"
RAW_CHIRPS = ROOT / "data" / "raw" / "chirps"
RAW_DEM = ROOT / "data" / "raw" / "dem"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUT_MAPS = ROOT / "outputs" / "maps"
OUT_METRICS = ROOT / "outputs" / "metrics"
OUT_FIGS = ROOT / "outputs" / "figures"
OUT_LAYER2 = ROOT / "outputs" / "layer2"
RAW_PANCHAYAT = ROOT / "data" / "raw" / "administrative" / "panchayat"
PRED_DIR = ROOT / "prediction"

# Layer-2 mapping
WET_DAY_MM = 1.0           # mm/day threshold for a "wet" Panchayat-day
MAX_FALLBACK_KM = 15.0     # nearest-centroid GPs farther than this stay unmapped
LAYER2_BBOX_BUF_DEG = 0.15 # ~16 km search buffer around Panchayat bounds

for _d in (RAW_IMD, RAW_CHIRPS, RAW_DEM, PROCESSED, MODELS,
           OUT_MAPS, OUT_METRICS, OUT_FIGS, OUT_LAYER2, PRED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Normalization (statistics computed from TRAIN split only) -------------
RAIN_MAX = 100.0   # mm/day clip before scaling; >99.9% of India daily rain
ELEV_MAX = 3000.0  # m clip; Western Ghats max ~2695 m

# ---- Heavy-rain weighting (loss='weighted' experiment) ---------------------
WEIGHT_RAIN_MM = 25.0  # pixels with target >= this get extra weight
WEIGHT_MULT = 3.0      # weight multiplier for heavy-rain pixels


def clip_roi(kwargs: dict) -> dict:
    """Merge CLI overrides over the default ROI."""
    roi = dict(ROI_DEFAULT)
    for k in ("lat_min", "lat_max", "lon_min", "lon_max"):
        if kwargs.get(k) is not None:
            roi[k] = float(kwargs[k])
    return roi
