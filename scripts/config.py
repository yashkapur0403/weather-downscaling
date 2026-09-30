"""Central configuration for the Layer-1 rainfall downscaling prototype."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---- Study regions ---------------------------------------------------------
# Each region: a contiguous study box + metadata. The pipeline is region-aware:
# pass --region <name> to any script (default: deccan).
#
# 'western_ghats' (pilot, retained): small ROI over the Western Ghats
#   (Maharashtra-Goa-Karnataka). Strong orographic rainfall gradient ->
#   elevation carries real signal. NOTE: IMD is land-only (Arabian Sea cells
#   are missing), so the pilot ROI starts at 13N to stay on land; a 12N box
#   would be ~13% missing every day.
# 'deccan' (expanded, default): contiguous Peninsular-India box spanning
#   Maharashtra, N Karnataka, Telangana and S Andhra Pradesh (the existing
#   Western Ghats pilot sits inside it). Bounded by ~25.5N (Vindhya/Satpura
#   rim) and ~11.5N (S Karnataka plateau edge), 71.5-81.5E (Arabian Sea
#   coast to the eastern plateau/Cauvery basin). The box is snapped to the
#   IMD 0.25-deg lattice so imd_lat/imd_lon are exact cell centers.
#   Land fraction ~91% of the 56x40 = 2240 coarse cells (sea margin at the
#   SW/W edge); interior samples drop only land cells.
REGIONS = {
    "western_ghats": {
        "lat_min": 13.0, "lat_max": 17.0,
        "lon_min": 74.25, "lon_max": 78.25,
    },
    "deccan": {
        "lat_min": 11.50, "lat_max": 25.50,
        "lon_min": 71.50, "lon_max": 81.25,
    },
}
REGION_DEFAULT = "deccan"

# ---- Time period -----------------------------------------------------------
# Five consecutive monsoon windows (Jun 1 - Sep 30, 2018-2022): the longest
# window for which IMD 0.25, CHIRPS v2.0 0.05 and ERA5-Land (Open-Meteo
# archive) all overlap consistently. IMD 2023 dropped 1 day (2023-07-11) in
# the official archive, breaking strict daily alignment; 2018-2022 is clean.
# Monsoon-only design decision: the pilot targeted the SW monsoon; soil
# moisture and vegetation fields (aux layers) are monsoon-relevant too.
YEARS = [2018, 2019, 2020, 2021, 2022]
MONSOON_START = "-06-01"   # appended to each year
MONSOON_END = "-09-30"
START_DATE_DEFAULT = "2018-06-01"
END_DATE_DEFAULT = "2022-09-30"
# Months of the year to download CHIRPS for (monsoon study design). Keeps the
# 5-year window at 20 monthly files (~2.1 GB) instead of 60 (~6.2 GB).
CHIRPS_MONTHS = [6, 7, 8, 9]

# ---- Time-based split ------------------------------------------------------
# Year-based: strict temporal holdout, no leakage. Override with --split-years
# on preprocess.py. 2018 is a wider (deccan) training year; the 2019/2021/2022
# scheme matches the published pilot results.
SPLIT_YEARS = {"train": [2018, 2019, 2020], "val": [2021], "test": [2022]}
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
# NOTE: era5_wind is available in the raw caches (0% missing, 2018-2022, incl.
# 2018 after the re-fetch) but is intentionally NOT in CHANNELS_ALL: the frozen
# deccan Layer-1 dataset was preprocessed with the 5-channel baseline
# (imd_rain, dem, era5_t2m, era5_t2m_max, era5_dewp). Do NOT change this list
# unless you rebuild data/processed; era5_wind can be evaluated only after a
# documented re-preprocess (see docs/HANDOVER.md / README limitations).

# ---- ERA5-Land auxiliary data (via Open-Meteo archive API, no auth) --------
# Hourly ERA5-Land aggregated to daily means/max server-side (physically
# sensible aggregation, Asia/Kolkata days); sampled on a degree lattice with
# a 1-cell margin (bilinearly resampled onto the fine grid in preprocess.py).
# Wind comes from ERA5 (not Land) - ERA5-Land lacks 10m wind.
# QUOTA DESIGN: the free archive API rations request weight by
# (locations x days). Two levers keep deccan inside it:
#   1. monsoon-only windows (Jun 1 - Sep 30 per year = the study design);
#   2. a 1.0-deg sampling lattice for deccan (0.5-deg for the pilot ROI).
# Temperature/dewpoint/wind daily means are smooth synoptic fields; the
# coarser lattice keeps their large-scale signal (documented in meta.json).
# Per-year caches make re-running safe: only missing year/model pairs are
# fetched, and holdout years are fetched first.
ERA5_API = "https://archive-api.open-meteo.com/v1/archive"
ERA5_DAILY_VARS = {
    "era5_land": "temperature_2m_mean,temperature_2m_max,dew_point_2m_mean",
    "era5": "wind_speed_10m_mean",
}
ERA5_TIMEZONE = "Asia/Kolkata"
ERA5_LOC_BATCH = 64  # coordinates per request (URL-length safety)
ERA5_STEP_DEG = {"western_ghats": 0.5, "deccan": 1.0}   # lattice step per region
ERA5_REQUEST_SLEEP = 20  # seconds between API calls (rate-limit backoff)

# ---- Auxiliary static/temporal layers (Layer-2 readiness) ------------------
# Soil: SoilGrids 250 m v2.0 (ISRIC) via the official REST point service,
# sampled on the 0.05-deg fine grid (depth 5-15 cm, batched; per-batch cache
# -> restartable). Stored as AUX layers, NOT U-Net channels (the current
# architecture is not designed for them; documented in README).
SOILGRID_API = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SOILGRID_PROPS = ("sand", "clay", "ocd", "phh2o", "bdod")
SOILGRID_DEPTH = "5-15cm"
SOIL_BATCH = 64          # coarse cells per REST batch
SOIL_WORKERS = 16        # parallel point queries per batch
# Vegetation: NOAA CDR VIIRS NDVI 0.05-deg (daily global NetCDF archive).
# Monthly composites from 2 sampled days (5th & 25th) x Jun-Sep 2018-2022.
NDVI_DAYS = (5, 25)
NDVI_WORKERS = 4         # parallel file downloads per month
# LULC: ESA WorldCover v200 (2021, 10 m, WGS84 COG tiles on S3):
# direct tile download + area-fraction aggregation onto the 0.05-deg grid.
WORLDCOVER_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
                  "v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")

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
RAW_ERA5 = ROOT / "data" / "raw" / "era5"
RAW_SOIL = ROOT / "data" / "raw" / "soil"
RAW_NDVI = ROOT / "data" / "raw" / "vegetation"
RAW_LULC = ROOT / "data" / "raw" / "lulc"
RAW_ADMIN = ROOT / "data" / "raw" / "admin"
PROCESSED = ROOT / "data" / "processed"
# NOTE: the folder is `aux_data` (not `aux`) because "aux" is a reserved
# device name on Windows; git for Windows cannot open files under data/aux/.
AUX = ROOT / "data" / "aux_data"
REPORTS = ROOT / "data" / "reports"
MODELS = ROOT / "models"
OUT_MAPS = ROOT / "outputs" / "maps"
OUT_METRICS = ROOT / "outputs" / "metrics"
OUT_FIGS = ROOT / "outputs" / "figures"

for _d in (RAW_IMD, RAW_CHIRPS, RAW_DEM, RAW_ERA5, RAW_SOIL, RAW_NDVI,
           RAW_LULC, RAW_ADMIN, PROCESSED, AUX, REPORTS, MODELS,
           OUT_MAPS, OUT_METRICS, OUT_FIGS):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Normalization (statistics computed from TRAIN split only) -------------
RAIN_MAX = 100.0   # mm/day clip before scaling; >99.9% of India daily rain
ELEV_MAX = 3000.0  # m clip; covers the Deccan box (Western Ghats max ~2695 m)

# ---- Heavy-rain weighting (loss='weighted' experiment) ---------------------
WEIGHT_RAIN_MM = 25.0  # pixels with target >= this get extra weight
WEIGHT_MULT = 3.0      # weight multiplier for heavy-rain pixels


def region_roi(name: str | None = None) -> dict:
    """ROI dict for a named region (default REGION_DEFAULT)."""
    key = name or REGION_DEFAULT
    if key not in REGIONS:
        raise ValueError(f"unknown region {key!r}; available: {sorted(REGIONS)}")
    return dict(REGIONS[key])


def clip_roi(kwargs: dict) -> dict:
    """Merge CLI overrides over the default ROI.

    Priority: explicit --lat/--lon > --region > REGION_DEFAULT.
    """
    if any(kwargs.get(k) is not None
           for k in ("lat_min", "lat_max", "lon_min", "lon_max")):
        roi = region_roi(kwargs.get("region"))
    elif kwargs.get("region"):
        roi = region_roi(kwargs["region"])
    else:
        roi = region_roi(REGION_DEFAULT)
    for k in ("lat_min", "lat_max", "lon_min", "lon_max"):
        if kwargs.get(k) is not None:
            roi[k] = float(kwargs[k])
    return roi
