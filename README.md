# Layer 1 — Coarse-to-Fine Weather Downscaling (dataset complete; model validated on pilot)

**What it is:** the coarse-to-fine weather refinement engine for the SIH project.
Coarse IMD 0.25° daily rainfall + SRTM elevation + ERA5-Land daily context →
small residual U-Net → fine 0.05° daily rainfall, evaluated against CHIRPS 0.05°
as the fine-resolution **reference**.

**Current status (be honest when presenting):**

* ✅ **Layer-1 DATASET complete for the Deccan region** (`region="deccan"`):
  5 monsoons (2018–2022), 610 days, 47,250 land cells, model-ready
  `X/Y/M_{train,val,test}.npy`, plus auxiliary layers (admin, soil, NDVI, LULC)
  for Layer-2. End-to-end QA passes (`scripts/verify_dataset.py`).
* ✅ Model **methodology validated on the Western Ghats pilot** (older, smaller
  run: 2019–2022, 122-day splits — §4). That pilot proved the pipeline works
  end-to-end and beat the baseline.
* ❌ **Model training/evaluation on the Deccan dataset is NOT done yet.** The
  numbers in §4 are pilot numbers, not Deccan numbers. `HANDOVER.md` explains
  exactly how to run training next.
* ❌ Layer-2 (Panchayat mapping, RAG, dashboards) is NOT implemented. The data
  Layer-2 needs is prepared and documented (§8).

## 0. File map — what is what, who uses what (read this first)

**New here?** Read this table, then §1–§3 for context, then `HANDOVER.md` for
step-by-step training/Layer-2 instructions.

| Path | What it is | Who consumes it |
|---|---|---|
| `data/processed/X_{train,val,test}.npy` | Model inputs, 5 channels × 285×200 fine grid, normalized (train-only stats) | **training only** (`train.py`) |
| `data/processed/Y_{train,val,test}.npy` | CHIRPS 0.05° daily rain (mm/day) — the reference target | **training only** |
| `data/processed/M_{train,val,test}.npy` | 1 = valid land target, 0 = excluded (sea/coastal). Loss & metrics MUST respect it | **training only** |
| `data/processed/meta.json` | Everything about the dataset: dates, split, grid, channels, land-mask rules, normalization, provenance. Committed so a rebuilt dataset can be diffed against the frozen one | humans + every script |
| `data/aux_data/admin/grid_admin_map_deccan.npz` | Every land cell → state / district / **block** (committed) | **Layer-2** aggregation |
| `data/aux_data/soil_soilgrids_deccan.npz` | Sand/clay/OC/pH/bulk density on land coarse cells (committed) | **Layer-3** agro-advisory |
| `data/aux_data/ndvi_monthly_deccan.npz` | 20 monthly NDVI composites Jun–Sep 2018–2022 (committed) | **Layer-3** agro-advisory |
| `data/aux_data/lulc_fractions_deccan.npz` | 6 land-cover fractions + dominant class per cell (committed) | **Layer-3** + crop context |
| `data/aux_data/build_summary_deccan.json` | Machine-readable build stats for every aux layer (incl. soil-moisture PENDING status) | humans / QA |
| `data/reports/*` | Data dictionary, coverage report, missingness report, verification JSON (committed) | humans — read the dictionary before touching data |
| `data/raw/` | Source downloads (IMD, CHIRPS, DEM, ERA5, GADM, SoilGrids batches, VIIRS slices, WorldCover tiles). NOT in git — **reproduce with §5 commands**; per-year/batch caches make re-runs cheap | only rebuilds |
| `scripts/` | One script per pipeline stage (see §6) — every stage is cached/resumable | the pipeline |
| `models/`, `outputs/`, `prediction/` | Pilot-run checkpoints, metrics, maps; `prediction/*.npz` is the Layer-2 contract format | Layer-2 demo |
| `HANDOVER.md` | **The ops manual**: exact training commands, evaluation rules, Layer-2 recipes, rebuild instructions, all dataset decisions | the person doing training / Layer-2 (you, probably) |

---

## 1. Problem Layer 1 solves

Agricultural advisories need rainfall at ~5 km scale, but IMD's gridded product
is 0.25° (~28 km). Layer 1 learns the mapping

```
IMD 0.25° rainfall (coarse) ─┐
SRTM elevation (DEM)        ─┼─►  small residual U-Net  ─►  0.05° rainfall field
ERA5-Land daily T/Tmax/Td   ─┘            (optional channels)         │
                                                                      ▼
                                                       scored against CHIRPS 0.05°
```

* **Input (X):** channels on the fine grid — bilinearly-upsampled IMD rainfall,
  resampled DEM, and (optionally) ERA5-Land daily mean/max temperature and
  dewpoint, all configurable for ablation.
* **Target (Y):** CHIRPS daily rainfall on the same fine grid — a **reference
  product, not absolute ground truth**.
* **Model:** ~150k-parameter residual U-Net (learns the *correction* to the
  bilinear baseline, so it can never do worse than the baseline by
  construction of the residual parameterization).
* **Loss:** masked MAE (default) or a heavy-rain-weighted MAE (selected by
  validation, see §3).

## 2. Regions and data (all real; no GEE authentication required)

Two named regions live in `scripts/config.py` (`config.REGIONS`):

| region | bbox (lat, lon) | role |
|---|---|---|
| `western_ghats` | 13–17 N, 74.25–78.25 E | pilot used for the §4 results |
| `deccan` (default) | 11.5–25.5 N, 71.5–81.25 E | full Layer-1 dataset, 2018–2022 |

| Dataset | Role | Source | Where it lands |
|---|---|---|---|
| IMD 0.25° daily rainfall | coarse input + land mask | imdpune.gov.in | `data/raw/imd/ind<YEAR>_rfp25.nc` |
| CHIRPS v2.0 0.05° daily | fine reference (Y) | UCSB CHC | `data/raw/chirps/` (monthly) |
| SRTM 30 m (Terrarium) | DEM channel | AWS Open Data tiles | `data/raw/dem/dem_roi_<region>.npz` |
| ERA5-Land daily T/Tmax/dewpoint | auxiliary channels | Open-Meteo archive API | `data/raw/era5/era5_daily_<region>.npz` (per-year caches) |
| GADM 4.1 admin polygons | **aux**: state/district/block mapping | gadom.org | `data/raw/admin/` → `data/aux_data/admin/` |
| SoilGrids v2.0 (ISRIC) | **aux**: sand/clay/OC/pH/bulk density | rest.isric.org | `data/raw/soil/batches/` → `data/aux_data/soil_soilgrids_<region>.npz` |
| NOAA CDR VIIRS NDVI | **aux**: vegetation composites | NCEI | `data/raw/vegetation/slices/` → `data/aux_data/ndvi_monthly_<region>.npz` |
| ESA WorldCover 2021 | **aux**: land-cover fractions | S3 COG tiles | `data/raw/lulc/` → `data/aux_data/lulc_fractions_<region>.npz` |
| ERA5-Land soil moisture (0–7 cm) | **aux (OPTIONAL, PENDING)**: daily volumetric water (Layer-3) | Open-Meteo archive API (same source as ERA5 channels) | builder in `build_aux.py`; resume: `python scripts/build_aux.py --region deccan --skip-admin --skip-soil --skip-ndvi --skip-lulc` |

Aux layers are **not** U-Net input channels — they exist for Layer-2
(agricultural interpretation, block/panchayat mapping, filtering, analysis).

## 3. Methods

* **Fine grid:** exactly 5×5 sub-cell centers per IMD cell (0.05°,
  area-tiling). Every resampling operation is documented in
  `data/processed/meta.json` (`resampling_operations`).
* **Land mask (important):** an IMD coarse cell is *land* iff ≥50% of aligned
  days have valid IMD rainfall. Fine pixels enter **Y/M** only if the parent
  coarse cell is land **and** CHIRPS is valid on *every* aligned day — the
  coastal strip within one CHIRPS cell of CHIRPS-ocean is structurally missing
  and is **excluded, never imputed** (44,243 of 47,250 target-valid fine cells;
  the remaining ~3,007 are that coastal strip). X keeps filled values over sea
  so convolutions stay finite; no sample day, target or metric ever uses sea.
* **Temporal split (no leakage):** train = 2018+2019+2020 (366 days),
  val = 2021 (122), test = 2022 (122). Normalization statistics come from
  **training years only**. Test is never used for tuning or model selection.
* **Model selection rule (fixed before test):** lowest val MAE; rows within
  0.1 mm are considered tied and the tie is broken by val correlation.
* **Loss experiment:** MAE vs heavy-rain-weighted MAE, chosen on validation.
* **Data-quality checks** (`quality.py`, run inside `preprocess.py`): date and
  coordinate alignment, latitude/longitude ordering, duplicate timestamps,
  missing-value fractions, rainfall units, DEM validity, ERA5 daily-aggregation
  sanity. Failures are loud, not silently repaired.
* **Aux-layer QC** (`scripts/verify_dataset.py`): array shapes vs meta, mask
  semantics, aux layers index the same land cells, value ranges sane. Run it
  after any rebuild; it exits non-zero on any problem.

## 4. Results — PILOT ONLY (Western Ghats, 2019–2022; NOT the Deccan run)

Reference: **CHIRPS 0.05° — a reference product, not ground truth.**
122 train / 122 val / 122 test monsoon days; same dates for every row.

| Row | Model | val MAE | val corr | test MAE | test corr | F1≥25mm (test) |
|---|---|---|---|---|---|---|
| A | Bilinear IMD baseline | 6.72 | 0.257 | 7.67 | 0.304 | 0.217 |
| B2 | Bias-corrected bilinear (train-derived) | 9.69 | 0.400 | 10.15 | 0.436 | 0.365 |
| B | U-Net rainfall only | 5.35 | 0.283 | 6.63 | 0.304 | 0.055 |
| C | U-Net + DEM | 5.39 | 0.289 | 6.58 | 0.314 | 0.061 |
| Cw | U-Net + DEM, weighted loss | 6.04 | 0.372 | 6.91 | 0.419 | **0.312** |
| D | **U-Net + DEM + ERA5-Land** | 5.41 | **0.331** | **6.45** | 0.376 | 0.146 |

* The pilot's selected model **D** had **15.9% lower MAE than the bilinear
  baseline** on unseen 2022 data. Correlation rose monotonically B → C → D.
* **B2 is an honest negative result:** naive per-cell bias correction does not
  generalize across monsoon years — the learned residual does it better.
* **MAE vs weighted loss trade-off is real:** MAE training minimizes mean
  error but smooths heavy rain; the weighted loss trades ~0.7 mm MAE for far
  better heavy-rain detection (relevant for agriculture).
* These tables come from `outputs/metrics/*` of the **pilot** run. **No
  metrics exist yet for the Deccan dataset** — producing them is the next step
  (see `HANDOVER.md`).

## 5. How to run everything

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt  # Windows Git-Bash

# --- dataset build (region-aware; default region = deccan) ---
python scripts/download_or_export.py --region deccan   # IMD + CHIRPS + DEM + ERA5 (resumable, cached per year)
python scripts/build_aux.py --region deccan            # admin + soil + NDVI + LULC aux layers (each --skip-<name>-able)
python scripts/preprocess.py                           # align, quality-check, split, normalize -> data/processed/
python scripts/verify_dataset.py                       # end-to-end QA; exits non-zero on any problem

# --- training (NEXT STEP, not yet run for deccan) ---
python scripts/train.py --channels imd_rain,dem,era5_t2m,era5_t2m_max,era5_dewp --out-name model_d
python scripts/ablation.py             # A-D table, val-based selection -> best_model.pt
python scripts/evaluate.py --models model_b,model_c,model_c_weighted,model_d
python scripts/infer.py --imd data/raw/imd/ind2022_rfp25.nc --date 2022-07-10
```

All scripts accept `--region {deccan,western_ghats}` (falls back to explicit
`--lat-min/--lat-max/--lon-min/--lon-max` if given). `preprocess.py` also
accepts `--split-years "2018,2019,2020|2021|2022"`.

WorldCover COG tiles are read with **spatial windowing** (zarr chunked reads of
only the 1024×1024 COG chunks overlapping the region, 4096-px strips): the
Deccan LULC build reads a fraction of each 36000×36000-px tile and never
decompresses a full page, so all 25 overlapping tiles process in minutes and
re-runs (tiles cached) in ~2.5 min.

## 6. Project structure

```
weather-downscaling/
├── data/
│   ├── raw/{imd,chirps,dem,era5,soil,vegetation,lulc,admin}/  # cached downloads
│   ├── processed/          # X/Y/M_{train,val,test}.npy + meta.json  (MODEL INPUT)
│   ├── aux/                # admin/, soil_soilgrids_*, ndvi_monthly_*, lulc_fractions_*, build_summary_*.json
│   └── reports/            # data dictionary, coverage, missingness, verify_dataset JSON
├── models/                 # checkpoints + histories (pilot only, so far)
├── scripts/
│   ├── config.py           # regions, years, split, dirs, all knobs
│   ├── netcdf3.py / imd_reader.py / grids.py / quality.py
│   ├── download_or_export.py / inspect_data.py / gee_export.js
│   ├── preprocess.py       # -> data/processed (model-ready)
│   ├── build_aux.py        # -> data/aux_data (admin/soil/NDVI/LULC) + reports
│   ├── retry_soil_nan.py   # soil NaN diagnostics (genuine SoilGrids nulls)
│   ├── verify_dataset.py   # end-to-end dataset QA (loud, exit-code)
│   ├── train.py / ablation.py / evaluate.py / infer.py
├── outputs/{maps,metrics,figures}/   # pilot run outputs
├── prediction/             # machine-readable Layer-2 contract output
├── HANDOVER.md             # how to train on this dataset + Layer-2 usage
└── README.md
```

## 7. Outputs of the dataset build (what exists NOW)

1. `data/processed/X/Y/M_{train,val,test}.npy` + `meta.json` — provenance, ROI,
   dates (610), split, grid, land-mask semantics, normalization stats
   (train-only: `rain_scale=100.0 mm`, `elev_scale=1222.7 m`), every resampling op.
   Channels: `imd_rain, dem, era5_t2m, era5_t2m_max, era5_dewp` (wind excluded:
   `era5_wind` is not in `CHANNELS_ALL`; the 2018 ERA5 cache had 122 fully
   missing wind days, so the channel is skipped, not imputed).
2. `data/aux_data/admin/admin_master_<region>.geojson` + `grid_admin_map_<region>.npz`
   — GADM 4.1 block tier (subdistricts), fine cell → state/district/subdistrict
   (nearest representative point; 200-cell PIP spot-check 0.68 agreement).
3. `data/aux_data/soil_soilgrids_<region>.npz` — sand/clay/ocd/phh2o/bdod at 5–15 cm
   on the 1,890 land coarse cells; each fine pixel inherits its parent coarse
   value. 8.8–9.4% property-wise missing = genuine SoilGrids nulls (documented,
   not rate-limiting; not imputed).
4. `data/aux_data/ndvi_monthly_<region>.npz` — 20 monthly composites (Jun–Sep
   2018–2022; days 5 & 25 of each month), validity-mask-aware bilinear
   resampling (no sentinel bleed); 8.0% missing (monsoon cloud), range
   −0.064..0.845.
5. `data/aux_data/lulc_fractions_<region>.npz` — 6 per-class area fractions +
   dominant class per land fine cell (WorldCover v200/2021 at 10 m; fractions
   from the 600×600-px stencil of each 0.05° cell). 97.14% of cells have class
   data; the 1,349 without are cells whose entire stencil is WorldCover ocean
   nodata (sea fringe of coastal IMD grid boxes) — they are never valid
   training targets (M=0). The 6 fractions do not sum to 1 in the 104 cells
   containing wetland (class 90), which has no fraction column — documented in
   the data dictionary.
6. **Soil moisture — OPTIONAL, PENDING (quota-blocked).** The builder
   (`build_soilmoisture` in `build_aux.py`) fetches daily ERA5-Land volumetric
   water (0–7 cm; 7–28 cm optional) on the SAME ERA5 lattice as the era5_*
   channels, cached per (batch, year), and refuses to write an incomplete
   file. The free Open-Meteo **daily** request limit was exhausted before
   materialization (one 64-loc × 122-day request ≈ 7.8k weight units; ~34
   such requests needed). **Resume on any later day** with:
   `python scripts/build_aux.py --region deccan --skip-admin --skip-soil --skip-ndvi --skip-lulc`
   — repeat across daily resets until it prints `done`; caches make each
   attempt additive. Layer-3 agro-advisory input, NOT a U-Net channel.
7. **Wind:** the raw ERA5 cache now contains clean 10 m wind for ALL 610 days
   (the 2018 fetch was repaired in place; land variables bit-identical). It is
   still NOT a model channel: the frozen dataset was preprocessed with the
   5-channel baseline and `CHANNELS_ALL` intentionally keeps it out — add it
   only via a documented re-preprocess (HANDOVER §7).
8. **ESA WorldCereal (crop type) — evaluated, NOT included:** the only
   no-auth distribution is Zenodo record 7875105 (global multi-GB ZIPs of 106
   agro-ecological-zone GeoTIFFs, AEZ-specific seasonality); acquisition +
   alignment to our lattice is a standalone task, out of scope for the data
   freeze. Revisit for Layer-3 crop-specific work.
9. `data/reports/` — `data_dictionary_<region>.md`,
   `coverage_report_<region>.md`, `missingness_report_<region>.md`,
   `verify_dataset_<region>.json`.
10. `data/aux_data/build_summary_<region>.json` — machine-readable build stats for
   every aux layer.

## 8. How Layer 2 consumes this (contract)

`prediction/*.npz` (emitted by `evaluate.py` / `infer.py`), per model/split:

```
dates        (n,)       YYYY-MM-DD strings
latitude     (H,)       fine-grid lat centers, deg N, ascending
longitude    (W,)       fine-grid lon centers, deg E, ascending
rainfall_mm  (n, H, W)  daily rainfall, mm/day
```

Layer-2 then intersects the 0.05° fields with Panchayat polygons (LGD IDs):
zonal-mean `rainfall_mm[t]` over each polygon per date. The aux layers make
this direct:

* `grid_admin_map_<region>.npz` already assigns every land cell a
  state/district/**subdistrict (block)** — GADM L3 is the block tier. Cell →
  block aggregation of predictions needs no new GIS: group by `subdistrict_ids`.
* Panchayat/village polygons are **not** in GADM; LGD publishes them without a
  bulk API. The ID schema in the admin map is LGD-joinable: fetch the LGD
  panchayat layer for the states of interest, intersect block polygons with
  panchayat polygons once, and every cell inherits its panchayat the same way.
* Soil/NDVI/LULC give the agricultural context for advisories (drought
  flags by soil water-holding proxies, vegetation state, dominant land use).

## 9. Limitations (documented honestly)

* **CHIRPS is the reference, not truth.** IMD and CHIRPS disagree substantially
  at daily scale (domain-mean daily coarse corr on the Deccan build ≈ 0.356);
  part of every error term is their disagreement, not model error. Weekly and
  monthly aggregates agree far better — downscaling value is clearest there.
* **Grid stagger:** CHIRPS' native lattice is offset half a cell from our
  area-tiling fine grid; Y is a half-cell bilinear sample of CHIRPS. Baseline
  and models are scored against the same Y, so comparisons are fair.
* **Monsoon-only window** (Jun–Sep, 2018–2022 for Deccan). No winter data.
* ~3,007 fine cells (coastal strip + sea fringe) are excluded from Y/M by
  design; LULC has no class data over the sea-fringe part of them.
* Soil has genuine SoilGrids nulls (8.8–9.4% per property); NDVI has monsoon
  cloud gaps (8.0%); neither is imputed — use NaN-aware statistics downstream.
* Heavy rain remains hard for MAE-trained models; the weighted-loss variant
  mitigates it at a small MAE cost. No probabilistic/uncertainty output yet.
* The U-Net is deliberately small (~150k params). Operational use would consume
  IMD Block forecasts as coarse input, whose error propagates through Layer 1.
