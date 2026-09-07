# Weather downscaling — Layer 1 (ML) + Layer 2 (Panchayat GIS)

**Layer 1:** coarse IMD 0.25° daily rainfall + SRTM elevation + ERA5-Land daily
context → small residual U-Net → fine 0.05° daily rainfall, scored against
CHIRPS 0.05° as the fine-resolution **reference**.

**Layer 2:** intersect that 0.05° field with LGD Gram Panchayat polygons and
write one rainfall value per Panchayat (GIS, not ML).

> Not yet an operational IMD **Block-forecast** → Panchayat system. Layer 1 still
> uses historical IMD grids. No RAG, agents, dashboards, or APIs.

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

## 2. Data (all real; no GEE authentication required)

| Dataset | Role | Source | Fetch |
|---|---|---|---|
| IMD 0.25° daily rainfall | coarse input | imdpune.gov.in (1901–2024) | `download_or_export.py` (official form endpoint + direct fallback) → `data/raw/imd/ind<YEAR>_rfp25.nc` |
| CHIRPS v2.0 0.05° daily | fine reference | UCSB CHC server | monthly global HDF5 files → `data/raw/chirps/` |
| SRTM 30 m (Terrarium, AWS Open Data) | DEM | elevation-tiles-prod | tile mosaic → `data/raw/dem/dem_roi.npz` |
| ERA5-Land daily T/Tmax/dewpoint (+ERA5 wind) | auxiliary input | Open-Meteo archive API (server-side daily aggregation, Asia/Kolkata days) | → `data/raw/era5/era5_daily.npz` (per-year caches) |

Optional GEE path (if direct downloads fail): `scripts/gee_export.js` exports
CHIRPS GeoTIFFs + `DEM_roi.tif` to Drive; `preprocess.py` auto-detects them in
`data/raw/chirps/` and `data/raw/dem/`.

## 3. Methods (what the model learns, how it is trained)

* **Fine grid:** exactly 5×5 sub-cell centers per IMD cell (0.05°, area-tiling).
  Every resampling operation is documented in `data/processed/meta.json`
  (`resampling_operations`): IMD bilinear 5×; CHIRPS bilinear (its lattice is
  staggered half a cell — identical for baseline and model, so the comparison
  is fair); DEM bilinear from the ~38 m mosaic; ERA5 nearest-sampled at IMD
  centers, sea-point NaNs filled nearest-neighbour (documented), bilinear to fine.
* **Residual learning:** the network predicts a correction to the bilinear
  baseline; final prediction = baseline + residual.
* **Temporal split (no leakage):** 2019–2020 → train, 2021 → validation,
  2022 → test. Normalization statistics come from **training years only**.
  Test is never used for tuning or model selection.
* **Model selection rule (fixed before test):** lowest val MAE; rows within
  0.1 mm are considered tied and the tie is broken by val correlation.
* **Loss experiment:** MAE vs heavy-rain-weighted MAE, chosen on validation.
* **Data-quality checks** (`quality.py`, run inside `preprocess.py`): date and
  coordinate alignment, latitude/longitude ordering, duplicate timestamps,
  missing-value fractions, rainfall units, DEM validity (range + glitches),
  ERA5 daily-aggregation sanity. Failures are loud, not silently repaired.
* ERA5-Land is land-only: sea points on the sampling margin are structural
  NaNs, filled by nearest valid value (documented, interior gradients intact).

## 4. Results (122 train / 122 val / 122 test monsoon days; same dates for every row)

Reference: **CHIRPS 0.05° — a reference product, not ground truth.**

| Row | Model | val MAE | val corr | test MAE | test corr | F1≥25mm (test) |
|---|---|---|---|---|---|---|
| A | Bilinear IMD baseline | 6.72 | 0.257 | 7.67 | 0.304 | 0.217 |
| B2 | Bias-corrected bilinear (train-derived) | 9.69 | 0.400 | 10.15 | 0.436 | 0.365 |
| B | U-Net rainfall only | 5.35 | 0.283 | 6.63 | 0.304 | 0.055 |
| C | U-Net + DEM | 5.39 | 0.289 | 6.58 | 0.314 | 0.061 |
| Cw | U-Net + DEM, weighted loss | 6.04 | 0.372 | 6.91 | 0.419 | **0.312** |
| D | **U-Net + DEM + ERA5-Land** | 5.41 | **0.331** | **6.45** | 0.376 | 0.146 |

Scientific reading (wording per SIH guidance — never "accuracy = X%"):

* The selected model **D** has **15.9% lower MAE than the bilinear baseline**
  on unseen 2022 data, and **36.5% lower MAE than the bias-corrected
  baseline**.
* DEM and ERA5-Land each help: correlation rises monotonically
  B → C → D (0.283 → 0.289 → 0.331 val), and D has the best test MAE. DEM's
  main effect is on RMSE (15.11 → 14.98), consistent with orographic structure.
* **B2 is an honest negative result:** bias correction computed from 2019–20
  rainfall *worsens* MAE by ~32% in 2021/22 — naive per-cell bias does not
  generalize across monsoon years. This is why the learned residual (which the
  U-Net applies *conditioned on the current day's state*) is the right tool.
* **The MAE/weighted-loss trade-off is real:** MAE-trained rows minimize mean
  error but smooth heavy rain (F1≥50mm ≈ 0). Cw trades ~0.7 mm val MAE for
  far better correlation and heavy-rain detection (F1≥25 = 0.312 vs 0.061) —
  relevant for agriculture, where missing a 50 mm day matters more than a
  0.5 mm bias on drizzle days.
* Full tables: `outputs/metrics/metrics.csv|json` (baselines + models + event
  precision/recall/F1) and `outputs/metrics/ablation.csv|json|_summary.md`.

## 5. How to run everything

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt  # Windows Git-Bash

python scripts/download_or_export.py   # IMD + CHIRPS + DEM + ERA5 (failsafe, resumable, cached per year)
python scripts/inspect_data.py         # prints auto-detected structure of every raw file

python scripts/preprocess.py           # align, quality-check, pair, split, normalize -> data/processed/
# ablation variants (same processed data, different channels/loss):
python scripts/train.py --channels imd_rain --out-name model_b
python scripts/train.py --channels imd_rain,dem --out-name model_c
python scripts/train.py --channels imd_rain,dem --loss weighted --out-name model_c_weighted
python scripts/train.py --channels imd_rain,dem,era5_t2m,era5_t2m_max,era5_dewp --out-name model_d

python scripts/ablation.py             # A-D table, val-based selection -> best_model.pt
python scripts/evaluate.py --models model_b,model_c,model_c_weighted,model_d
python scripts/infer.py --imd data/raw/imd/ind2022_rfp25.nc --date 2022-07-10

# Layer 2 — Panchayat mapping (needs LGD parquet + Layer 1 .npz)
python scripts/layer2_panchayat_mapping.py --input prediction/infer_2022-07-10.npz
python scripts/layer2_panchayat_mapping.py --block MANGALURU
python scripts/layer2_panchayat_mapping.py --state MAHARASHTRA --out outputs/layer2_mh
```

`era5_wind` is currently excluded (its 2019/2021 fetches hit the Open-Meteo
quota; `config.CHANNELS_ALL` documents how to re-enable it once back-filled).

## 6. Everything is configurable

Every script accepts `--lat-min --lat-max --lon-min --lon-max --start --end`.
`preprocess.py` also accepts `--split-years "2019,2020|2021|2022"` (or a legacy
fraction split) and `--era5-mode strict|optional`. Defaults live in
`scripts/config.py`. Examples:

```bash
python scripts/preprocess.py --split-years "2019|2021|2022"       # 1/1/1 years
python scripts/preprocess.py --era5-mode optional                 # drop dates lacking ERA5 (ablation fallback)
python scripts/preprocess.py --lat-min 18 --lat-max 22 --lon-min 73 --lon-max 77
```

If a ROI/date range yields too few valid samples, `preprocess.py` says so
loudly and suggests widening it.

## 7. Project structure

```
weather-downscaling/
├── data/
│   ├── raw/{imd,chirps,dem,era5}/     # downloaded as-is (NetCDF, HDF5, npz)
│   └── processed/                     # X/Y/M_{train,val,test}.npy + meta.json
├── models/                            # best_model.pt + model_{b,c,c_weighted,d}.pt (+ histories)
├── scripts/
│   ├── config.py                      # all knobs in one place
│   ├── netcdf3.py / imd_reader.py     # NetCDF-3 reader + auto-detecting IMD reader
│   ├── grids.py                       # fine-grid def, bilinear ops, aggregation
│   ├── quality.py                     # loud data-quality checks
│   ├── inspect_data.py / download_or_export.py / gee_export.js
│   ├── preprocess.py                  # align, quality-check, pair, split, normalize
│   ├── train.py                       # residual U-Net, loss options, early stopping
│   ├── evaluate.py                    # baselines, event metrics, maps, Layer-2 .npz
│   ├── ablation.py                    # A-D table + val-based model selection
│   ├── infer.py                       # channel-aware inference, no retraining
│   ├── layer2_panchayat_mapping.py    # 0.05° field → Panchayat rainfall
│   └── check_blocks.py                # diagnose GPs on the Layer 1 grid edge
├── outputs/{maps,metrics,figures,layer2}/
├── prediction/                        # Layer-1 .npz contract for Layer 2
├── requirements.txt
└── README.md
```

## 8. Outputs produced by a full run

1. `models/best_model.pt` — selected checkpoint + training history
2. `data/processed/X/Y/M_{train,val,test}.npy` + `meta.json` — provenance, ROI,
   dates, split, grid, normalization stats (train-only), every resampling op
3. `outputs/metrics/metrics.csv|json` — MAE / RMSE / corr + event
   precision/recall/F1 at 10/25/50 mm/day for baselines and all models
4. `outputs/metrics/ablation.csv|json|_summary.md` — the A–D table
5. `outputs/maps/` + `outputs/figures/comparison_<date>.png` — same-colour-scale
   maps (IMD, B1, B2, each U-Net, CHIRPS, error map)
6. `prediction/<model>_<split>.npz` + `_schema.json` — per-day fine-grid
   rainfall, machine-readable (Layer 2 contract)
7. `outputs/layer2/` — Panchayat CSV / GeoJSON / map / `layer2_qc.json`

## 9. Layer 2 — Panchayat mapping

### Contract (from Layer 1)

`prediction/*.npz` (also written by `infer.py`):

```
dates        (n,)       YYYY-MM-DD strings
latitude     (H,)       fine-grid lat centers, deg N, ascending
longitude    (W,)       fine-grid lon centers, deg E, ascending
rainfall_mm  (n, H, W)  daily rainfall, mm/day
```

`.npy` maps still work; lat/lon are then rebuilt from `config.ROI_DEFAULT`.

### Files

| File | Role |
|---|---|
| `scripts/layer2_panchayat_mapping.py` | Load Layer 1 field, spatial join to LGD GPs, write outputs |
| `scripts/check_blocks.py` | Debug: which GPs have no cell center inside the polygon |
| `scripts/config.py` | `OUT_LAYER2`, `RAW_PANCHAYAT`, `WET_DAY_MM`, `MAX_FALLBACK_KM` |
| `scripts/infer.py` | Also saves `outputs/maps/infer_<date>.npy` next to the `.npz` |
| `requirements.txt` | `geopandas`, `shapely`, `pyarrow` for GIS |
| `data/raw/administrative/panchayat/LGD_Panchayats.parquet` | GP polygons (not in git) |
| `outputs/layer2/panchayat_weather.csv` | one row per Panchayat per date |
| `outputs/layer2/panchayat_summary.csv` | mean/max/min, wet days |
| `outputs/layer2/panchayat_weather.geojson` | polygons + map-date rainfall |
| `outputs/layer2/panchayat_weather_map.png` | choropleth |
| `outputs/layer2/layer2_qc.json` | coverage / method counts / grid |
| `outputs/layer2_mh/` | same products for `--state MAHARASHTRA` |

CSV columns: `date`, `state`, `district`, `block_id`, `block_name`,
`panchayat_id`, `panchayat_name`, `rainfall_mm`, `n_cells`,
`mapping_method`, `fallback_distance_m`.

### How a value is assigned

Default region: every GP that intersects the Layer 1 0.05° box.

1. **`direct_grid`** — mean of 0.05° cell *centers* that fall inside the polygon.
2. **`area_weighted`** — if no center is inside (small GPs): intersection-area
   weighted mean of overlapping cells.
3. **`nearest_fallback`** — still unmatched: nearest cell to the centroid, only
   if closer than `MAX_FALLBACK_KM` (15 km).
4. **`unmapped`** — otherwise `rainfall_mm` is NaN.

Checked on `infer_2022-07-10.npz`: 8,236 GPs, all mapped, 4,780 direct / 3,456
area, 0 nearest / 0 unmapped. West-coast rain is higher than rain-shadow east.

## 10. Limitations (documented honestly)

* **CHIRPS is the reference, not truth.** IMD and CHIRPS disagree substantially
  at daily scale over the Western Ghats (domain-mean daily corr ≈ 0.35–0.57
  depending on window); part of every error term is their disagreement, not
  model error. Weekly/monthly aggregates agree far better — downscaling value
  is clearest there.
* **Grid stagger:** CHIRPS' native lattice is offset half a cell from our
  area-tiling fine grid; Y is a half-cell bilinear sample of CHIRPS. Baseline
  and models are scored against the same Y, so comparisons are fair.
* **Monsoon-only window** (Jun–Sep). 2020 is missing from the current ERA5
  stack (free-API quota; per-year caches make back-filling one re-run), so the
  ERA5 rows use 2019/2021/2022. Wind channel pending the same back-fill.
* Heavy rain remains hard for MAE-trained models (regression to the mean);
  Cw mitigates it at a small MAE cost. Neither variant is a probabilistic
  forecast — no uncertainty quantification yet.
* The U-Net is deliberately small (~150k params). This is a *validated
  refinement engine*, not a production forecast system: operational use would
  consume IMD Block forecasts as coarse input, whose error propagates through
  Layer 1.
