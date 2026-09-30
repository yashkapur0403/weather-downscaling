# Panchayat Rainfall Downscaling & Agro-Advisory — Layers 1–3

**What it is:** the complete SIH pipeline, all three layers, in one repository.
**Layer 1** refines coarse IMD 0.25° daily rainfall with SRTM elevation and ERA5-Land
context through a small residual U-Net into a fine 0.05° daily rainfall field, scored
against CHIRPS 0.05° as the fine-resolution **reference**. **Layer 2** maps that field to
the LGD Panchayat boundaries (86,103 of 87,735 mapped, 98.1%). **Layer 3** turns a
Panchayat's Layer-1 rainfall into a rule-based, explained crop advisory served by a FastAPI
backend (`backend/`, port 8000) and shown in a Next.js dashboard (`Frontend/`, port 3000).

**Run it in two terminals:**

```bash
cd backend  && pip install -r requirements.txt && uvicorn app:app --reload --port 8000
cd Frontend && npm install && npm run dev          # http://localhost:3000
```

The backend runs with **no API keys** (advisory and explain fall back to deterministic
templates). `backend/README.md` documents every route and what data backs it.

**Current status (be honest when presenting):**

* ✅ **Layer-1 DATASET complete for the Deccan region** (`region="deccan"`):
  5 monsoons (2018–2022), 610 days, 47,250 land cells, model-ready
  `X/Y/M_{train,val,test}.npy`, plus auxiliary layers (admin, soil, NDVI, LULC)
  for Layer-2. End-to-end QA passes (`scripts/verify_dataset.py`).
* ✅ Model **methodology validated on the Western Ghats pilot** (older, smaller
  run: 2019–2022, 122-day splits — §4). That pilot proved the pipeline works
  end-to-end and beat the baseline.
* ✅ **Model training/evaluation on the Deccan dataset IS done** — six ablation
  rows (A–F) plus the deployed E+F ensemble, selected on the **validation** split
  only. On the unseen 2022 test split the shipped model gives MAE 8.43 mm /
  RMSE 15.00 / corr 0.527 against the bilinear baseline's 9.60 / 18.18 / 0.385,
  and beats it at every reported heavy-rain threshold (§4 and
  `outputs/metrics/ablation_summary.md`). The Western-Ghats pilot numbers are
  kept in §4, clearly labelled as historical.
* ✅ **Layer-2 executed and verified for the Deccan config:** block aggregation
  and Panchayat mapping produced the final CSV, summary, GeoJSON, map, and QC
  artifacts in `outputs/layer2/`. The run mapped 86,103 of 87,735 Panchayats
  (98.1% coverage); 1,632 coastal/off-grid Panchayats remain unmapped.
  RAG / dashboards are not built.

## 0. File map — what is what, who uses what (read this first)

**New here?** Read this table, then §1–§3 for context, then `docs/HANDOVER.md` for
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
| `data/raw/` | Source downloads (IMD, CHIRPS, DEM, ERA5, GADM, SoilGrids batches, VIIRS slices, WorldCover tiles, LGD panchayats). NOT in git — **reproduce with §5 commands**; per-year/batch caches make re-runs cheap | only rebuilds |
| `scripts/` | **ACTIVE pipeline (Deccan)**: one script per stage (see §6) — every stage is cached/resumable | the pipeline |
| `backend/` | The FastAPI app (**Layer 3**, §9): data store, rule engine, Groq/Sarvam text, all `/api/*` routes. Serves on port 8000 and reads every other directory in this repo | the web app, judges |
| `Frontend/` | The Next.js dashboard (**§9.7**). Reads the backend at `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`) | the user |
| `requirements.txt`, `prediction/` | Pipeline dependencies (**§5**), and the extra U-Net runs `scripts/infer.py` writes locally (`prediction/infer_*.npz`, gitignored — the backend falls back to these if the served grid is absent) | training / inference |
| `legacy/old-app-snapshot/` | **Legacy, superseded — archive only.** The teammate's original app snapshot (formerly `weather-downscaling-main/`), merged in early; the app now lives at `backend/` + `Frontend/` at the repo root. Kept because it holds the **only committed copies** of the per-Panchayat `panchayat_weather.geojson` (46 MB) and `layer2_mh/` GeoJSON, the Panchayat CSV/PKL exports and `layer1_model.pkl`. Do not wire it up. The same folder holds the old `legacy/frontend_backend_match.patch` | archive / data recovery |
| `data.zip` | Current **Deccan** model-ready archive (`data/processed/`, ~929 MiB; CRC + shapes verified). **Not in git** — copy from the shared OneDrive folder and run `unzip data.zip` at the repo root (or rebuild, §5). The old pilot archive is kept as `data_pilot_westernghats_LEGACY.zip` (~1.7 GB) | transfer / offline rebuild |
| `models/`, `outputs/` | **The Deccan run**: checkpoints A–F + `ensemble.json` (the *deployed* model), the served grid `outputs/prediction_test.npz`, the ablation table, the provenance manifest `layer1_manifest.json` and the Layer-2 products | backend, Layer-2, humans |
| `docs/HANDOVER.md` | **The ops manual**: exact training commands, evaluation rules, Layer-2 recipes, rebuild instructions, all dataset decisions | the person doing training / Layer-2 (you, probably) |
| `docs/` (rest) | `LOGICAL_VALIDATION_REPORT.md` (the logical/scientific validation report), `COMPREHENSIVE_QA_TEST_REPORT.md` (the executed QA report) and `END_TO_END_QA_TEST_PLAN.md` (its plan). All long-form documentation lives here | humans, judges |
| `qa/` | The QA **evidence**: one script per investigation plus its recorded JSON result, cited by `docs/LOGICAL_VALIDATION_REPORT.md` §19. `qa/README.md` says how to re-run them | reviewers |

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
* **Model:** 117,329-parameter residual U-Net (learns the *correction* to the
  bilinear baseline, so it can never do worse than the baseline by
  construction of the residual parameterization).
* **Loss:** masked MAE (default) or a heavy-rain-weighted MAE (selected by
  validation, see §3).

## 2. Regions, datasets and references (all real; no GEE authentication required)

### 2.1 Regions

Two named regions live in `scripts/config.py` (`config.REGIONS`):

| region | bbox (lat, lon) | grid | role |
|---|---|---|---|
| `deccan` (default, **shipped**) | 11.5–25.5 N, 71.5–81.25 E | 285 × 200 @ 0.05° | the full Layer-1 dataset, 2018–2022, and every number in §4 |
| `western_ghats` | 13–17 N, 74.25–78.25 E | 0.05° | historical pilot only (§4.5) |

### 2.2 Datasets and references — the final set the shipped run used

Exactly the nine sources below were consumed by the shipped Deccan build. Anything
evaluated and **not** used (ESA WorldCereal crop type, ERA5-Land soil moisture) is
listed as such in §7 — this table is deliberately not aspirational.

| Dataset | Version / resolution | Period used | Role here | Lands at | Reference |
|---|---|---|---|---|---|
| **IMD daily gridded rainfall** | 0.25° NetCDF (`ind<YEAR>_rfp25.nc`) | Jun–Sep 2018–2022 | Layer-1 **coarse input** channel + the land mask | `data/raw/imd/` | India Meteorological Department, Pune — [cmpg/Griddata](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html) |
| **CHIRPS** | v2.0, 0.05° daily, global | same window | the fine-resolution **reference (Y)** every metric is scored against — a reference, not ground truth | `data/raw/chirps/` (monthly) | UCSB Climate Hazards Center — [CHIRPS-2.0/global_daily](https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/) |
| **SRTM elevation** | 30 m, "terrarium" elevation tiles | static | Layer-1 **DEM channel** | `data/raw/dem/` | NASA/USGS SRTM via AWS Open Data — [elevation-tiles-prod](https://s3.amazonaws.com/elevation-tiles-prod/terrarium/) |
| **ERA5-Land** daily mean/max temperature + dewpoint | 0.1° reanalysis, daily aggregates on a 1.0° lattice | Jun–Sep 2018–2022 | Layer-1 **auxiliary channels** (`era5_t2m`, `era5_t2m_max`, `era5_dewp`) | `data/raw/era5/` (per-year caches) | Copernicus C3S ERA5-Land, via the [Open-Meteo archive API](https://archive-api.open-meteo.com/v1/archive) |
| **GADM** administrative polygons | 4.1 (`gadm41_IND_<lvl>.json.zip`), levels **1–3** (India) | static | **aux**: fine cell → state / district / subdistrict (block) | `data/raw/admin/` → `data/aux_data/admin/` | [GADM 4.1](https://geodata.ucdavis.edu/gadm/gadm4.1/json/) |
| **SoilGrids** | v2.0, 5–15 cm, 250 m | static | **aux**: sand / clay / organic carbon / pH / bulk density for Layer 3 (`clay` decides R7) | `data/raw/soil/batches/` → `data/aux_data/` | ISRIC — [SoilGrids REST API](https://rest.isric.org/soilgrids/v2.0/properties/query) |
| **NOAA CDR VIIRS NDVI** | `VIIRS-Land_v001`, 0.05° | Jun–Sep 2018–2022 → 20 monthly composites (days 5 & 25) | **aux**: vegetation state (`ndvi` decides R6) | `data/raw/vegetation/slices/` → `data/aux_data/` | NOAA NCEI — [land-normalized-difference-vegetation-index](https://www.ncei.noaa.gov/data/land-normalized-difference-vegetation-index/access) |
| **ESA WorldCover** | v200, 2021, 10 m COG | 2021 | **aux**: 6 per-class area fractions + dominant class (R8 rule) | `data/raw/lulc/` → `data/aux_data/` | ESA — [WorldCover S3 tiles](https://esa-worldcover.s3.eu-central-1.amazonaws.com/) |
| **LGD Gram-Panchayat boundaries** | release tag `admin/panchayats` (~368 MB GeoParquet) | static | **the Layer-2 join key**: the 87,735 Panchayats of the ROI, with real LGD codes | `data/raw/administrative/panchayat/` | Local Government Directory, Ministry of Panchayati Raj — [lgdirectory.gov.in](https://lgdirectory.gov.in); CC0-1.0 redistribution via [india-geodata](https://github.com/yashveeeeeeer/india-geodata), aggregating [ramSeraph/indian_admin_boundaries](https://github.com/ramSeraph/indian_admin_boundaries) |

Fetching is two commands plus one: `scripts/download_or_export.py` (IMD, CHIRPS,
DEM, ERA5), `scripts/build_aux.py` (GADM, SoilGrids, NDVI, WorldCover) and
`scripts/fetch_lgd_panchayats.py` for the LGD boundaries — the last one only
because they are a separate ~368 MB release that needs a SHA-256 check and a clip
to the Deccan ROI. All of them cache per year / per batch, so re-runs are cheap.

Aux layers are **not** U-Net input channels — they exist for Layer-2
(agricultural interpretation, block/panchayat mapping, filtering, analysis) and,
for clay / NDVI / land cover, for the Layer-3 rules.

### 2.3 Attribution

Each dataset stays under its provider's own terms; where this repository records a
licence explicitly it is the LGD redistribution above (**CC0-1.0, public domain**,
stated in `scripts/fetch_lgd_panchayats.py`). Derived layers are documented per
file in `data/reports/data_dictionary_deccan.md`, which is the artefact to quote
when describing what was built from what.

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
* **Model selection rule (fixed before test, validation split only):** lowest
  val MAE wins; rows within **0.75 mm** are treated as tied (noise level) and
  the tie is broken by val heavy-rain skill (**F1 at >=25 mm**). This replaced
  the earlier "0.1 mm tie broken by correlation" rule, which selected a model
  that barely detected heavy events (F1>=50 mm = 0.015).
* **Deployed model = an ensemble (row EF).** A single MAE-trained residual
  network is calibrated to the conditional mean, so >=50 mm/day events are
  predicted at ~26-28 mm and F1 at that threshold collapses (measured test
  precision 0.44, recall 0.08). `models/ensemble.json` therefore declares a
  weighted mean of two checkpoints that share architecture/channels/scaling
  but differ in loss emphasis: **E** (heavy-rain weighted) plus **F** (>=50 mm
  weighted). The mixing weight is searched **on validation only**, constrained
  to stay inside the same 0.75 mm MAE window, then ranked by val F1>=25 mm -
  it independently selected E weight 0.5. Every serving path
  (`generate_pred.py`, `backend/live_infer.py`) resolves the deployed model
  through `scripts/ensemble.py`, so the served grid and the live-inference
  route can never come from different models. `models/best_model.pt` is kept
  as the single-checkpoint fallback, used only when `ensemble.json` is absent.
* **Provenance:** `generate_pred.py` writes
  `outputs/metrics/layer1_manifest.json` with the sha256 of the grid, of each
  contributing checkpoint, and the weights - and `/api/metrics` republishes it
  as `layer1_provenance`. A retrain therefore cannot silently leave a stale
  grid in place: hashes either match or they do not.
* **Loss experiment:** MAE vs heavy-rain-weighted MAE, chosen on validation.
* **Data-quality checks** (`quality.py`, run inside `preprocess.py`): date and
  coordinate alignment, latitude/longitude ordering, duplicate timestamps,
  missing-value fractions, rainfall units, DEM validity, ERA5 daily-aggregation
  sanity. Failures are loud, not silently repaired.
* **Aux-layer QC** (`scripts/verify_dataset.py`): array shapes vs meta, mask
  semantics, aux layers index the same land cells, value ranges sane. Run it
  after any rebuild; it exits non-zero on any problem.

## 4. Results — the Deccan run (shipped) and the Western-Ghats pilot (historical)

Reference product for every number below: **CHIRPS v2.0 0.05° — a reference, not
absolute ground truth.** The test split is **122 unseen monsoon days**
(2022-06-01 → 2022-09-30) over **44,243 valid land cells/day**, i.e.
**5,397,646 scored cell-days**. Every row was trained on 2018–2020, decided on the
2021 **validation** split, and scored on 2022 **once** — the test split never
influenced any choice. Machine-readable: `outputs/metrics/ablation.json`;
rendered: `outputs/metrics/ablation_summary.md`.

### 4.1 What each metric is for

| Metric | What it measures | Why it matters for this product |
|---|---|---|
| **MAE** (mm/day) | mean absolute error over valid land cells | the overall accuracy of the field the advisory reads; it decides the risk band a farmer sees |
| **RMSE** (mm/day) | the same error, dominated by the largest misses | catches the failure that actually hurts — missing a downpour |
| **corr** | Pearson r of predicted vs reference across all cells and days | whether the *pattern* (where and when it rains) is right, independent of bias |
| **F1 ≥ 10 mm** | event detection, moderate-rain band | the irrigation rule (R2) and the "substantial rain" advisory band live here |
| **F1 ≥ 25 mm** | event detection, heavy-rain band | the tie-breaker in model selection; waterlogging / drainage risk starts here |
| **F1 ≥ 50 mm** | event detection, extreme tail | the hardest and most consequential case; a plain MAE-trained net collapses at this threshold |

Base rates the F1 values are computed on (test): **1,618,246** cell-days ≥ 10 mm
(30.0 %), **624,843** ≥ 25 mm (11.6 %) and **167,065** ≥ 50 mm (3.1 %) out of
5,397,646. `outputs/metrics/ablation.json` also carries precision and recall for
every row at every threshold, so any single number here can be audited.

### 4.2 Deccan — every ablation row (validation and test)

| Row | Model | val MAE | val RMSE | val corr | test MAE | test RMSE | test corr | F1≥10 | F1≥25 | F1≥50 |
|---|---|---|---|---|---|---|---|---|---|---|
| **A** | Bilinear IMD baseline (no learning) | 8.55 | 16.33 | 0.288 | 9.60 | 18.18 | 0.385 | 0.477 | 0.343 | 0.234 |
| B2 | Bias-corrected bilinear (train-derived) | 12.13 | 18.07 | 0.343 | 12.75 | 19.45 | 0.427 | 0.546 | 0.398 | 0.271 |
| B | U-Net, rainfall only | 7.13 | 14.52 | 0.293 | 8.51 | 16.93 | 0.396 | 0.281 | 0.226 | 0.115 |
| C | U-Net + DEM | 7.04 | 14.16 | 0.318 | 8.36 | 16.50 | 0.419 | 0.332 | 0.231 | 0.104 |
| Cw | U-Net + DEM, weighted loss | 11.40 | 18.80 | 0.226 | 12.27 | 20.11 | 0.287 | 0.440 | 0.305 | 0.188 |
| D | U-Net + DEM + ERA5-Land | **6.93** | 13.79 | 0.346 | 8.17 | 16.01 | 0.446 | 0.393 | 0.281 | 0.015 |
| E | D + heavy-rain-weighted loss | 7.44 | **12.82** | 0.460 | **8.21** | **14.57** | 0.518 | **0.602** | 0.425 | 0.129 |
| F | D + ≥50 mm-weighted loss | 8.64 | 15.67 | 0.449 | 9.03 | 16.53 | 0.510 | 0.584 | **0.445** | **0.320** |
| **EF** | **DEPLOYED — E+F ensemble (E weight 0.5)** | 7.88 | 13.71 | **0.466** | **8.43** | 15.00 | **0.527** | 0.601 | 0.442 | 0.284 |

**What the deployed row buys, as deltas rather than adjectives** (test split):

| Metric | Bilinear baseline (A) | Deployed EF | Change |
|---|---|---|---|
| MAE (mm/day) | 9.60 | **8.43** | **−12.2 %** |
| RMSE (mm/day) | 18.18 | **15.00** | **−17.5 %** |
| corr | 0.385 | **0.527** | **+0.142** |
| F1 ≥ 10 mm | 0.477 | **0.601** | **+26 %** relative |
| F1 ≥ 25 mm | 0.343 | **0.442** | **+29 %** relative |
| F1 ≥ 50 mm | 0.234 | **0.284** | **+21 %** relative |

* **EF and F are the only rows that beat the baseline on every reported metric at
  once.** EF is the deployed one because selection happens inside a
  **pre-registered 0.75 mm validation-MAE window** around the best single row
  (E, val MAE 7.44); F's val MAE (8.64) is outside it, so the rule — fixed before
  the test split was touched — cannot select F however good its tail looks. Inside
  the window, ranking is by validation F1 ≥ 25 mm, which selected the equal-weight
  E+F blend (val F1 ≥ 25 mm **0.366**, best of the 6 candidates in the window).
* **Validation MAE alone would have been the wrong criterion.** Row D has the best
  validation MAE of all (6.93) yet a useless tail (F1 ≥ 50 mm = **0.015**); row E
  gives up 0.51 mm of validation MAE against D and buys much better heavy-band
  detection with it (val F1 ≥ 25 mm 0.355 vs 0.174, val F1 ≥ 50 mm 0.131 vs
  0.023). That is why the selection rule pairs MAE with a heavy-rain tie-break.
* **E has the better mean error, the ensemble has the better balance.** E alone:
  MAE 8.21 (−14.5 % vs baseline), RMSE 14.57 (−19.9 %) — but F1 ≥ 50 mm 0.129,
  *worse than the baseline's 0.234*. EF gives up 0.22 mm of MAE relative to E and
  buys back the tail (0.284 > 0.234). Nothing shipped is worse than the baseline.
* **B2 is an honest negative result:** naive per-cell bias correction does not
  generalise across monsoon years (test MAE 12.75, worse than doing nothing) —
  the learned residual generalises better than the hand-derived correction.
* **Tail honesty.** Even deployed, ≥ 100 mm/day events are largely missed (recall
  ≈ 0.005) and cells observed above 100 mm are predicted at ≈ 35 mm. Against this
  reference the model is better than not modelling; it is **not** a reliable
  extreme-rainfall predictor. See §10.

### 4.3 Why these numbers can be trusted

* **The grid that is served is the grid that was scored.** `generate_pred.py`
  writes `outputs/metrics/layer1_manifest.json` carrying the sha256 of
  `outputs/prediction_test.npz` (`09410cf6…5b20`), of each contributing checkpoint
  (`model_e.pt` `75187a83…`, `model_f.pt` `569b3adc…`), the member weights
  (0.5 / 0.5), the 5 input channels and the parameter count (**117,329**).
  `/api/metrics` republishes all of it as `layer1_provenance`, so a retrain cannot
  silently leave a stale grid in front of the API — the hashes either match or
  they do not. `backend/tests/test_artifact_consistency.py` asserts exactly that.
* **The ensemble arithmetic is pinned to one definition.** `scripts/ensemble.py`
  is the single source of truth for the deployed model, and every consumer
  (`generate_pred.py`, `backend/live_infer.py`) resolves through it;
  `qa/qa_ensemble_smoke.py` checks the loader against a manual weighted mean
  (max difference **0.0**), and `qa/qa_blend.py` records the full blend search.
* **The predicted field is physically plausible:** 122 × 285 × 200 grid, max
  193.96 mm/day, mean 9.41 mm/day, no negative values, `NaN` where the land mask
  excludes the coastal strip and sea.
* **Spatial and temporal sanity** (`qa/qa_logic_core.json`): neighbouring cells
  differ by 1.10 mm on average (p95 3.83 mm, max 24.62), and the daily field
  correlates **0.836** with CHIRPS and **0.898** with IMD-observed rainfall.
* **Reproducible from committed artefacts, in minutes:** `python scripts/ablation.py`
  rebuilds this table on CPU (~80 s) and re-applies the same validation-only
  selection rule; `python generate_pred.py` rebuilds the served grid and its
  manifest; `python -m pytest backend/tests` runs 69 tests including the
  artifact-consistency contract.

### 4.4 The benchmark it is scored against (row A)

Row A is **bilinear upsampling of the IMD 0.25° field with no learning at all** —
the trivial baseline. It is deliberately kept in the table and, on the
heavy/event bands, it is genuinely competitive (F1 ≥ 50 mm 0.234, better than
rows B, C, Cw, D and E). Quoting an improvement against a *strong* trivial
baseline is the point: the learned model's value shows up as accuracy **and**
event skill at the same time, not as a weak-baseline artefact.

### 4.5 Historical pilot (Western Ghats) — kept for the methodology trail

The pilot is an earlier, smaller run (Western Ghats bbox, 2019–2022, 122-day
splits, region rows A–D only). Its numbers are **historical** and are not the
shipped product; they are kept so the methodology trail stays visible.

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
* The table above is the historical **pilot** run (Western Ghats, 2019–2022),
  kept so the methodology trail stays visible. The Deccan numbers that ship are
  the box at the top of this section; `outputs/metrics/ablation_summary.md` is
  the artefact of record.

## 5. How to run everything

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt  # Windows Git-Bash

# --- dataset build (region-aware; default region = deccan) ---
python scripts/download_or_export.py --region deccan   # IMD + CHIRPS + DEM + ERA5 (resumable, cached per year)
python scripts/build_aux.py --region deccan            # admin + soil + NDVI + LULC aux layers (each --skip-<name>-able)
python scripts/fetch_lgd_panchayats.py                 # Layer-2 input: LGD Gram-Panchayat boundaries (~368 MB, CC0)
python scripts/preprocess.py                           # align, quality-check, split, normalize -> data/processed/
python scripts/verify_dataset.py                       # end-to-end QA; exits non-zero on any problem

# --- training ---
python scripts/train.py --channels imd_rain,dem,era5_t2m,era5_t2m_max,era5_dewp --out-name model_d
python scripts/train.py --channels imd_rain,dem,era5_t2m,era5_t2m_max,era5_dewp \
    --loss weighted --out-name model_e                       # heavy-rain weighted
python scripts/train.py --channels imd_rain,dem,era5_t2m,era5_t2m_max,era5_dewp \
    --loss weighted --weight-rain-mm 50 --weight-mult 10 --out-name model_f
python scripts/ablation.py             # A..F + EF table, val-only selection
                                       # -> models/ensemble.json (deployed)
                                       #    models/best_model.pt (fallback)
python generate_pred.py                # outputs/prediction_test.npz
                                       # + outputs/metrics/layer1_manifest.json
python scripts/evaluate.py --models model_b,model_c,model_c_weighted,model_d,model_e,model_f
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
│   ├── aux_data/           # admin/, soil_soilgrids_*, ndvi_monthly_*, lulc_fractions_*, build_summary_*.json
│   └── reports/            # data dictionary, coverage, missingness, verify_dataset JSON
├── models/                 # checkpoints + histories + ensemble.json (deployed model)
├── scripts/
│   ├── config.py           # regions, years, split, dirs, all knobs
│   ├── ensemble.py         # THE definition of the deployed model
│   │                       #   (models/ensemble.json -> weighted members)
│   ├── netcdf3.py / imd_reader.py / grids.py / quality.py
│   ├── download_or_export.py / inspect_data.py / gee_export.js
│   ├── preprocess.py       # -> data/processed (model-ready)
│   ├── build_aux.py        # -> data/aux_data (admin/soil/NDVI/LULC) + reports
│   ├── fetch_lgd_panchayats.py  # Layer-2 input: LGD panchayat boundaries (CC0)
│   ├── retry_soil_nan.py   # soil NaN diagnostics (genuine SoilGrids nulls)
│   ├── verify_dataset.py   # end-to-end dataset QA (loud, exit-code)
│   ├── train.py / ablation.py / evaluate.py / infer.py
│   ├── layer2_panchayat_mapping.py / produce_block_rainfall.py / layer2_config.py
│   └── check_blocks.py
├── backend/                # FastAPI app (Layer 3 advisory + /api/*), port 8000
├── Frontend/               # Next.js dashboard, port 3000
├── prediction/             # extra U-Net runs written by scripts/infer.py (gitignored, local)
├── requirements.txt        # pipeline deps (backend + Frontend have their own)
├── outputs/{maps,metrics,figures,layer2}/  # served grid, ablation + provenance, Layer-2 products
├── generate_pred.py        # rebuilds outputs/prediction_test.npz + layer1_manifest.json
├── docs/                   # HANDOVER.md + the validation / QA reports
├── qa/                     # the QA evidence: qa_*.py scripts + their qa_*.json results
├── legacy/                 # superseded app snapshot (unique data) + the old patch
├── data.zip                          # current Deccan archive (data/processed/, ~929 MiB)
├── data_pilot_westernghats_LEGACY.zip # old Western-Ghats pilot archive (~1.7 GB, kept separate)
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
  **This is the ONLY remaining external Layer-2 input**: the mapping expects
  `data/raw/administrative/panchayat/LGD_Panchayats.parquet` (override with
  `--panchayats`), and `layer2_panchayat_mapping.py` exits with a clear error
  until it is present.
* Pipeline: `train.py` → `generate_pred.py` (writes the served grid
  `outputs/prediction_test.npz`) → `scripts/layer2_panchayat_mapping.py` (writes
  `outputs/layer2/*`). The per-date Panchayat CSV is >100 MB and is deliberately
  **not** committed, so the backend does not read it: it reads the grid directly
  at each Panchayat's own polygon point (`outputs/layer2/panchayat_index.csv`).

**LGD Panchayat input (`LGD_Panchayats.parquet`) — prepared in one command.**
Layer-2's only external input is a GeoParquet of Gram-Panchayat polygons at
`data/raw/administrative/panchayat/LGD_Panchayats.parquet`, with a `geometry`
column plus `gpcode, gpname, stname, dtname, blklgdcode, blkname` (CRS
`EPSG:4326`; override the path with `--panchayats`). Get or refresh it with:

```bash
python scripts/fetch_lgd_panchayats.py   # download + sha256 verify + clip to the deccan ROI
```

It is **LGD-derived** (official Local Government Directory, Ministry of
Panchayati Raj, bundled in a CC0 public-domain redistribution) and carries real
LGD codes, so it is the authoritative tier — not a fallback. Non-LGD boundary
sets (data.gov.in / Datameet / state GIS) remain **geometry-only substitutes and
are NOT equivalent to LGD**. The 15 states the Deccan grid covers (from
`data/aux_data/admin/grid_admin_map_deccan.npz`): AndhraPradesh, Chhattisgarh,
DadraandNagarHaveli, DamanandDiu, Goa, Gujarat, Karnataka, Kerala,
MadhyaPradesh, Maharashtra, Puducherry, Rajasthan, TamilNadu, Telangana,
UttarPradesh. Schema, provenance and verification: `docs/HANDOVER.md` §4.2.
* Soil/NDVI/LULC give the agricultural context for advisories (drought
  flags by soil water-holding proxies, vegetation state, dominant land use).

### 8.1 Layer 2 executed results

The Deccan Layer 2 run used `outputs/prediction_test.npz` for 122 monsoon days
(`2022-06-01` to `2022-09-30`) on a 285 x 200 fine grid at 0.05 degree
resolution. The generated artifacts were checked for existence, non-zero size,
readability, and expected schema:

| Artifact | Result |
|---|---|
| `outputs/layer2/panchayat_weather.csv` | Daily Panchayat rainfall time series |
| `outputs/layer2/panchayat_summary.csv` | Seasonal Panchayat aggregates |
| `outputs/layer2/block_rainfall.csv` | Daily block-level rainfall aggregation |
| `outputs/layer2/panchayat_weather.geojson` | Mapped Panchayat geometries with rainfall attributes |
| `outputs/layer2/panchayat_weather_map.png` | Generated choropleth map |
| `outputs/layer2/layer2_qc.json` | Provenance and QC record |

> **Note (QA hardening).** `panchayat_weather.csv` and `panchayat_weather.geojson` are >100 MB and are
deliberately **not committed** (see `.gitignore`). The backend therefore serves the per-date
value from the U-Net grid at each Panchayat's *own* polygon point (built by
`backend/build_panchayat_index.py`); run `scripts/layer2_panchayat_mapping.py` locally if you need
the area-weighted per-date CSV.

#### 8.1.1 Panchayat mapping QC (record: `outputs/layer2/layer2_qc.json`)

* Total ROI Panchayats: **87,735**
* Mapped Panchayats: **86,103 (98.1%)**
* Direct/area-based joins: **84,633**
* Nearest-cell fallbacks within 20 km: **1,470** (median fallback distance
  **11.7 km**, max 20.0 km — no Panchayat is placed beyond the 20 km cap)
* Unmapped Panchayats: **1,632**, primarily off-grid or coastal-edge cases

#### 8.1.2 Panchayat rainfall values — the served numbers, restated

The QC record's own rainfall statistics and the values the API returns today come
from **different Layer-1 grids**, so both are given explicitly rather than
quietly merged:

| | `layer2_qc.json` (Layer-2 run) | Served today (`outputs/prediction_test.npz`) |
|---|---|---|
| Grid behind it | the pre-hardening Layer-1 grid | the deployed E+F ensemble grid (grid sha `09410cf6…`, §4.3) |
| min / mean / max (mm/day) | 0.00 / 4.01 / **163.28** | 0.00 / 9.97 / **193.96** |
| Panchayat rainfall values | area-weighted over polygons, per date | the grid value at each Panchayat's **own polygon point** — the nearest cell, which is what `GET /api/weather` and the advisory serve |

* The served column covers **10,290,212** finite (Panchayat, day) pairs out of
  87,735 × 122 = 10,703,670 possible. It was computed the way the API resolves a
  Panchayat (`live_infer.sample_grid` — nearest cell to the polygon's
  representative point) and spot-checked against
  `data_store.panchayat_grid_value` for specific ids and dates.
* **Masked cells are not given a number.** **3,389** of the 87,735 ROI Panchayats
  sit on cells the land mask excludes (coastal strip / sea). They have no value on
  **any** date, and the API answers `422 masked_cell` for them rather than
  inventing a figure — so 84,346 Panchayats resolve to a real number and 3,389 do
  not.
* **Therefore the committed `panchayat_summary.csv` (max 163.28 mm/day) is a stale
  season aggregate.** Re-run `scripts/layer2_panchayat_mapping.py` before quoting
  season totals or the Panchayat search list's seasonal means from it; anything
  the API serves is already on the current grid.

Layer 2 processing is complete and validated. The remaining limitation is
spatial coverage of source polygons, not an unfinished processing step.

## 9. Layer 3 — the advisory layer (rules, API, dashboard)

Layer 3 turns one value from Layer 1 into an explained, actionable advisory for a
named Panchayat, crop and growth stage — and **refuses to answer when it cannot
verify what it is advising on**. The division of labour is deliberate: a
deterministic rule engine decides what the advisory says, an LLM may only rephrase
it, and a guard rejects any rephrase that introduces a number the rules did not
produce.

### 9.1 One request, end to end

```
GET /api/advisory?panchayat_id&crop&stage&date
        |
        v
routes_data : store().panchayat_grid_value(id, date)      <- Layer 1, VERIFIED
        |        outputs/prediction_test.npz read at that Panchayat's OWN polygon
        |        point (panchayat_index.csv) -> or 409 / 422 with the reason
        v
aux_layers  : clay + NDVI + land cover for that same cell  <- data/aux_data/*.npz
        |
        v
advisory.py : 9 deterministic rules -> fired? evaluable? severity, margin to the
        |      threshold, what would flip it, per-rule farmer advice
        v
evaluate()  : risk band + headline + CONFIDENCE (from which inputs arrived)
        |
        v
Groq (optional) rephrases the trace -> faithfulness guard -> else the template
        |
        v
JSON: risk_level, headline, actions, per-rule evidence, aux, verification block
```

### 9.2 The rule engine — nine rules, every threshold

| Rule | Fires when | Severity → UI band | Data it needs |
|---|---|---|---|
| `R1_HEAVY_RAIN` | rainfall ≥ **64.5 mm/day** (IMD's heavy-rain lower bound) | **high** → alert | Layer-1 rainfall |
| `R1B_SUBSTANTIAL_RAIN` | **24.5** ≤ rainfall < 64.5 | **medium** → warning | Layer-1 rainfall |
| `R2_IRRIGATION` | rainfall < **3.0 × window_days** **and** soil moisture < the crop's dry level | **medium** → warning (or **low** if soil moisture is missing: a rain-only check, labelled as weaker evidence) | rainfall + soil moisture |
| `R3_HEAT_STRESS` | tmax ≥ the crop's heat threshold **and** the stage is one of that crop's heat-sensitive stages | **high** → alert | temperature + **a known stage** |
| `R4_DISEASE` | humidity ≥ **85 %** **and** 22 ≤ temperature ≤ **32 °C** | **medium** → warning | humidity + temperature |
| `R5_LODGING` | wind ≥ **40 km/h** **and** the crop is tall | **medium** → warning | wind + crop |
| `R6_VEGETATION` | NDVI < **0.30** (composite for the current month) | **low** → watch, **medium** if rainfall is also low | NDVI (sawn layer) |
| `R7_SOIL_DRAINAGE` | rainfall ≥ **24.5 mm** **and** clay ≥ **350 g/kg** (poorly drained soil) | **medium** → warning, **high** if rainfall ≥ 64.5 | clay (aux) + rainfall |
| `R8_LANDCOVER` | dominant land cover is built-up / water / bare, **or** cropland fraction < 0.20 | **low** → watch | land-cover aux |

Per-crop parameters live in `CROP_PARAMS` (`backend/advisory.py`): wheat, rice,
maize, mustard, cotton, bajra and pulses, each with its heat threshold
(32–40 °C), its heat-sensitive stages, its dry-soil level, its typical disease and
whether it is tall. The numbers are **documented placeholders** for the prototype,
not calibrated agronomy — stated as such in the module.

Severity maps to what the UI shows: `none → info`, `low → watch`, `medium →
warning`, `high → alert`. The rainfall risk band is shared with the frontend:
`< 2.5 → no_rain`, `< 10 → light`, `< 25 → moderate`, `< 50 → heavy`, else
`very_heavy` (both sides use the same cut-points, so a chip cannot disagree with
the advisory).

### 9.3 What every rule returns — the explainability contract

Not a boolean. Each of the nine rules returns its `fired` flag, an **`evaluable`**
flag, its `severity`, the `inputs` it read, the `condition` as text, a
**`margin_pct`** (how close the value was to the threshold), a **`flip_hint`**
(what would have changed the answer) and its own farmer-facing `advice_en`. That is
what `/api/explain` and the dashboard's XAI panel render — the explanation is
computed, not generated.

A rule with a missing input reports **`evaluable: false`** with the reason instead
of staying silent or guessing. The sharpest example is `R3_HEAT_STRESS`: a heat
threshold only applies inside a crop's sensitive window, so an **unknown crop stage
makes the rule unevaluable** rather than assuming the crop is heat-sensitive (this
was a real defect, fixed — see `docs/LOGICAL_VALIDATION_REPORT.md` A-6, and
`backend/tests/test_advisory.py`).

### 9.4 Confidence is computed, never asked of the model

The trace carries a `confidence` in [0.3, 1.0] with `confidence_reasons` listing
every deduction: **−0.1** for each of soil moisture / humidity / wind / crop stage
missing, **−0.2** if temperature is missing entirely (or −0.1 if only the mean is
available, which can under-detect heat stress), **−0.1** if the aux layers are
unavailable for the cell, **−0.1** if any fired rule sits within 10 % of its
threshold, and **−0.1** if the downscaling attribution leaves a large unexplained
residual. The minimum is 0.3 — the advisory never claims more certainty than its
inputs support.

### 9.5 Text: Groq rephrases, Sarvam translates, neither decides

* **Groq** is called through an **ordered multi-model chain** with cooldowns: if a
  model fails or is rate-limited the next one is tried, and the attempts are
  returned in the response. It receives the deterministic trace and returns
  prose — it cannot add, remove or re-rank a rule.
* **Faithfulness guard:** a rewrite is accepted **only if it introduces no number
  absent from the trace** (`is_faithful`: every number in the message must already
  appear in the rule evidence). A rephrase that invents a figure is discarded and
  the deterministic template is used instead.
* **Sarvam** (`mayura:v1`) translates the advisory for `lang=hi-IN` and other
  supported codes; a missing key returns a clear 502 and leaves the server up.
* **No keys, fully functional:** with empty `GROQ_API_KEY` / `SARVAM_API_KEY` the
  advisory is the deterministic text and `/api/explain` reports
  `provider: "rules"`. Nothing in Layer 3 requires an external service.

### 9.6 The API surface (15 route entries, all in `backend/`)

| Route | What it answers, and from where |
|---|---|
| `GET /` , `GET /health` | health; `panchayats_loaded` is correct even on a cold process |
| `GET /api/panchayats?q=` | Panchayat search over `outputs/layer2/panchayat_summary.csv` (real `mapping_method`, `n_cells`; unmapped rows hidden) |
| `GET /api/geocode` | a Panchayat's own LGD polygon point if known (exact), else a district/block centre **flagged as approximate**; a name shared by several Panchayats returns **409** with the candidates |
| `GET /api/weather` | Open-Meteo temperature/humidity + DEM elevation at a lat/lon |
| `GET /api/metrics` | the ablation table, the model card **and the Layer-1 provenance** (grid + checkpoint sha256, members, weights, parameter count) |
| `GET /api/advisory` | the rule engine **plus** the committed soil/NDVI/land-cover layers, on **rainfall verified against Layer 1** for that Panchayat and date (below) |
| `POST /api/advisory` | the same engine for deliberate what-if runs not tied to the stored field (temperature and humidity are caller-supplied and labelled as such) |
| `POST /api/advisory/block` | up to **60** Panchayats in one call, for a block-level view |
| `POST /api/explain` , `POST /api/explain/generic` , `GET /api/explain/status` | rule-derived factors, optionally Groq-rewritten (falls back to `provider: "rules"`); status reports the model chain |
| `POST /api/translate` | Sarvam translation of advisory text |
| `POST /auth/` , `POST /auth` | the NL query path: resolves the named Panchayat, serves the Layer-1 value for the date, and returns an answer with its sources |

The verification contract is the part worth reading twice: `GET /api/advisory`
resolves the Panchayat's own Layer-1 value through
`data_store.panchayat_grid_value()` and then

| situation | response |
|---|---|
| supplied `rainfall_mm` matches the stored value (tolerance 0.011 mm, half a rounding step) | 200, `verification.rainfall = verified_against_layer1` |
| `rainfall_mm` omitted | 200, the stored value is used — `resolved_from_layer1` |
| supplied value disagrees | **409** with `supplied_mm`, `expected_mm`, `difference_mm` |
| date outside the served grid / masked cell / unknown Panchayat | **422** with `reason` = `date_unavailable` / `masked_cell` / `unknown_panchayat` |
| no data layer mounted (text-only deployment) | `rainfall_mm` is required, and 200 is labelled `unverified_no_data_source` |

So an advisory can never be produced for a rainfall figure that is not the one
the model actually produced for that place and day. `backend/README.md` documents
the same contract route by route.

### 9.7 The dashboard (`Frontend/`)

Next.js on port 3000, reading the backend at `NEXT_PUBLIC_API_URL` (default
`http://localhost:8000`). Panchayat search (with district/block disambiguation),
the rainfall card and map for a chosen date, the advisory panel (risk chip, action
bullets, the evidence row with the exact rainfall and date used), the XAI panel
(which rule fired, by how much it missed the threshold, and what would flip it) and
a model card fed by `/api/metrics`.

The frontend holds **no second copy of the rules**. When the backend refuses
(409/422) the panel shows an explicit *"Advisory withheld"* with the reason, and if
the service is unreachable it shows *"Advisory unavailable"* with a retry — it never
substitutes its own guess. (A client-side fallback rule table used to exist and
disagreed with the engine in the band a farmer actually sees; it was removed —
`docs/LOGICAL_VALIDATION_REPORT.md` A-5, guarded by
`backend/tests/test_frontend_contract.py`.)

### 9.8 How Layer 3 is verified

`python -m pytest backend/tests` — **69 tests**, no network and no keys needed:
the rule engine at every threshold boundary, the advisory-verification contract
(409/422 for every refusal reason), the artifact-consistency contract (the served
grid matches the provenance manifest), the aux wiring and the frontend contract.
On top of that, `qa/qa_logic_advisory.py` sweeps **every** threshold at T±ε and
runs 11 one-input perturbations to prove each input is causally used, and
`qa/qa_logic_http.py` checks identity binding and cache isolation over live HTTP.

## 10. Limitations

* **`GET /api/advisory` accepts a smaller crop-stage vocabulary than `POST`**
  (finding A-9). The GET query enum is `general | sowing | vegetative | flowering
  | ripening | harvest`, while the POST body takes any stage string. `grain_filling`
  and `maturity` — the two stages the heat rule depends on most — are therefore
  reachable through POST but rejected as **422** by GET (`ripening` covers
  `maturity` via a synonym; `grain_filling` has no equivalent), so a GET caller
  cannot express the stage that makes `R3_HEAT_STRESS` fire. The rules themselves
  are correct (§9.2/§9.3); the fix is to widen `StageQ` in `backend/main.py`.

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
* Heavy rain remains hard for MAE-trained models; the weighted-loss variant and
  the deployed E+F ensemble mitigate it (F1 ≥ 50 mm 0.284 vs the baseline's
  0.234). Be precise about the tail, though: it is still largely missed — recall
  for ≥ 100 mm/day events is ~0.005, and cells observed above 100 mm are
  predicted at ~35 mm. The ensemble beats the baseline at every reported
  threshold; it does not make extreme rainfall well predicted. No
  probabilistic/uncertainty output yet.
* The U-Net is deliberately small (117,329 params for the 5-channel/width-16 deployed
  model — the value is stamped in `outputs/metrics/layer1_manifest.json` and served by
  `/api/metrics` as `n_parameters`, so it can be checked rather than trusted). Operational
  use would consume IMD Block forecasts as coarse input, whose error propagates through
  Layer 1.
