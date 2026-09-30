# HANDOVER — Layer-1 Deccan dataset → model training → Layer-2

**For:** the person doing model training (and later Layer-2) on this dataset.
**State:** the dataset is COMPLETE and VERIFIED, **and the Deccan models are
trained** — rows A–F plus the deployed E+F ensemble (`models/ensemble.json`,
selected on the validation split only; test MAE 8.43 mm against the bilinear
baseline's 9.60 mm). §2 is kept as the exact, reproducible recipe that produced
them. Everything below is what you need, in the order you need it.

---

## 0. TL;DR — your first hour

### Getting the data (a fresh clone has NO data!)

The `.npy` arrays and raw downloads are **not in git** (too big / reproducible):

* **Deccan model-ready arrays:** copy `data.zip` (~929 MiB) from the shared
  OneDrive folder, then at the repo root run `unzip data.zip` — it restores
  `data/processed/` (`X/Y/M_{train,val,test}.npy` + `meta.json`). Or rebuild
  from raw — §6.
* **Layer-2 panchayat polygons:** `python scripts/fetch_lgd_panchayats.py`.

Already committed (nothing to do): `data/processed/meta.json`, the aux layers
(`data/aux_data/*.npz`), the reports, all scripts and these docs.

```bash
cd weather-downscaling
# Windows Git Bash: ALWAYS use the venv python (system python lacks tifffile)
PY=.venv/Scripts/python.exe

$PY scripts/verify_dataset.py          # ~1 min; must print ALL CHECKS PASSED
$PY scripts/train.py --help            # see all training knobs
$PY scripts/train.py --channels imd_rain --out-name model_b          # smoke test (~small)
$PY scripts/train.py --channels imd_rain,dem,era5_t2m,era5_t2m_max,era5_dewp \
     --out-name model_d                                               # the full model
$PY scripts/train.py --channels imd_rain,dem,era5_t2m,era5_t2m_max,era5_dewp \
     --loss weighted --out-name model_e                     # heavy-rain weighted
$PY scripts/train.py --channels imd_rain,dem,era5_t2m,era5_t2m_max,era5_dewp \
     --loss weighted --weight-rain-mm 50 --weight-mult 10 --out-name model_f
$PY scripts/ablation.py                # A..F + EF table, val-only selection
                                       # -> models/ensemble.json (deployed)
                                       #    models/best_model.pt (fallback)
$PY generate_pred.py                   # -> outputs/prediction_test.npz (served grid)
$PY scripts/evaluate.py --models model_b,model_c,model_c_weighted,model_d,model_e,model_f
```

If `verify_dataset.py` fails, STOP — re-read §6 (rebuilds) before training.

> Training path was smoke-tested on these exact arrays (2 epochs, CPU,
> ~8 s, checkpoint reload + `predict_mm` verified) — §2.1. Just run it.

### Layer 2 in three commands (spoon-feed)

```bash
PY=.venv/Scripts/python.exe
$PY scripts/fetch_lgd_panchayats.py        # one-off: LGD panchayat polygons (LGD-derived, CC0)
$PY scripts/infer.py --imd data/raw/imd/ind2022_rfp25.nc --date 2022-07-10   # -> prediction/infer_2022-07-10.npz
$PY scripts/layer2_panchayat_mapping.py
# -> outputs/layer2/panchayat_weather.csv  (>100 MB, gitignored: the backend
#    serves per-date values from outputs/prediction_test.npz instead)
```

Run `train.py` first (§2) so `infer.py` has a checkpoint; Layer 2 then needs no
further input.

---

## 1. What exists (inventory)

### 1.1 Model-ready arrays — `data/processed/` (the only thing training reads)

| file | shape | meaning |
|---|---|---|
| `X_train.npy` | (366, 5, 285, 200) float32 | 5 channels × 285×200 fine grid, one slice per day |
| `X_val.npy` / `X_test.npy` | (122, 5, 285, 200) | monsoon 2021 / 2022 |
| `Y_*.npy` | (n, 1, 285, 200) | CHIRPS 0.05° daily rain (mm/day) — the **reference target** |
| `M_*.npy` | (n, 1, 285, 200) | 1 = valid land target, 0 = excluded (sea / coastal strip). **Always multiply loss/metrics by M** |
| `meta.json` | — | provenance, dates, grid, split, land-mask rules, normalization |

* Channels (order matters): `imd_rain, dem, era5_t2m, era5_t2m_max, era5_dewp`.
  (`era5_wind` is NOT included — the 2018 ERA5 cache had 122 fully-missing
  wind days; the channel was skipped, not imputed.)
* **Normalization is already applied inside X** (train-only stats):
  `rain_scale_mm = 100.0`, `elev_scale_m = 1222.65`. To get physical mm:
  `X[:,0] * 100.0` = IMD rain in mm. Targets: train in normalized units too
  (`Y / rain_scale` — `train.py` already does this).
* Valid target pixels: 44,243 fine cells × days → 16,192,938 (train) /
  5,397,646 (val) / 5,397,646 (test). Split: train 2018+2019+2020,
  val 2021, test 2022 — **temporal split, no leakage**; normalization stats
  are train-only.
* 3,007 fine cells are excluded from Y/M by design (sea + coastal strip within
  one CHIRPS cell of CHIRPS-ocean). Excluded ≠ imputed. Don't "fix" this.

### 1.2 Auxiliary layers — `data/aux_data/` (NOT model inputs; for Layer-2 & analysis)

| file | contents | key fields |
|---|---|---|
| `admin/grid_admin_map_deccan.npz` | every land fine cell → state / district / block (GADM 4.1 L3 subdistrict = block tier) | `fine_i, fine_j, fine_lat, fine_lon, state_ids, state_names, district_ids, district_names, subdistrict_ids, subdistrict_names, subdistrict_idx, nearest_dist_deg` |
| `admin/admin_master_deccan.geojson` | block-tier polygons (bbox-clipped) + state/district polygons' IDs/names/centroids | GADM UIDs, GID_*, NAME_* |
| `soil_soilgrids_deccan.npz` | sand, clay, ocd, phh2o, bdod @ 5–15 cm on the 1,890 land **coarse** cells | `values (1890,5)` raw SoilGrids units; `fine_values (47250,5)` = each fine pixel inherits its parent coarse value; `fine_i, fine_j` |
| `ndvi_monthly_deccan.npz` | 20 monthly composites (Jun–Sep 2018–2022; days 5 & 25) | `ndvi (20, 47250)`, `months`, `valid_frac_per_month`, `fine_i, fine_j` |
| `lulc_fractions_deccan.npz` | 6 land-cover area fractions + dominant class per land fine cell (WorldCover 2021 @ 10 m) | `fractions (47250, 6)`, `fraction_names`, `dominant_class`, `class_legend` (JSON), `fine_i, fine_j` |
| `soilmoisture_daily_deccan.npz` | **OPTIONAL / PENDING (quota-blocked)** — daily volumetric soil moisture (m3/m3, 0–7 cm) on the ERA5 lattice, 610 monsoon days — **Layer-3 advisory input, not a model channel** | when materialized: `dates (610,)`, `lat, lon`, `soil_moisture_0_to_7cm_mean (610,17,13)`, `note` |
| `build_summary_deccan.json` | machine-readable build stats for all layers | — |

**Shared indexing:** all aux layers EXCEPT soil moisture use the SAME 47,250
land cells, indexed by `(fine_i, fine_j)` on the 285×200 fine grid — identical
arrays across files (verified by `verify_dataset.py`). To go from a dense
`(H, W)` field to the cell list: `values[fine_i, fine_j]`. Reverse (sparse →
dense): scatter by `[fine_i, fine_j] = cell_values`. Soil moisture is kept at
its native ERA5 lattice `(610, 17, 13)` — resample with the same bilinear
routine `preprocess.py` uses for era5 channels, or sample lattice values
nearest a fine cell (fields are smooth).

**⚠️ Object arrays:** the admin map stores ID/name arrays as numpy object
arrays — load with `np.load(path, allow_pickle=True)`.

**Units / caveats per layer:**
* **Soil** raw SoilGrids mapped units (×10 factors): sand/clay/ocd/pH ÷10
  (→ %, %, g/dm³, pH), bdod ×1000 (→ kg/m³). `units_note` inside the file.
  Missing 8.8–9.4% per property = **genuine SoilGrids nulls** (SW coastal
  strip + 2 inland points; verified NOT rate-limiting). Not imputed — use
  NaN-aware stats.
* **NDVI** unitless −1..1 (observed −0.064..0.845); 8.0% missing = monsoon
  cloud, not error. Per-month valid fractions in the file.
* **LULC** fractions 0..1: cropland(40), forest(10+95), water(80), built-up(50),
  grassland(30+20), barren(60+70+100). Two honest caveats: (a) 2.86% of cells
  (1,349) have NO class data — their entire stencil is WorldCover ocean
  nodata (sea fringe of coastal IMD grid boxes); all are M=0 cells, so they
  never affect training. (b) the 6 fractions do NOT sum to 1 in the 104 cells
  containing wetland (class 90 has no fraction column). Covered cells =
  `fractions.sum(axis=1) > 0`.

### 1.3 Reports — `data/reports/`

* `data_dictionary_deccan.md` — every field, unit, semantics (read this).
* `coverage_report_deccan.md` — region, days, grids, admin/soil/NDVI/LULC
  coverage, model-ready summary.
* `missingness_report_deccan.md` — every missingness number and its cause.
* `verify_dataset_deccan.json` — the QA result (shapes, mask semantics,
  cross-layer alignment, ranges).

---

## 2. How to train (recommended protocol)

The model is a ~150k-param residual U-Net: it predicts a *correction* to the
bilinear IMD baseline, so it can never do worse than the baseline by
construction. Training is cheap (CPU-feasible; GPU makes ablations fast).

### 2.1 Rows to run (same naming as the pilot, so tables stay comparable)

```bash
PY=.venv/Scripts/python.exe
$PY scripts/train.py --channels imd_rain --out-name model_b
$PY scripts/train.py --channels imd_rain,dem --out-name model_c
$PY scripts/train.py --channels imd_rain,dem --loss weighted --out-name model_c_weighted
$PY scripts/train.py --channels imd_rain,dem,era5_t2m,era5_t2m_max,era5_dewp --out-name model_d
$PY scripts/ablation.py                 # builds A (baseline) / B2 (bias-corr) rows,
                                        # loads the B/C/Cw/D checkpoints,
                                        # applies the FIXED selection rule -> best_model.pt
$PY scripts/evaluate.py --models model_b,model_c,model_c_weighted,model_d
```

* `model_d` = all 5 channels. `--loss weighted` = heavy-rain-weighted MAE.
* `train.py` normalizes targets by `rain_scale`, applies the M mask, does
  early stopping on val, and saves `models/<name>.pt` + history.
* **Smoke-tested on these exact deccan arrays** (CPU torch 2.14, 2 epochs,
  `--val-subset 16`): ~8 s total, full checkpoint written/reloaded cleanly and
  `predict_mm` produced sane 0–90 mm fields. Full training will take minutes
  per row on this machine — no GPU required.
* Keep the pilot's row semantics: A = bilinear baseline (no learning),
  B2 = per-cell bias-corrected baseline (a **negative result** on the pilot —
  bias doesn't transfer across monsoon years; keep it as the honest control),
  B/C/Cw/D as above.

### 2.2 The fixed evaluation protocol (do NOT improvise)

* **Selection rule (fixed before touching test, validation only):** lowest val
  MAE; rows within **0.75 mm** are tied (the noise level of this split) and the
  tie is broken by val heavy-rain skill (**F1 at ≥25 mm**). This replaced the
  original "0.1 mm, broken by correlation" rule, which selected a model with
  F1 ≥ 50 mm of 0.015 — worse than the baseline it must beat. Test is evaluated
  ONCE for the selected row.
* **Metrics** (emitted by `evaluate.py`): MAE, RMSE, per-day spatial
  correlation, plus event detection precision/recall/F1 at **10 / 25 / 50
  mm/day** thresholds. All masked by M.
* **Wording per SIH guidance:** never "accuracy = X%". Report "MAE lower by
  X% vs baseline", "correlation improves 0.29 → 0.33", etc.
* **Also report aggregated horizons** (quick win): score 3-day / weekly /
  monthly means in addition to daily. IMD–CHIRPS daily agreement over the
  Deccan is limited (coarse domain-mean daily corr ≈ 0.356 — part of every
  error is product disagreement, not model error); aggregated scores show the
  real downscaling value and will be your best presentation slide.
* Report **both** the MAE-trained model_d and weighted-loss variant on the
  heavy-rain trade-off (pilot: weighted cost ~0.7 mm MAE, huge F1≥25 gain).
  For an agriculture audience, heavy-rain detection usually matters more.

### 2.3 What "good" looks like (expectations, not promises)

* Pilot (Western Ghats, steeper orography): D beat the bilinear baseline by
  ~16% test MAE; correlations rose monotonically B→C→D.
* Deccan is flatter — DEM's marginal value may be smaller; ERA5 humidity
  channels and the 3× more training days are the bigger levers. Do not
  compare Deccan numbers to pilot numbers; they are different domains.

---

## 3. After training — outputs Layer-2 will consume

`evaluate.py` (and `infer.py` for arbitrary new dates) write
`prediction/<model>_<split>.npz` + `_schema.json`:

```
dates        (n,)       YYYY-MM-DD
latitude     (H,)       ascending fine-grid lat centers (deg N)
longitude    (W,)       ascending fine-grid lon centers (deg E)
rainfall_mm  (n, H, W)  daily rainfall (mm/day)
```

This is done: README §4 now carries the Deccan numbers (with the pilot table kept
and re-labelled), and `outputs/metrics/*` — specifically
`ablation_summary.md` and `layer1_manifest.json` — is the artefact of record.

---

## 4. Layer-2: block & panchayat mapping (how to use the aux data)

Run order: `train.py` → `generate_pred.py` (emits the served grid
`outputs/prediction_test.npz`) → `scripts/layer2_panchayat_mapping.py`. Block
aggregation (§4.1) works today with no new GIS; the Panchayat tier (§4.2) needs
one external file, now prepared by a single command.

### 4.1 Cell → block aggregation (works TODAY, no new GIS needed)

GADM L3 subdistricts ARE the block tier. Every land cell already carries its
block — aggregating a prediction field to blocks is a group-by:

```python
import numpy as np
from collections import defaultdict

adm = np.load("data/aux_data/admin/grid_admin_map_deccan.npz", allow_pickle=True)
ii, jj = adm["fine_i"], adm["fine_j"]
sub_ids, sub_names = adm["subdistrict_ids"], adm["subdistrict_names"]

pred = np.load("prediction/model_d_test.npz")   # contract file
rain = pred["rainfall_mm"]                       # (n, H, W)

# per-block cell lists, computed once
cells_by_block = defaultdict(list)
for k in range(len(ii)):
    cells_by_block[str(sub_ids[k])].append((ii[k], jj[k]))

t = 0                                            # any date index
block_mean = {sid: float(np.nanmean([rain[t, i, j] for i, j in cells]))
              for sid, cells in cells_by_block.items()}
# join names: dict(zip(map(str, sub_ids), map(str, sub_names)))
```

Faster vectorized alternative: `np.bincount(sub_idx, weights=rain[t][ii, jj])`
/ `np.bincount(sub_idx)` per day (valid where M==1; combine with the M mask of
the same date). Zonal means with the actual polygons
(`admin/admin_master_deccan.geojson` + rasterio/geopandas) give nearly
identical results (pilot spot-check: 0.68 PIP agreement for nearest-point
assignment) — use the point-based mapping first, polygons for final maps.

### 4.2 Panchayat tier — prepared with one command

* GADM has no panchayat/village tier. The LGD panchayat boundaries Layer 2 needs
  are now **prepared by a single command** (no manual state-by-state scraping):

  ```bash
  python scripts/fetch_lgd_panchayats.py           # ~368 MB download, verifies sha256,
                                                  # clips to the deccan ROI -> LGD_Panchayats.parquet
  python scripts/fetch_lgd_panchayats.py --verify  # re-check the local file
  python scripts/fetch_lgd_panchayats.py --full    # keep all-India instead of clipping
  ```

  The Layer-2 scripts (`scripts/layer2_panchayat_mapping.py`,
  `scripts/check_blocks.py`) then run against this Deccan config via
  `layer2_config.py`; without the file they exit with a clear error (or pass
  `--panchayats` / `--input`).

#### LGD Panchayat input — exact path, source, schema

**Exact path the script reads:**

```
data/raw/administrative/panchayat/LGD_Panchayats.parquet
```

Override with `--panchayats <path>` if it lives elsewhere.

**Source (verified).** Release tag **`admin/panchayats`** of
<https://github.com/yashveeeeeeer/india-geodata> — a CC0 (public-domain)
GeoParquet aggregating <https://github.com/ramSeraph/indian_admin_boundaries>,
which sources the **official Local Government Directory** (LGD, Ministry of
Panchayati Raj) plus ISRO Bhuvan. LGD (<https://lgdirectory.gov.in>) remains the
upstream authority and publishes panchayats **state-wise with no bulk API**; the
release above is the convenience distribution of that same data and carries the
real LGD codes (`gpcode`, `blklgdcode`), so it is the authoritative tier.

Verified locally: asset `LGD_panchayats.parquet`, 368,147,580 bytes, sha256
`d1585c16…6400` (checked by `fetch_lgd_panchayats.py`); 319,287 features /
226,985 unique Gram Panchayats, CRS `OGC:CRS84` (= EPSG:4326), 99.7% valid
geometry, all six required columns present with 0 nulls. ~17% of rows carry an
**empty `gpcode`** (unmapped GPs) — `load_panchayats()` drops those rows
automatically. The prepared, clipped file is 116,126 features / 87,736 GPs /
14 states (179 MB, ZSTD), already EPSG:4326.

**Fallback boundary sources (NOT equivalent to LGD).** GP polygons from
data.gov.in, Datameet, or a state GIS portal are **geometry-only substitutes**:
they are *not* authoritative and their codes do **not** match LGD. Use them
knowingly, record the substitution, and never present them as LGD.

**Required schema (exact — the script hard-codes these names).** The file must
be a GeoParquet with a `geometry` column plus these attributes:

| column | meaning | becomes |
|---|---|---|
| `gpcode` | Gram-Panchayat code | dissolve key / `panchayat_id` |
| `gpname` | Gram-Panchayat name | `panchayat_name` |
| `stname` | state name | `state` (used by `--state`) |
| `dtname` | district name | `district` |
| `blklgdcode` | block LGD code | `block_id` |
| `blkname` | block name | `block_name` (used by `--block`) |

Rows with an empty `gpcode` are dropped; invalid geometries are repaired with
`buffer(0)`.

**CRS.** A CRS is required: if the file has none set, the script **assumes
`EPSG:4326`**; otherwise it reprojects to `EPSG:4326`. Write it in `EPSG:4326`
to be unambiguous.

**The 15 states the current Deccan grid actually covers** (derived from
`data/aux_data/admin/grid_admin_map_deccan.npz` — this is the exact set; fetch
only these): **AndhraPradesh, Chhattisgarh, DadraandNagarHaveli, DamanandDiu,
Goa, Gujarat, Karnataka, Kerala, MadhyaPradesh, Maharashtra, Puducherry,
Rajasthan, TamilNadu, Telangana, UttarPradesh**.
(GADM IDs in the same order: IND.2_1, IND.7_1, IND.8_1, IND.9_1, IND.10_1,
IND.11_1, IND.16_1, IND.17_1, IND.19_1, IND.20_1, IND.27_1, IND.29_1,
IND.31_1, IND.32_1, IND.34_1.) The admin map stores these names **without
spaces** (e.g. `DadraandNagarHaveli`, `MadhyaPradesh`, `TamilNadu`) — normalise
names before name-joining to LGD.

**Convert a source Shapefile / GeoJSON to the required parquet:**

```python
import geopandas as gpd, pandas as pd, glob

frames = []
for path in glob.glob("data/raw/administrative/panchayat/*.shp"):  # or *.geojson
    g = gpd.read_file(path)
    g = g.rename(columns={            # source names vary; map them to the schema
        "GP_CODE": "gpcode",  "GP_NAME":  "gpname",
        "STATE":   "stname",  "DISTRICT": "dtname",
        "BLK_LGD": "blklgdcode", "BLOCK": "blkname",
    })
    frames.append(g[["gpcode", "gpname", "stname", "dtname",
                     "blklgdcode", "blkname", "geometry"]])

g = pd.concat(frames, ignore_index=True)   # one parquet for the whole grid

if g.crs is None:                # script assumes EPSG:4326 when CRS is unset
    g = g.set_crs("EPSG:4326")
else:
    g = g.to_crs("EPSG:4326")

g = gpd.GeoDataFrame(g, geometry="geometry", crs="EPSG:4326")
g.to_parquet("data/raw/administrative/panchayat/LGD_Panchayats.parquet")
```

Then run the unchanged Layer-2 step:

```bash
.venv/Scripts/python.exe scripts/layer2_panchayat_mapping.py
# -> outputs/layer2/panchayat_weather.csv   (>100 MB; gitignored by design)
```
* The admin map is **LGD-joinable by (state, district, block) NAME** — GADM
  IDs are NOT LGD codes. Path: fetch the LGD panchayat layer per state
  (15 states are hit by the grid), intersect each state's block polygons
  (`admin_master_deccan.geojson`) with panchayat polygons ONCE, assign each
  fine cell its panchayat exactly like `build_admin` did (KD-tree nearest
  representative point + spot-check), and ship a v2 `grid_admin_map`.
* Then the same group-by as §4.1 yields panchayat-level rainfall. Advisory
  context comes from soil/NDVI/LULC (drought proxies, vegetation state,
  dominant land use) — all already on the same cell indexing.

### 4.3 Operational loop (later)

`scripts/infer.py --imd <new IMD file> --date YYYY-MM-DD` builds the channel
stack from raw inputs and emits the same `prediction/` contract — no
retraining. Feed it the operational IMD Block forecast as the coarse input
when available; its error propagates through Layer 1 (documented limitation).

---

## 5. Provenance & semantics — where to look before changing anything

* `data/processed/meta.json`: `resampling_operations` (how every channel was
  resampled), `land_mask` (who enters Y/M and why), `channel_semantics`,
  `normalization`, `split`, `provenance` (source files + dates fetched).
* `data/aux_data/build_summary_deccan.json`: per-layer build stats (tiles, batches,
  months, coverage, nodata counts).
* `scripts/config.py`: REGIONS, YEARS (2018–2022), split, SOILGRID_*, NDVI_*,
  WORLDCOVER_*, all dirs. Single source of truth.

---

## 6. Rebuilding pieces (all resumable/cached; you rarely need any of this)

```bash
PY=.venv/Scripts/python.exe
$PY scripts/preprocess.py                       # rebuild processed/ (~minutes; overwrites data/processed)
$PY scripts/build_aux.py --region deccan --skip-admin --skip-soil --skip-ndvi   # LULC only
$PY scripts/build_aux.py --region deccan --report-only                          # refresh reports
$PY scripts/verify_dataset.py                   # MUST pass after any rebuild
```

* **WorldCover (LULC):** 25 COG tiles (3°×3°, 36000² px) in `data/raw/lulc/`.
  Reads are **spatially windowed** (zarr chunked reads of only the 1024² COG
  chunks overlapping the region, 4096-px strips) — never decompress a full
  tile. Cached full re-run ≈ 2.5 min; with downloads ≈ 20 min.
* **SoilGrids:** 30 batch caches in `data/raw/soil/batches/`; API params in
  config (`SOIL_BATCH=64`, `SOIL_WORKERS=16`). `scripts/retry_soil_nan.py`
  diagnoses NaNs (they are genuine service nulls, not failures).
* **NDVI:** raw day-slices cached in `data/raw/vegetation/slices/`.
* **ERA5:** per-year caches in `data/raw/era5/`; Open-Meteo quota resets
  hourly (`ERA5_REQUEST_SLEEP=20` s between requests). Only needed if you
  back-fill wind or add years.
* **Admin:** GADM zips cached in `data/raw/admin/`.

Environment notes (Windows Git Bash):
* Use `.venv/Scripts/python.exe` (Python 3.14 + numpy/requests/h5py/tifffile/
  PIL/scipy/zarr). System python lacks tifffile.
* Long jobs: run resumable slices (`timeout 560 ... > log 2>&1`) and **never
  pipe long-running jobs to `head`/`tail`** (SIGPIPE silently kills them).
* `data.zip` (~929 MiB) at the repo root is the **current Deccan** model-ready
  archive (`data/processed/`: X/Y/M_{train,val,test}.npy + meta.json; CRC and
  shapes verified). The old Western-Ghats **pilot** archive is kept separately
  as `data_pilot_westernghats_LEGACY.zip` (~1.7 GB) — do not delete either.
* `models/*.pt` and `outputs/metrics/*` in this repo are now the **Deccan**
  run (rows A–F + `ensemble.json`, and the ablation table + provenance manifest).
  The old Western-Ghats pilot artefacts are no longer the ones in place; the
  pilot archive itself is still `data_pilot_westernghats_LEGACY.zip`.

---

## 7. Wind, WorldCereal, SMAP — the three "should we?" items, decided

* **Wind — data present, channel excluded (deliberate).** The 2018 ERA5
  cache was re-fetched through the same downloader (same source/lattice; land
  variables verified bit-identical to the old cache) and the combined raw file
  now has 10 m wind for ALL 610 days, 0% missing. It is still NOT a model
  channel: the frozen arrays were preprocessed with the 5-channel baseline
  (`imd_rain, dem, era5_t2m, era5_t2m_max, era5_dewp`). To train a wind
  ablation: add `era5_wind` to `config.CHANNELS_ALL`, re-run
  `preprocess.py`, and document the new channel order — the old arrays are
  reproducible because the raw caches are intact.
* **ESA WorldCereal (crop type) — evaluated, NOT added.** Only no-auth
  distribution = Zenodo 7875105: global multi-GB ZIPs, each containing 106
  AEZ-zone GeoTIFFs with AEZ-specific season definitions (winter/spring
  cereals, maize-main/second); per-region extraction + season alignment is a
  standalone Layer-3 task, and it does not exist as one clean 2021 layer over
  our box. Decision per the data-freeze rule: document, don't force it. For a
  crop-type signal today, `lulc_fractions_deccan.npz` already separates
  cropland vs forest vs built-up per cell.
* **SMAP soil moisture — substituted with ERA5-Land soil moisture (OPTIONAL /
  PENDING).** Genuine SMAP L4 (spl2smp_e) requires NASA Earthdata auth +
  granule stitching + reprojection (the "time sink" case), so the builder
  (`build_soilmoisture` in `build_aux.py`) uses the Open-Meteo daily ERA5-Land
  volumetric water (0–7 cm) — same underlying model as our era5_* channels,
  same lattice, zero auth, per-(batch, year) cached and resumable, and it
  refuses to write an incomplete file. It is **NOT materialized yet**: the
  free Open-Meteo **daily** request limit was exhausted by the wind repair
  before the ~34 batch-year requests could complete. **Resume (any later
  day, repeat across daily resets until done; each attempt is additive):**

  ```bash
  python scripts/build_aux.py --region deccan --skip-admin --skip-soil --skip-ndvi --skip-lulc
  ```

  If real SMAP is ever required, `build_soilmoisture` is the seam to replace.

## 8. Known caveats (repeat these in any presentation)

1. **CHIRPS is the reference, not ground truth.** IMD vs CHIRPS coarse daily
   corr ≈ 0.356 over the Deccan box; a chunk of every error metric is product
   disagreement. Weekly/monthly agreement is much stronger.
2. CHIRPS' lattice is staggered half a cell; Y is a bilinear sample of it.
   Identical for baseline and models → comparisons remain fair.
3. Monsoon-only (Jun–Sep), 2018–2022. No winter, no pre-2018.
4. Coastal strip (~3,007 fine cells) + sea are structurally excluded from
   Y/M; LULC is nodata over the sea fringe of those cells (2.86%).
5. Soil (8.8–9.4%) and NDVI (8.0%) missingness is genuine-source missingness —
   NaN-aware stats, no imputation.
6. No uncertainty quantification; deterministic point forecasts only.
7. Training/eval on this dataset is **not done** — nothing in
   `outputs/metrics/` describes the Deccan dataset yet.
