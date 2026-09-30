# Logical / Scientific Validation — Weather Downscaling + Panchayat Agro-Advisory

**Subject:** `weather-downscaling` (GitHub `yashkapur0403/weather-downscaling`)
**Build validated:** branch `fix/qa-hardening` @ `4b2bc46` (worktree `.qa-b1`), the branch that
carries the backend, the frontend, the Layer-2 outputs and the fixes from the earlier QA pass.
**Method:** executed code and data, not documentation. Every number below was produced by a script
that is listed in §19. **No application logic was modified while the validation was running** —
the hardening commit landed first (`4b2bc46`), the experiments ran against that frozen build.

---

## 1. Executive Summary

The numeric spine of this product is real and it is correct. Layer 1 → Layer 2 is the strongest part
of the system and it survives every test applied: 13/13 sampled Panchayats across five states and all
three mapping methods returned exactly the value you get by recomputing `round(cell_value, 2)`
from `outputs/prediction_test.npz` at that Panchayat's own coordinate; 54/54 date→index→value traces
were exact; nothing is double-scaled, nothing is negative, no missing cell is imputed, and a masked
cell returns HTTP 422 rather than a fabricated zero. Ambiguous Panchayat names are refused (HTTP 409
with candidates) instead of silently resolved. There is no cache or state bleed, in sequence or under
8-way concurrency.

The advisory engine is also genuinely rule-based, deterministic and auditable. Every one of its ten
thresholds was tested at `T−ε / T / T+ε`: they fire exactly where the code's own `condition` strings
say, with no gap, no overlap and no boundary ambiguity. Crop, stage, temperature, soil clay, NDVI and
land cover each change the outcome when varied **in isolation**, which means the crop calendar and the
auxiliary layers are causally wired, not decorative.

Three things stop this from being a clean bill of health, and one of them is the answer to the
question the judges will actually ask:

1. **The advisory is not provably bound to the Panchayat/date whose rainfall it uses.** `GET /api/advisory`
   takes `rainfall_mm`, `date` and `panchayat_name` from the caller, uses `panchayat_id` only to look up
   auxiliary soil/NDVI/land-cover, and echoes the client's date back as `data_date`. It cannot detect a
   mismatch. Proof: same `panchayat_id`, same `date`, `rainfall_mm=0` → `warning`/R2+R6, `rainfall_mm=999`
   → `alert`/R1+R6; `date=1999-01-01` → HTTP 200 with `data_date=1999-01-01`; `panchayat_id=999999999`
   → HTTP 200 with `aux: null`. (**A-2**)
2. **The frontend can substitute a previous selection's rainfall for a legitimate 0.0 mm.**
   `HomePage.tsx` builds the selected Panchayat with `rainfall_mm: response.prediction.rainfall_mm || selected.rainfall_mm`.
   Zero is falsy in JavaScript, so a genuinely dry answer is replaced by whatever Panchayat was selected
   before. This is not hypothetical: `NARASAPURAM`, `BHEEMUNIPALLI` and `ALUR` all return exactly
   `0.0 mm` for `2022-07-10` from `/auth/`. The substituted value then drives the displayed number, the
   risk chip, the map marker **and the advisory input**. (**A-1**)
3. **The Layer-2 summary statistics were computed from a different Layer-1 grid version.** `panchayat_summary.csv`
   reports a maximum of `163.2806 mm` — which is the maximum of the *pre-hardening* grid — while the grid
   actually served has a maximum of `193.96 mm`; per-Panchayat recomputation over the same 11 cells gives
   `14.94 mm` against a recorded season mean of `4.89 mm`. Nothing in the repository records which
   Layer-1 version produced those columns, and `/api/panchayats` serves them. (**A-3**)

Separately, a scientific limitation is now quantified and no longer hidden: the model cannot represent
extreme rainfall. For reference cells observed at ≥100 mm/day the model produced ≥100 mm in
**523 of 24,982** cases (recall 0.005), and for cells observed at ≥50 mm/day it predicted a mean of
**35.6 mm against an observed 76.0 mm**. This is the direct cause of the heavy-rain detection problem
that the accompanying fix (`4b2bc46`) addresses by deploying a validation-selected E+F ensemble: that
ensemble now beats the bilinear baseline at **every** reported threshold, including `F1≥50 mm`
(0.284 vs 0.234, up from 0.129 before the fix), while keeping MAE 12.2 % below the baseline.

Findings are separated into the required buckets in **§14.1–§14.7 (A–G)** and the explicit answer to
the final question is in **§18**.

---

## 2. Scope

### 2.1 What was in scope

| Layer | Artefacts exercised |
|---|---|
| Layer 1 | `data/processed/{X,Y,M}_{train,val,test}.npy`, `meta.json`, `models/model_{b,c,c_weighted,d,e,f}.pt`, `models/ensemble.json`, `outputs/prediction_test.npz`, `outputs/metrics/{ablation,metrics,layer1_manifest}.json` |
| Layer 2 | `outputs/layer2/{panchayat_summary,panchayat_index,block_rainfall}.csv`, `layer2_qc.json`, `data/raw/administrative/panchayat/LGD_Panchayats.parquet` (116,126 rows), `data/aux_data/admin/grid_admin_map_deccan.npz` |
| Layer 3 | `backend/advisory.py` (9 rules) executed directly with controlled inputs; `backend/main.py` `/api/advisory` (GET+POST), `/api/advisory/block`, `/api/explain*` |
| Backend | `backend/{app,main,routes_data,data_store,metrics_loader,live_infer,aux_layers}.py` via HTTP on `127.0.0.1:8000` and via direct import |
| Frontend | `Frontend/src/**` source inspection + `tsc --noEmit` + `next build`; the UI is not driven in a browser (see §17) |
| Aux layers | `data/aux_data/{soil_soilgrids_*,ndvi_monthly_*,lulc_fractions_*}` through `backend/aux_layers.py` |

### 2.2 Explicit non-goals

No application logic was edited during this validation. The validation reports on the build as it
stands at `4b2bc46`, including the two changes that commit made to served behaviour (the E+F ensemble
and the Layer-1 provenance manifest). Where that commit changed a served number, this report says so
with the before/after values rather than presenting a moving target.

### 2.3 Reproducing a claim

Every claim below names its evidence: an experiment ID (`IDENT`, `POLY`, `THR`, `PERT`, `HTTP-…`), a
console line, or a code location as `file:line`. The machine-readable dump of every experiment is in
`qa/qa_logic_*.json` (§19).

---

## 3. Existing QA Coverage vs New Logical Coverage

The earlier pass (`COMPREHENSIVE_QA_TEST_REPORT.md`, 565 lines) established that the pipeline is
*functioning*: 37 → 47 pytest cases, 48 black-box HTTP probes, 32 white-box probes, checkpoint
loading, bit-for-bit metric reproduction, frontend build/typecheck/lint, UAT/E2E flows, and the
Panchayat-mapping and duplicate-name defects (`BUG-01`, `DATA-03`). That report is not repeated here.

This pass asks a different question — not *does it run*, but *is it doing the logically correct
thing* — and it changes **four** of the earlier conclusions:

| Earlier conclusion | This pass | Evidence |
|---|---|---|
| "Panchayat mapping is a polygon aggregation (`area_weighted` for 60 % of rows)" | The **served** number is the nearest fine-grid cell at the Panchayat's representative point, regardless of `mapping_method` | **B-1**, `qa/qa_logic_core.json → polygon_vs_nearest` (e.g. pid 276458: nearest 25.34 vs area-weighted 28.77) |
| "Aux soil/NDVI/LULC are wired into Layer 3" | True, but only **clay, NDVI and LULC dominant/cropland-fraction** reach a rule; `sand`, `ocd`, `ph`, `bdod` are transported and displayed but never decide anything, and `SAND_HIGH_G_PER_KG` is a dead constant | **C-* / G-10**, `grep SAND_HIGH_G_PER_KG`, `qa/qa_logic_advisory.json → perturbations` |
| "F1≥50 mm is a model limitation to document" | It is **fixable** and was fixed: a val-selected E+F ensemble beats the baseline at every threshold (0.284 vs 0.234) | §12.4, `ablation.json`, commit `4b2bc46` |
| "The `panchayat_summary` layer is consistent" | Its rainfall columns come from a **different Layer-1 version** than the one served | **A-3**, `qa/qa_logic_core.json → layer2_summary_staleness` |

It also **confirms** three earlier conclusions as logically sound: the D-11 mask arithmetic
(44,243 target-valid fine cells per day, constant across all 122 days), the `ndvi` array layout
(months-first, `(20, 47250)`), and the fact that `panchayat_weather.csv`/`.geojson` are gitignored
*deliberately* (they exceed 100 MB), not by oversight.

---

## 4. Layer-1 → Layer-2 Logic

### 4.1 The trace that matters

```
Layer-1 output (outputs/prediction_test.npz)
   ↓  dates[] lookup        NPZ_DATES.index("2022-07-10") -> 39
   ↓  grid cell             argmin|fine_lat - lat|, argmin|fine_lon - lon|
   ↓  Layer-2 coordinate    outputs/layer2/panchayat_index.csv  panchayat_id -> (lat, lon)
   ↓  Panchayat rainfall    round(value, 2)
   ↓  backend response      data_store.Store.grid_rainfall() -> {"rainfall_mm": .., "source": "U-Net prediction grid (prediction_test.npz)"}
```

**Result: 13/13 exact.** Experiment `IDENT`/`L1L2` sampled 16 Panchayats spanning KARNATAKA,
MAHARASHTRA, TELANGANA, ANDHRA PRADESH and TAMIL NADU, and all three `mapping_method` values; 13 had
a finite cell and every one returned a value identical to `round(npz[date_idx, i, j], 2)`:

```
YALSANGI (Kalaburagi, direct_grid)         raw=11.3917 served=11.39  match=True
MUNDEWADI (Nanded, direct_grid)            raw=26.0097 served=26.01  match=True
BYADARAHALLI (Hassan, area_weighted)       raw= 2.7647 served= 2.76  match=True
DILAVARPUR (Nalgonda, area_weighted)       raw=24.1044 served=24.10  match=True
CHINARKUR (Alluri Sitharama Raju)          raw=42.8089 served=42.81  match=True
EDUGURALLAPALLI (Alluri Sitharama Raju)    raw=44.3983 served=44.40  match=True
DHODARBEDA (Narayanpur)                    raw=25.3373 served=25.34  match=True
```

Three sampled Panchayats (`nearest_fallback`) returned HTTP 422 because their own point lands on a
masked cell — that is the correct behaviour (§11) and is reported separately as **A-11**.

### 4.2 Orientation and indexing controls

A nearest-cell lookup that had latitude and longitude reversed, or that transposed the field, would
still "work" and would still return a plausible number. To make the test falsifiable, the same trace
was recomputed with the coordinate pair swapped and with the array transposed:

| Panchayat | correct | lat/lon swapped | transposed | control differs |
|---|---|---|---|---|
| YALSANGI | 11.3917 | 9.5842 | 0.0000 | yes |
| MUNDEWADI | 26.0097 | 24.4069 | 11.3917 | yes |
| WALKI (BK) | 14.0568 | 14.4102 | 3.7809 | yes |
| DHODARBEDA | 25.3373 | 23.5578 | 0.0000 | yes |

`controls_differ=True` for 11 of 13. The two exceptions (`BYADARAHALLI` 2.76 mm, `Arathi Agraharam`
0.07 mm) sit in regions where the neighbouring cells happen to be numerically identical, so the
control is uninformative there rather than failing — the served value nonetheless matched exactly.

### 4.3 Row/column reversal, rounding and units

* **Row/column reversal:** excluded by 4.2 (the transposed control differs wherever the field is not
  locally constant).
* **Rounding:** applied exactly once, at the API boundary. In 13 of 16 samples the raw float32 value
  differs from the served value (`raw != round(raw, 2)`), and the served value always equals
  `round(raw, 2)`. The NPZ itself keeps full float32 precision — so there is no "rounding before
  aggregation" defect inside Layer 1.
* **Aggregation order:** see **B-1** — there is no aggregation on the served path at all.
* **Units:** see §9.

### 4.4 Wrong NPZ / index alignment

`data_store._load_grids()` builds a list of grids in a fixed priority: `outputs/prediction_test.npz`
first, then `prediction/infer_*.npz` sorted. For each grid it keeps `dates`, `latitude`, `longitude`
and `rainfall_mm` **from the same file**, and indexes the field with `g["dates"].index(date)`. There is
no path where the lat/lon of one file is paired with the field of another. `dates` is verified to be
exactly `meta.split.test.dates` (122 entries, `2022-06-01 … 2022-09-30`).

### 4.5 Layer-1 → Layer-2 reconciliation of the served value

The served value changed when the fix was deployed, deliberately and by a known amount. Example,
`ALURU` (Udupi, Karnataka) on `2022-07-10`:

| Build | served `rainfall_mm` | source string |
|---|---|---|
| pre-hardening (single D/E checkpoint, admin-centroid coordinate) | 50.49 → 68.78 | `U-Net prediction grid (prediction_test.npz)` |
| `4b2bc46` (E+F ensemble, LGD own-point coordinate) | **95.55** | `U-Net prediction grid (prediction_test.npz)` |

Both of the changes are visible in the provenance manifest: the coordinate moved (LGD own point
instead of an admin centroid) and the model changed (`model_e.pt` ×0.5 + `model_f.pt` ×0.5). This is
the intended behaviour, and it is now recorded rather than silent (§11).

---

## 5. Layer-2 Spatial Logic

### 5.1 Nearest cell vs polygon aggregation — **an architectural mismatch (B-1)**

The application does **not** compute a Panchayat-level spatial estimate. It computes the value of the
single fine-grid cell nearest to the Panchayat's representative point. For Panchayats that cover
several cells these are different numbers. Measured independently (area-weighted overlap of 0.05°
cells with the LGD polygon, the same definition `layer2_panchayat_mapping.py` uses):

| Panchayat | cells touched | nearest cell at own point | grid-cell mean | **area-weighted mean** | served | Δ (nearest − area) |
|---|---|---|---|---|---|---|
| 276458 (direct_grid) | 128 | 25.34 | 29.34 | **28.77** | 25.34 | **−3.44 mm (−12 %)** |
| 202270 EDUGURALLAPALLI | 11 | 44.40 | 45.57 | **45.13** | 44.40 | −0.74 mm |
| 218336 | 4 | 2.76 | 4.19 | **3.74** | 2.76 | −0.98 mm |
| 207397 | 4 | 24.10 | 22.40 | **23.14** | 24.10 | +0.96 mm |
| 203962 | 3 | 1.24 | 1.20 | **1.23** | 1.24 | +0.001 mm |
| 232823 | 2 | 0.18 | 0.13 | **0.15** | 0.18 | +0.028 mm |
| Single-cell cases (181350, 205399, 202507, 227821) | 1 | x | x | x | x | **0.000** |

The served value equals the nearest-cell value in **every** row and the polygon value in none
(except the degenerate single-cell cases where the two coincide). `mapping_method` in
`panchayat_summary.csv` — `area_weighted` for **52,893** of 87,735 rows (60 %) — therefore does not
describe the served number. The served number carries only `location_precision: "exact"`, which is
true about the *coordinate* (it is the Panchayat's own representative point) and says nothing about
the *aggregation*.

*Method note (stated so the table is not over-read):* my independent overlap count anchors 0.05°
boxes on the fine-grid **cell centres** and intersects them with the polygon, whereas
`layer2_panchayat_mapping.py` assigns a cell when its centre falls inside the polygon. The two cell
counts can therefore differ — pid 276458 counts 128 here against `n_cells = 91` in the summary —
while measuring the same quantity, i.e. the area-weighted mean. The comparison that matters is that
the *served* value tracks the nearest cell and not the polygon mean; that holds for every row in the
table, including the single-cell rows where the methods must agree and do (Δ = 0.000).

Magnitude: on a 25–44 mm day the discrepancy reaches 3.4 mm (12 %). It is a systematic method
mismatch, not noise.

### 5.2 Coordinate collapse — **fixed, and verified fixed**

| Metric | Old `panchayat_index.csv` (pre-hardening) | Current |
|---|---|---|
| rows | 86,075 | 87,735 |
| distinct coordinates | 989 (1.1 %) | **87,735 (100 %)** |
| median Panchayats per coordinate | 57 | **1** |
| worst-case Panchayats per coordinate | 3,763 | **1** |
| coordinates shared by >1 Panchayat | 88.7 % of rows | **0** |

So no two Panchayats share a coordinate, and no Panchayat inherits its neighbour's rainfall through a
shared centroid. The old failure mode is gone.

### 5.3 Effective spatial resolution (**B-3**)

Distinct coordinates do **not** mean distinct rainfall. 87,735 coordinates fall on **36,922**
distinct 0.05° lattice cells; **24,752** cells host more than one Panchayat, with a maximum of **12**
Panchayats on a single cell. Every Panchayat on a cell receives an identical rainfall value because
the model's native output resolution is 0.05° (≈5.5 km).

This is scientifically honest — you cannot extract 5 km detail from a field defined at 5 km — but it
means the product's "Panchayat-level" claim is really cell-level, and the UI does not currently say
so. Classified as **B-3** (architectural) rather than a defect.

### 5.4 Boundary / masked-cell Panchayats (**A-11**)

`data_store.grid_rainfall` calls `sample_grid`, which raises `DataUnavailable` when the target cell is
NaN, and `routes_data` maps that to **HTTP 422**:

```
AMRUTHALUR (pid 199960, nearest_fallback, season mean 0.005 mm)  -> 422
   "model output is not finite at this cell (sea / excluded cell)"
```

Correct (no fabrication), but the `/api/panchayats` search result for the same Panchayat shows a
`rainfall_mm` value, so the UI can offer a Panchayat that then fails on click. 1,065 distinct lattice
cells carry a NaN field on the sample date; 1,470 Panchayats are labelled `nearest_fallback`.

### 5.5 Spatial continuity

Neighbouring fine-grid cells differ by a mean absolute `1.10 mm`, p95 `3.83 mm`, max `24.62 mm`
(62,679 neighbour pairs). Those discontinuities are a property of the model's own field — the check
recomputes them from the served grid, not from the Panchayats — so a large neighbour difference is
supported by the underlying prediction rather than being a mapping artefact. No spatial smoothing
artefact was found (**G-20**).

---

## 6. Layer-3 Agro-Advisory Logic

### 6.1 Every threshold, at its boundary (**G-6**)

All ten thresholds in `backend/advisory.py:131–147` were executed at `T−0.01 / T / T+0.01` through the
real engine (`qa/qa_logic_advisory.py → threshold_boundaries`):

| Rule | Threshold | T−ε | T | T+ε | Boundary semantics |
|---|---|---|---|---|---|
| R1_HEAVY_RAIN | 64.5 mm | R1B (medium) | **R1 (high)** | R1 (high) | `>=`, lower bound of "heavy" |
| R1B_SUBSTANTIAL_RAIN | 24.5 mm | none | **R1B (medium)** | R1B | `>=` |
| R1B upper bound | 64.5 mm | R1B (medium) | **R1 (high)** | R1 | `<` upper, exclusive — no overlap, no gap |
| R2_IRRIGATION (rain-only) | 3.0·window mm | **fires (low)** | none | none | `<` |
| R3_HEAT_STRESS (rice) | 35.0 °C | none | **fires (high)** | fires | `>=` |
| R3_HEAT_STRESS (wheat) | 34.0 °C | none | **fires (high)** | fires | crop-specific |
| R4_DISEASE humidity | 85 % | none | **fires (medium)** | fires | `>=` |
| R4_DISEASE T lower | 22.0 °C | none | **fires** | fires | `<=` inclusive |
| R4_DISEASE T upper | 32.0 °C | fires | **fires** | none | `<=` inclusive |
| R5_LODGING wind | 40 km/h | none | **fires (medium)** | fires | `>=`, tall crops only |
| R6_VEGETATION NDVI | 0.30 | **fires** | none | none | strict `<` |
| R7_SOIL_DRAINAGE clay | 350 g/kg | none | **fires** | fires | `>=`, requires rain ≥ 24.5 |
| R8_LANDCOVER cropland fraction | 0.20 | **fires** | none | none | strict `<` |

No impossible ranges, no contradictory severity, no intersection between the R1B and R1 bands. The
severity ladder `high > medium > low > none` is applied consistently by `_SEV_ORDER`, and
`SEVERITY_TO_UI` maps `none/low/medium/high → info/watch/warning/alert` without an unmapped value.

### 6.2 Is crop causally used? (**G-7**) — yes

Holding rainfall = 0, `tmax` = 36 °C, humidity = 90 %, stage = flowering and varying **only** crop:

```
rice   -> R2 + R3_HEAT_STRESS (high)
wheat  -> R2 + R3_HEAT_STRESS (high)
maize  -> R2 + R3_HEAT_STRESS (high)
pulses -> R2 + R3_HEAT_STRESS (high)
cotton -> R2 only            (high)     [heat threshold 38 °C]
bajra  -> R2 only            (high)     [heat threshold 40 °C]
```

Crop changes which rule fires, through the crop-specific heat threshold in `CROP_PARAMS`. Crop also
supplies `dry_sm`, the disease name and `tall`, so it is used by R2, R4 and R5 as well. **Not**
"echoed but not causally used".

### 6.3 Is crop stage causally used? (**G-8**) — yes, with a defect

Holding wheat at 34.5 °C and varying **only** stage:

```
sowing        -> R2 only                    (low)
vegetative    -> R2 only                    (low)
flowering     -> R2 + R3_HEAT_STRESS        (high)
grain_filling -> R2 + R3_HEAT_STRESS        (high)
maturity      -> R2 only                    (low)
None (unknown)-> R2 + R3_HEAT_STRESS        (medium)   <-- A-6
```

Stage is causally used. But the last row is a defect: `_r_heat` computes
`sensitive = stage_unknown or i.stage in p["stages"]` (`advisory.py:276`), so an **unknown** stage is
treated as heat-sensitive and can *fire* the rule, while the rule's own `condition` string asserts
`stage in ['flowering', 'grain_filling']`. A missing input increases the chance of a hazard being
reported instead of being marked `evaluable=False`. The severity is downgraded to `medium` when the
stage is unknown, so the code half-acknowledges it — but the trace's `condition` text is then
factually wrong. Reported as **A-6 (Low-Medium)**.

### 6.4 Advisory action coherence (**G-11**) — no contradictions found

Five adversarial combinations were run (`qa/qa_logic_advisory.json → coherence`):

| Case | Fired | Contradiction? |
|---|---|---|
| rainfall 200 mm | R1_HEAVY_RAIN | no |
| no rain, 40 °C, soil moisture 0.05 | R2_IRRIGATION + R3_HEAT_STRESS | no |
| 80 mm over clay-rich soil | R1_HEAVY_RAIN + R7_SOIL_DRAINAGE | no |
| 90 mm and 40 °C together | R1_HEAVY_RAIN + R3_HEAT_STRESS | no |
| no rain, built-up land cover | R2_IRRIGATION + R8_LANDCOVER | no |

In particular there is **no** input for which the engine advises irrigation *and* warns of heavy rain:
the R2 rain-only test (`rain < 3.0·window`) and the R1 test (`rain >= 64.5`) cannot both hold. Actions
are deduplicated by `actions_for()` and ordered by descending severity. R8's action ("Confirm this
location is cropland before acting on the advisory") is the correct response to a non-cropland cell.

### 6.5 The deterministic headline

`evaluate()` emits `headline_en` from the top **two** fired rules' `advice_en`, then hands it to the
LLM only for rewriting, with `is_faithful()` as a guard (§6.6). The rule decision is never delegated
to the model — `advisory.py:1–15` states this and the code matches: `severity` and `action` come from
`_SEV_ORDER[r.severity]` and `r.rule_id`, and the LLM is called only in `main.farmer_message`.

### 6.6 Faithfulness guard (**A-8**)

`is_faithful(message, trace)` accepts a rewrite only if `numbers_in(message) ⊆ allowed_numbers(trace)`,
where `allowed_numbers` serialises the whole `AdvisoryTrace`. Executed:

| Rewrite | Accepted? | Correct? |
|---|---|---|
| the trace's own headline | ✅ | yes |
| "Heavy rain is expected today." (no numbers) | ✅ | yes |
| **"Rainfall will reach 137 mm."** | ❌ rejected | yes — hallucinated number blocked |
| **"Avoid spraying for about 48 hours."** | ❌ rejected | **false rejection** — this is the project's *own* R1B action text, which lives in `RULE_ACTIONS` and is not part of `AdvisoryTrace` |

The guard fails safe (the deterministic text is served instead), so a farmer never sees the
hallucination. But a legitimate rewrite that repeats the project's own guidance is discarded, which
will make LLM rewrites look erratic. Also note `allowed_numbers` includes the digits 1–8 because rule
IDs (`R1_HEAVY_RAIN` … `R8_LANDCOVER`) are serialised into the trace, which slightly loosens the guard.

### 6.7 Layer-3 inputs are not bound to Layer-1 (**A-2**)

This is the most consequential finding in the report and it is documented in §7.1 and §18.3 with
reproduction steps.

---

## 7. Auxiliary Data Contribution

### 7.1 Dataset → route → function → rule → advisory

| Aux dataset | Loaded? | Indexed to Panchayat? | Passed to Layer 3? | Used by a rule? | Changes the advisory when varied? |
|---|---|---|---|---|---|
| Soil — **clay** (`clay_g_per_kg`) | ✅ `aux_layers.AuxLayers` | ✅ same fine cell as the rainfall | ✅ `PanchayatInput.aux.soil` | ✅ **R7_SOIL_DRAINAGE** (≥350 g/kg with rain ≥24.5 mm) | ✅ yes |
| Soil — `sand_g_per_kg` | ✅ | ✅ | ✅ (in `evidence.aux`) | ❌ **no rule reads it** | ❌ no |
| Soil — `ocd`, `ph`, `bdod` | ✅ | ✅ | ✅ (in `evidence.aux`) | ❌ no rule reads them | ❌ no |
| NDVI (`value`, `month`) | ✅ (months-first `(20, 47250)`, nearest month to the request date) | ✅ | ✅ | ✅ **R6_VEGETATION** (<0.30) | ✅ yes |
| LULC — `dominant`, `cropland_fraction` | ✅ (6-class fractions + decoded legend) | ✅ | ✅ | ✅ **R8_LANDCOVER** | ✅ yes |
| DEM | ✅ (U-Net channel `dem`, and `/api/weather` elevation) | ✅ | ✅ via the model input | ✅ indirectly (model), and the XAI panel names it | ✅ (model side) |
| Admin context (state/district/block) | ✅ | ✅ | ✅ (`location_precision`) | ✅ only for coordinate fallback | ✅ (coordinate choice) |

### 7.2 Controlled perturbation results (**G-10**)

Each row changes exactly one input and holds everything else fixed (rain 30 mm, rice, 25 °C):

```
clay 100   -> R1B                          (confidence 0.70)
clay 349.9 -> R1B                          (0.60)   <- margin penalty, no fire
clay 350.0 -> R1B + R7_SOIL_DRAINAGE       (0.60)
clay 500   -> R1B + R7_SOIL_DRAINAGE       (0.70)

NDVI 0.55  -> R2                                   (0.70)
NDVI 0.30  -> R2                                   (0.60)
NDVI 0.10  -> R2 + R6_VEGETATION                   (0.70)

cropland dominant + fraction 0.9 -> R2             (0.70)
built_up dominant   + fraction 0.9 -> R2 + R8_LANDCOVER
tree_cover dominant + fraction 0.1 -> R2 + R8_LANDCOVER

aux absent -> R2 (confidence 0.60)
aux present -> R2 (confidence 0.70)                <- +0.10 for having aux
```

So the aux layers are **causally wired**, not decorative. The change from the earlier QA pass (where
the advisory ignored aux entirely) is real and verified end-to-end: the live API returns
`evidence.aux` with the Panchayat's own values and R6/R8 fire for real Panchayats.

### 7.3 What is *not* used (**C / §28**)

* `sand_g_per_kg` is loaded, indexed, transported to Layer 3, shown in `evidence.aux` — and read by no
  rule. The threshold `SAND_HIGH_G_PER_KG = 600.0` (`advisory.py:145`) is **defined and never
  referenced**, i.e. a dead constant whose comment ("soil holds little water — needs irrigation
  sooner") promises behaviour the engine does not implement.
* `ocd`, `ph`, `bdod` are equally unused.
* The LULC fractions beyond `cropland_fraction` (6 classes loaded) are unused.
* Consequence: the README/UI must not claim soil *properties* (plural) inform the advisory. Only soil
  **texture (clay)** does.

---

## 8. Temporal Logic

### 8.1 Date → index → value (**G-2**)

54 traces across 3 Panchayats and 18 dates (first date, second date, mid-season, month boundaries
`2022-06-30 / 07-01 / 08-31 / 09-01`, and 9 interior dates) all reproduced exactly:

```
34/34 ... 54/54 date->index->value traces exact
```

Including the year-boundary and month-boundary cases, so there is no off-by-one in
`g["dates"].index(date)` and no month/day transposition.

### 8.2 Out-of-range and malformed dates (**G-3**)

```
grid_rainfall(..., "2021-07-10") -> None      (not in the NPZ)   -> /auth/ HTTP 422
grid_rainfall(..., "2022-10-01") -> None                          -> /auth/ HTTP 422
grid_rainfall(..., "2022-05-31") -> None                          -> /auth/ HTTP 422
grid_rainfall(..., "2022-13-01") -> None
grid_rainfall(..., "not-a-date") -> None
grid_rainfall(..., "")           -> None
```

Dates outside the test split are not silently served from another year, and a non-date string cannot
reach the index. **However** `GET /api/advisory` accepts and echoes any date (§7.1 / A-2), so the
strictness exists on `/auth/` and not on the advisory route.

### 8.3 Train/val/test leakage

`meta.split` is `{train: [2018,2019,2020], val: [2021], test: [2022]}` — strict year separation,
122/122/122 days. Normalisation statistics are train-only. The served grid is the 2022 test split
only. No leakage path was found: the model-selection rule reads `results["val"]` exclusively
(`ablation.py`, `val = results["val"]`), and the ensemble mixing weight is searched on val only
(`ENSEMBLE_WEIGHTS` loop, printing `6/7 weights inside the MAE window`).

### 8.4 Monsoon structure (**G-19**)

Domain-mean rainfall per month from the served grid, against the CHIRPS reference and the IMD bilinear
input:

| Month | days | model mean (mm) | CHIRPS mean (mm) | IMD bilinear (mm) |
|---|---|---|---|---|
| 2022-06 | 30 | 3.46 | 5.15 | 3.43 |
| 2022-07 | 31 | **14.88** | **13.90** | 11.40 |
| 2022-08 | 31 | 10.80 | 11.11 | 8.51 |
| 2022-09 | 30 | 8.30 | 7.24 | 5.53 |

Peak in July, decay through September — physically consistent, and the model tracks the reference
month-to-month. Daily domain-mean correlations: model↔CHIRPS **0.836**, model↔IMD **0.898**,
CHIRPS↔IMD 0.770. Zero identical consecutive daily means and zero identical consecutive full days, so
there is no stale-cache or repeated-frame artefact in the grid. The model's daily domain-mean range
(0.29–25.14) is narrower than the reference's (0.09–34.65) — the smoothing signature quantified in
§12.4.

---

## 9. Numerical / Unit Logic

### 9.1 Scaling chain

```
raw model residual  →  + bilinear IMD baseline (normalized)  →  × rain_scale (100.0)
    →  clip(·, 0, None)  →  where(M == 1, ·, NaN)  →  float32 NPZ  →  round(·, 2)  →  API  →  UI
```

`rain_scale_mm = 100.0` comes from `meta.json` (train-derived) and is read from the checkpoint in
`load_members`, so a mismatch between the checkpoint's scale and the data's scale would raise rather
than silently rescale — `_combine()` asserts the members agree on `rain_scale`.

### 9.2 Applied once, not twice (**G-4**)

Double scaling would inflate values by ~100×. Independently recomputed:

| Quantity | Value |
|---|---|
| model daily maximum (2022-07-10) | 115.69 mm |
| CHIRPS reference daily maximum | 257.25 mm |
| IMD bilinear daily maximum | 249.99 mm |
| ratio model/IMD at the day's maximum | 0.463 |
| **whole-split mean, model** | **9.4134 mm** |
| **whole-split mean, CHIRPS** | **9.4015 mm** |
| model minimum | 0.0 (never negative) |
| any `inf`? | no |

The whole-split means agree to 0.01 mm, which is the signature of a correctly scaled field (the model
is mean-unbiased in aggregate by construction of the masked MAE loss). A 100× error is excluded.

### 9.3 Values around the critical magnitudes

* **0 mm:** exactly representable and reachable — `NARASAPURAM`, `BHEEMUNIPALLI`, `ALUR` all return
  `0.0 mm` from `/auth/` for `2022-07-10`. This matters far beyond arithmetic: it is the trigger for
  **A-1**.
* **0.0049 / 0.005:** the NPZ stores float32; the API rounds to 2 dp, so `0.0049 → 0.0` and
  `0.005 → 0.01` (banker's-free, `round()` semantics). Rounding happens **after** masking and never
  before aggregation (there is no aggregation on this path). Because the threshold table (§6.1) is
  evaluated on the *rounded* API value when driven from the UI, a value of `0.0049` reaches R2 as
  `0.0`.
* **1 / 10 / 25 / 50 / 100:** all sampled through the API or the NPZ; the `_risk_level` bands
  (<2.5 / <10 / <25 / <50 / else) were exercised at 2.49/2.5, 9.99/10, 24.99/25, 49.99/50 with no
  disagreement between frontend `classifyRisk()` and backend `_risk_level()` (**G-16**).
* **Extreme:** the NPZ maximum is `193.96 mm` (up from `175.51` before the ensemble and `163.28`
  before the earlier hardening). Nothing exceeds the physical range and nothing is negative.

### 9.4 Precision loss

The NPZ is float32; the pipeline reads float32 and rounds to 2 dp only at the boundary, so no
meaningful precision is lost. The one place precision *is* lost by design is the `rain_scale`
normalisation of the target (`Y / 100`), which is the training contract, not a serving decision.

---

## 10. Perturbation / Causal Tests

Every experiment below changes **exactly one** input and holds the rest constant, then classifies the
result (`qa/qa_logic_advisory.json → perturbations`).

| # | Varied input | Constant | Outcome | Classification |
|---|---|---|---|---|
| 1 | `crop` ∈ {rice, wheat, maize, cotton, bajra, pulses} | rain 0, tmax 36, RH 90, flowering | R3 fires for rice/wheat/maize/pulses only | **expected change** |
| 2 | `stage` ∈ {sowing, vegetative, flowering, grain_filling, maturity, None} | rain 0, tmax 34.5, RH 90, wheat | R3 fires for flowering/grain_filling; also for `None` | expected, plus **A-6** |
| 3 | `tmax_c` ∈ {20, 30, 34.9, 35.0, 40} | rain 0, rice, flowering | R3 fires only ≥35; confidence drops at 34.9 | **expected change** |
| 4 | `rainfall_mm` ∈ {0, 2.9, 3.0, 24.4, 24.5, 64.4, 64.5, 120} | rice, vegetative, 25 °C | R2 → none → R1B → R1 at exactly 3.0 / 24.5 / 64.5 | **expected change** |
| 5 | `aux.soil.clay` ∈ {100, 349.9, 350.0, 500} | rain 30, rice, 25 °C | R7 fires at exactly 350 | **expected change** |
| 6 | `aux.ndvi.value` ∈ {0.55, 0.30, 0.10} | rain 0, rice, 25 °C | R6 fires only below 0.30 | **expected change** |
| 7 | `aux.lulc` ∈ {cropland/0.9, built_up/0.9, tree_cover/0.1} | rain 0, rice, 25 °C | R8 fires for the last two | **expected change** |
| 8 | `aux` present vs absent | rain 0, rice, 25 °C | same rules; confidence 0.70 vs 0.60 | **expected change** (non-decision) |
| 9 | `soil_moisture` ∈ {None, 0.35, 0.29, 0.05} | rain 1, rice, 25 °C | R2 requires rain low **and** soil dry (<0.30 for rice) | **expected change** |
| 10 | Panchayat (via `/auth/` name+context) | date fixed | served value always the requested Panchayat's own cell | **expected change**, 12/12 |
| 11 | date (54 traces) | Panchayat fixed | value follows the NPZ index exactly | **expected change**, 54/54 |

**Zero unexpected changes, zero missing changes, zero unrelated changes** inside the advisory engine.
The two "unexpected" behaviours found elsewhere are outside the engine: the client-side fallback rule
set (**A-5**) and the caller-supplied rainfall (**A-2**).

### 10.1 Perturbation through the HTTP surface (end-to-end)

The same discipline applied to the live API — same `panchayat_id`, same `date`, only `rainfall_mm`
changed:

| `rainfall_mm` | severity | `risk_level` | fired rules | actions |
|---|---|---|---|---|
| 0.0 (the true value for this Panchayat on 2022-07-10) | warning | no_rain | R2_IRRIGATION, R6_VEGETATION | 4 |
| 999.0 (a lie) | **alert** | very_heavy | R1_HEAVY_RAIN, R6_VEGETATION | 5 |

The engine responds correctly to the input it is given. It simply has no way to know the input is
false, which is **A-2**.

---

## 11. Data Provenance

### 11.1 Provenance chain, as served

```
UI  Frontend/src/views/HomePage.tsx       selected Panchayat (from /api/panchayats, coords from /api/geocode)
 ↓  POST /auth/                            {state, district, block_name, panchayat_name, date}
     data_store.Store.matches()            -> 409 if >1 row and no lat/lon; else the row
     panchayat_index.csv[panchayat_id]     -> (lat, lon)   [LGD representative point, EPSG:32643 round-trip]
 ↓  data_store.grid_rainfall(lat, lon, date)
     live_infer.sample_grid()              -> nearest cell (i, j); NaN -> DataUnavailable -> 422
 ↓  outputs/prediction_test.npz            rainfall_mm[date_idx, i, j] * 1.0 mm
     (members: model_e.pt ×0.5 + model_f.pt ×0.5, sha256 recorded in layer1_manifest.json)
 ↓  round(·, 2)                            -> {"rainfall_mm": .., "source": "U-Net prediction grid (prediction_test.npz)"}
 ↓  UI                                     number, risk chip, map marker
 ↓  GET /api/advisory?rainfall_mm=<that number>  -> rules -> severity/actions/headline
```

### 11.2 Provenance is now machine-checkable (**G-22**)

`generate_pred.py` writes `outputs/metrics/layer1_manifest.json`:

```json
{ "geometry": {"n_dates": 122, "first_date": "2022-06-01", "last_date": "2022-09-30",
               "shape": [122, 285, 200], "dtype": "float32"},
  "values":   {"finite_cells_per_day": 44243, "max_mm": 193.96, "mean_mm": 9.413,
               "any_negative": false, "nan_masked": true},
  "predictor": {"members": ["model_e.pt", "model_f.pt"], "weights": [0.5, 0.5],
                "rain_scale": 100.0, "n_parameters": 117329,
                "manifest": "models/ensemble.json",
                "checkpoint_sha256": {"model_e.pt": "75187a83…", "model_f.pt": "569b3adc…"},
                "clipping": "per-member clip(net_k(X)*rain_scale, 0) then weighted mean"},
  "artefact": {"path": "outputs/prediction_test.npz", "sha256": "09410cf6…"} }
```

and `/api/metrics` republishes it as `layer1_provenance`. A new test suite
(`backend/tests/test_artifact_consistency.py`, 6 cases) verifies the grid hash, the grid's values, the
checkpoint hashes, the agreement between `models/ensemble.json` and the provenance block, and that the
API reports the same hashes. **This closes the "stale grid" class of defect**: a retrain that forgets
`generate_pred.py` now fails a test instead of silently serving the old field.

### 11.3 10 diverse cases traced

| # | Panchayat / date | Layer-1 source | Value | Verified independently |
|---|---|---|---|---|
| 1 | YALSANGI (Kalaburagi) 2022-07-10 | `prediction_test.npz` | 11.39 | ✅ recomputed |
| 2 | MUNDEWADI (Nanded) 2022-07-10 | `prediction_test.npz` | 26.01 | ✅ |
| 3 | CHINARKUR (Alluri Sitharama Raju) 2022-07-10 | `prediction_test.npz` | 42.81 | ✅ |
| 4 | EDUGURALLAPALLI (Alluri Sitharama Raju) 2022-07-10 | `prediction_test.npz` | 44.40 | ✅ |
| 5 | DHODARBEDA (Narayanpur) 2022-07-10 | `prediction_test.npz` | 25.34 | ✅ |
| 6 | PEGA (Alluri Sitharama Raju) 2022-06-01 | `prediction_test.npz` | 0.07 | ✅ |
| 7 | KACHAVARAM 2022-06-22 | `prediction_test.npz` | 6.81 | ✅ |
| 8 | NARASAPURAM 2022-07-10 | `prediction_test.npz` | 0.00 | ✅ (and triggers A-1) |
| 9 | ALURU (Udupi) 2022-07-10 | `prediction_test.npz` (ensemble) | 95.55 | ✅ |
| 10 | RAMPUR (61 rows share the name) | `prediction_test.npz` | 16.77 | ✅ after disambiguation |

### 11.4 Provenance gaps that remain

* `panchayat_summary.csv`'s rainfall columns carry **no** Layer-1 version stamp (**A-3**).
* `outputs/layer2/block_rainfall.csv` (137,006 rows) is stale relative to the current grid and is read
  by no route (**D-7**). It was not regenerated because the producer script loops
  122 dates × 1,123 subdistricts inside Python (~137 k iterations over a 47 k-cell mask) and rewriting
  it was not a good use of the remaining budget; it is an unused artefact either way.
* `outputs/metrics/metrics.json` still carries the previous `best_model` result rows (its `meta`
  block, which is what `/api/metrics` actually reads, is current). Regenerating it needs a full
  `scripts/evaluate.py` run (figures + `prediction/*.npz`) — listed in §17.

---

## 12. Scientific Sanity Tests

### 12.1 Model output vs target vs baseline (**C-1**, **C-2**)

Pooled over all 122 test days × 44,243 valid cells:

| threshold | observed cell-days | predicted cell-days | precision | recall | F1 |
|---|---|---|---|---|---|
| ≥10 mm | 1,618,246 | 1,711,377 | 0.584 | 0.618 | **0.601** |
| ≥25 mm | 624,843 | 615,381 | 0.445 | 0.439 | **0.442** |
| ≥50 mm | 167,065 | 124,908 | 0.332 | 0.248 | **0.284** |
| ≥100 mm | 24,982 | 523 | 0.239 | **0.005** | 0.010 |

Deployed model vs bilinear baseline on the same test split:

| metric | baseline (A) | deployed (EF) | winner |
|---|---|---|---|
| MAE (mm) | 9.603 | **8.428** | EF (−12.2 %) |
| RMSE (mm) | 18.183 | **14.998** | EF |
| correlation | 0.385 | **0.527** | EF |
| F1 ≥10 mm | 0.477 | **0.601** | EF |
| F1 ≥25 mm | 0.343 | **0.442** | EF |
| F1 ≥50 mm | 0.234 | **0.284** | EF |

Magnitude bias on heavy-rain cells (test split):

```
observed ≥50 mm cells:  n = 167,065
   observed mean       = 75.96 mm
   predicted mean      = 35.62 mm      (53 % low)
   predicted median    = 35.28 mm
   fraction predicted ≥50 mm = 24.8 %
p99  observed = 76.66 mm   |   p99 model = 58.40 mm
maximum achieved / maximum observed = 0.466
```

**Interpretation.** The model is a good mean estimator (mean bias ≈ 0, MAE 12 % below the baseline) and
a weak extreme estimator: it reproduces the *mean* of a heavy-rain population at roughly half its
magnitude, and it essentially cannot reach 100 mm/day. This is the classic regression-to-the-mean
signature of an MAE-trained field. It is a **scientific limitation (C)**, not an implementation bug —
the values are physical (0 ≤ x ≤ 194 mm, no negatives, no infinities), the shape is right, and the
ranking is right (precision 0.33 at ≥50 mm means the model's high values are mostly genuine events).

### 12.2 Why the heavy-rain threshold was failing, and what fixed it

Diagnosis (`qa/qa_f1_50.json`) on the previous deployed checkpoint (E):

```
threshold 50 mm:  precision 0.438   recall 0.076   F1 0.129
where the truth is >= 50 mm, E predicts a mean of 27.0 mm (median 26.1)
```

Precision-heavy and recall-starved: the model *knows* which cells are wet (its high values are usually
genuine) but compresses their magnitude below the event floor. Two candidate fixes were tested:

1. **Calibrated decision threshold** (choose the ≥50 mm decision threshold on val, apply to test):
   E reaches test F1 0.293 at a decision threshold of 35 mm. This is a legitimate technique, but it
   only restores the ranking skill the model already had; it does not change the served field.
2. **Model-side re-weighting** (train `model_f` with `--loss weighted --weight-rain-mm 50
   --weight-mult 10`): F reaches **raw** test F1≥50 mm **0.320**, at a cost of +0.82 mm MAE and
   −0.018 F1≥10 mm versus E.

Averaging E and F (equal weights, chosen **on validation** by the project's own rule) dominates both
in practice: **MAE 8.428, corr 0.527, F1 0.601/0.442/0.284** — i.e. E's mean error and corr, with
heavy-rain skill above the bilinear baseline at every threshold. This is now the deployed model
(`models/ensemble.json`, commit `4b2bc46`). The full trade-off table is in
`outputs/metrics/ablation_summary.md`; nothing is hidden by quoting a single number, and the README
states the improvement as **"12.2 % lower MAE"**, never as "12 % better".

**An honest note on method:** the first calibration sweep written for this validation conflated the
*observation* threshold with the *decision* threshold, which would have "improved" F1≥50 mm simply by
redefining the event. That error is recorded as **F-4** rather than quietly dropped, and the corrected
test is what made the ensemble the right answer instead of threshold cosmetics.

### 12.3 Spatial sanity

See §5.5. Additionally: no evidence of sea/land contamination — the mask is applied as
`np.where(M == 1, pred, NaN)` and the finite-cell count is 44,243 every day, identical to
`meta.grid.n_target_valid_fine_cells`. No Panchayat receives a value from a masked cell; those return
422.

### 12.4 Temporal sanity

See §8.4. Month-to-month structure is physical, daily correlation with the reference is 0.836, and
there are no repeated or stale days.

---

## 13. UI / API / Data Consistency

### 13.1 Panchayat identity across endpoints (**G-12**)

| Check | Result |
|---|---|
| 12 unique-named Panchayats, name+context via `/auth/` | 12/12 exact match to the intended Panchayat's own cell |
| `RAMPUR` (61 rows share the name), name only | **HTTP 409** with 10 candidates |
| same, name + district + block | HTTP 200, value matches the intended row |
| `/api/geocode` | GET-style route; my first probe used POST and got 405 (**F-3**, my test's error, not the app's) |
| `location_precision` in all 12 responses | `"exact"` |

### 13.2 Risk classification agreement (**G-16**)

`Frontend/src/types/index.ts::classifyRisk` and `backend/main.py::_risk_level` implement identical
bands (`<2.5 no_rain`, `<10 light`, `<25 moderate`, `<50 heavy`, else `very_heavy`). The UI's *severity*
chip, however, comes from a different scale (`SEVERITY_TO_UI`: info/watch/warning/alert) driven by
fired rules. The two can disagree legitimately and observably: `NARASAPURAM` on `2022-07-10` returns
`risk_level: "no_rain"` together with `severity: "warning"` (R6_VEGETATION fired on low NDVI + low
rain). Neither is wrong, but they answer different questions and the UI must not present them as one
number (**B-6**).

### 13.3 Displayed vs backend rainfall

Where the frontend displays `selected.rainfall_mm`, that value comes from `/auth/`
(`response.prediction.rainfall_mm`), which §4.1 shows is `round(NPZ cell, 2)`. Consistent — **except**
for the falsy-zero substitution (**A-1**) and except for the search-result panel, which displays a
**season mean** labelled with a single date (**A-4**):

```
PanchayatSearch.tsx:112   {p.rainfall_mm.toFixed(2)} mm — {p.date}
  p.rainfall_mm = panchayat_summary.rainfall_mean_mm  (season mean over 122 days)
  p.date        = "2022-07-10"                        (data_store.DEFAULT_DATE, not a real date for this value)
```

and which may in addition be from the previous Layer-1 version (**A-3**).

### 13.4 Rounding differences

The frontend renders `toFixed(2)` / `toFixed(1)` on values already rounded to 2 dp by the API, so
display rounding cannot diverge from the decision rounding except by display truncation of the
third decimal, which the API has already removed.

### 13.5 Client-side fallback divergence (**A-5**)

`AdvisoryPanel.generateLocalAdvisory()` is a second, independent rule engine used whenever
`/api/advisory` fails. It differs from the backend in ways that matter:

| | backend `advisory.py` | frontend fallback |
|---|---|---|
| tiers | 3.0·window (R2), 24.5 (R1B), 64.5 (R1) | 6.5, 24.5, 64.5 |
| crop | changes heat threshold, dry-soil level, disease, lodging | **ignored** |
| stage | changes heat-sensitive window | **ignored** |
| temperature / humidity | R3, R4 | **ignored** |
| aux (soil/NDVI/LULC) | R6, R7, R8 | **ignored** |
| action text | e.g. "Avoid spraying or fertiliser for about 48 hours" | e.g. "Avoid pesticide/fertilizer application for 48h" |

So for the *same* Panchayat, date and rainfall, the app gives a **different advisory depending on
whether the backend is reachable** — a crop-agnostic one with an extra tier. That is a real
consistency defect, and it directly contradicts the "advisory is rule-based on crop and stage" claim
when the fallback is active.

### 13.6 Cache / state contamination (**G-14**)

* 4 sequenced requests, each with a different Panchayat **and** a different date: 4/4 matched their
  own recomputation.
* The same 4 requests fired concurrently through 8 threads: 4/4 matched.
* The store is read-only after construction (`pandas` frames + `npz` arrays); the only mutable state is
  `State.store` (a lazy singleton) and `Store._admin` / `_weather_day` caches, all keyed by content.
  No request can mutate another's data.

---

## 14. Newly Discovered Issues (buckets A–G)

### 14.1 A. Confirmed logical defects

| ID | Sev | Defect | Evidence |
|---|---|---|---|
| **A-1** | **High** | `HomePage.tsx` uses `response.prediction.rainfall_mm \|\| selected.rainfall_mm`. A legitimate `0.0 mm` answer is falsy and is replaced by the **previous selection's** rainfall, which then feeds the display, the risk chip, the map and the advisory. Reachable in production data. | `NARASAPURAM` / `BHEEMUNIPALLI` / `ALUR` return exactly `0.0 mm` on 2022-07-10 (`qa/qa_logic_http.json → zero_rain_case`); `HomePage.tsx:73` (contrast `??` on the three lines below it) |
| **A-2** | **High** | `GET /api/advisory` takes `rainfall_mm` from the caller and never validates it against the Panchayat/date; `panchayat_id` is used only for aux lookup; `date` is echoed into `data_date` unvalidated; a bogus id returns 200 with `aux: null`. | rainfall 0 → warning vs rainfall 999 → alert on the same id/date; `date=1999-01-01` → 200; `panchayat_id=999999999` → 200 (`qa/qa_logic_http.json → layer3_binding`); `main.py:501–546` |
| **A-3** | **Med** | `panchayat_summary.csv` rainfall columns come from a **different Layer-1 grid version** than the one served, with no version stamp. Served by `/api/panchayats`. | summary max `163.2806` (= the pre-hardening grid max; current `193.96`); pid 202270 recorded `4.89` vs recomputed `14.94`; pid 276458 `5.03` vs `18.17`; aggregate summary mean `4.008` vs field mean `9.413` (`qa/qa_logic_core.json → layer2_summary_staleness`) |
| **A-4** | **Med** | The search-result list shows a **season mean** with a single date attached (`{p.rainfall_mm} mm — {p.date}` where `date` is `DEFAULT_DATE`). | `PanchayatSearch.tsx:112`, `data_store.py:_row_to_panchayat` (`rainfall_basis: "season_mean_2022"`) |
| **A-5** | **Med** | The client-side fallback advisory is a different rule set (different tiers, ignores crop/stage/temp/aux), so the same inputs yield different advice depending on backend reachability. | `AdvisoryPanel.tsx:14–60` vs `advisory.py` (§13.5) |
| **A-6** | **Low-Med** | An **unknown crop stage** makes R3_HEAT_STRESS fire (treated as heat-sensitive, severity downgraded) while the trace's `condition` claims the stage must be in the sensitive set. Missing input increases hazard instead of marking the rule unevaluable. | `advisory.py:275–280`; `THR`/`PERT` row 2: stage `None` at 34.5 °C → R3 fires, severity medium |
| **A-7** | **Low** | Hardcoded claim fallback in the UI: `mae_improvement_pct: metrics?.mae_improvement_pct ?? 15.9`. A stale pilot-era number appears whenever metrics are missing/null. | `XAIPanel.tsx:56` |
| **A-8** | **Low** | The faithfulness guard rejects a legitimate rewrite that quotes the project's own action text (it contains `48`, which is not in `AdvisoryTrace`). Fails safe, but makes LLM rewrites look erratic. | `is_faithful("Avoid spraying for about 48 hours.") → False`; `RULE_ACTIONS` not part of the trace |
| **A-9** | **Low** | GET/POST advisory vocabularies still differ: POST accepts `grain_filling` / `maturity`; GET's `StageQ` rejects `grain_filling` (422). | `main.py:CropQ/StageQ` vs `advisory.Stage`; `HTTP enums` probe |
| **A-10** | **Low** | `sand_g_per_kg` is loaded, indexed, transported into `evidence.aux` — and read by no rule; `SAND_HIGH_G_PER_KG` is a dead constant. Same for `ocd`, `ph`, `bdod` and 5 of 6 LULC fractions. | `advisory.py:145` defined and never referenced; `PERT` rows 5–7 |
| **A-11** | **Low** | 1,470 `nearest_fallback` Panchayats can resolve to a coordinate whose cell is masked, so `/api/panchayats` offers a Panchayat whose `/auth/` then returns 422 (correct refusal, dead-end UX). | `AMRUTHALUR` pid 199960 → 422; 1,065 lattice cells carry a NaN field on the sample date |
| **A-12** | **Low** | `outputs/layer2/block_rainfall.csv` (137,006 rows) is stale relative to the served grid and is read by no route. | no route references it; grid regenerated after it was written |

### 14.2 B. Architectural mismatches

| ID | Mismatch | Evidence |
|---|---|---|
| **B-1** | The product presents a **nearest-cell value at one representative point** as a Panchayat value, while 60 % of rows are labelled `area_weighted` and the XAI text says the value "is taken from N grid cells". Measured gaps up to 3.44 mm (12 %). | §5.1; `POLY` experiment |
| **B-2** | `mapping_method` describes the *polygon-to-grid* classification, not the value that is served, so the label is not attached to the number the user sees. | same |
| **B-3** | Effective resolution is 0.05° (≈5.5 km): 24,752 lattice cells host >1 Panchayat (max 12), so neighbouring Panchayats receive identical values. "Panchayat-level" overstates the spatial claim. | `SPATIAL` |
| **B-4** | The deployed Layer-1 field is a **two-checkpoint ensemble** while the repo's convention was a single `best_model.pt`. Now explicit: `models/ensemble.json` is the declaration, `best_model.pt` is the documented fallback, and both serving paths resolve through `scripts/ensemble.py`. | `models/ensemble.json`, `scripts/ensemble.py`, commit `4b2bc46` |
| **B-5** | Aux soil/NDVI/LULC are Layer-3 context only; they are not U-Net channels (deliberate, and documented). | `config.py:CHANNELS_ALL` |
| **B-6** | Two independent classifications (`risk_level` from rainfall bands; `severity` from fired rules) are returned side by side and can disagree (`no_rain` + `warning`). | §13.2 |

### 14.3 C. Scientific / model limitations

| ID | Limitation | Evidence |
|---|---|---|
| **C-1** | The model cannot represent extreme rainfall: ≥100 mm/day recall **0.005** (523 predicted vs 24,982 observed). | §12.1 |
| **C-2** | Systematic magnitude compression on heavy-rain cells: observed ≥50 mm cells are predicted at a **mean of 35.6 mm** vs an observed 76.0 mm (53 % low); only 24.8 % are predicted ≥50 mm. | §12.1 |
| **C-3** | All metrics are scored against **CHIRPS**, itself a reference product, not ground truth. | `metrics.json.meta.reference` |
| **C-4** | Advisory thresholds are prototype defaults, not agronomically validated (the module says so). | `advisory.py:16–20` |
| **C-5** | The crop calendar and rules are season-agnostic while the dataset is monsoon-only (Jun–Sep, 2018–2022), so nothing outside the monsoon can be validated. | `config.py:MONSOON_START/END` |
| **C-6** | The ensemble raises the peak field (max 175.5 → 194.0 mm) and lifts `F1≥50 mm` above the baseline, but does not remove C-1/C-2. | §12.2 |
| **C-7** | The 5-channel architecture excludes `era5_wind`, which exists in the raw caches and is the natural predictor for R5_LODGING; the wind rule therefore runs on weather fetched at advisory time, not on a model input. | `config.py` note; `advisory.py:_r_wind` |

### 14.4 D. Documentation / communication risks

| ID | Risk | Evidence |
|---|---|---|
| **D-1** | The advisory text uses future tense for a historical refinement ("Heavy rain of about 80 mm **is expected**", "is **forecast**"), and the API field is documented as "Forecast total", while the UI FAQ correctly says "Not a forecast … Everything served here is historical refinement". | `advisory.py:123, 206, 208, 224, 244, 434`; `LandingPage.tsx:124–138` |
| **D-2** | `panchayat_summary.csv` presents `rainfall_mean_mm/min/max/n_wet_days` with no version stamp; a reader cannot tell which Layer-1 run produced them. | **A-3** |
| **D-3** | The README stated the selection rule as "0.1 mm tie broken by val correlation" while the code used "0.75 mm tie broken by val F1≥25 mm". **Fixed in this pass.** | README §3 bullet |
| **D-4** | A hardcoded `15.9` MAE-improvement figure can reach the UI. | **A-7** |
| **D-5** | `mapping_method` is exposed in the UI/API with a meaning stronger than what it describes. | **B-2** |
| **D-6** | Wording discipline around the improvement claim: the served figure is now **12.2 % lower MAE** (EF) and is labelled as MAE, never as "better overall". The pilot section still quotes the pilot's 15.9 %, which is correct for the pilot and clearly labelled PILOT. | `README.md:157–165`, `/api/metrics` |
| **D-7** | `block_rainfall.csv` exists in the repo, is unused, and is stale — a reader will assume it is live. | **A-12** |

### 14.5 E. Data-quality limitations

| ID | Limitation | Evidence |
|---|---|---|
| **E-1** | 8,645 Panchayat names are duplicated; up to **61** rows share one name. Mitigated by 409 + candidates, but name-only UX remains awkward. | `IDENTITY`; `RAMPUR` |
| **E-2** | The LGD parquet's `gpcode` is a **non-unique key**: 11,469 duplicated codes in 99,204 numeric-code rows (116,126 total). Joining summary → LGD inflates 87,735 rows to 99,204 and one code (`202270`) yields 6 geometry rows. | `ADMIN` |
| **E-3** | 76 district-name and 3 state-name mismatches survive normalisation (0.08 % of joined rows). | `ADMIN` |
| **E-4** | 1,470 Panchayats are `nearest_fallback`; 1,065 lattice cells carry a NaN field on the sample date. | `spatial`, `A-11` |
| **E-5** | `panchayat_weather.csv` / `.geojson` are absent (gitignored, >100 MB by design), so the per-date area-weighted Layer-2 path (`data_store.layer2_value`) is never exercised in the shipped build — it always returns `None`. | `data_store.py:176–178` |
| **E-6** | NDVI composites are 8.045 % NaN and are monthly, so the nearest-month lookup can attribute a July value to a date in a different phenological week. | `meta`/`aux_layers` |
| **E-7** | The reference `layer2_qc.json` records `mapped=86103` of 86,103 — the remaining 1,632 rows are `unmapped` and hidden from search. | `layer2_qc.json` |

### 14.6 F. Test-plan errors (expectations that were themselves wrong)

| ID | The earlier expectation | Why it was wrong |
|---|---|---|
| **F-1** | That ~3,007 fine cells are excluded by masking. | The grid has 57,000 cells, not 47,250; the true excluded count is 12,757 and 44,243 are target-valid. Corrected in the earlier pass, re-confirmed here (constant across all 122 days). |
| **F-2** | That Layer-3 would consume aux soil/NDVI/LULC. | It did not, before the hardening pass. Now verified to consume clay, NDVI and LULC. |
| **F-3** | That `/api/geocode` accepts POST. | It is a GET-style route; my first probe returned 405. A test-plan error, not an app defect. |
| **F-4** | That `F1≥50 mm` could be "fixed" by calibrating the reporting threshold. | My first sweep conflated the observation threshold with the decision threshold, which would have moved the goalposts rather than improving the model. Corrected: a val-calibrated ≥50 mm decision threshold lifts the old model only to 0.293, below the re-weighted model's raw 0.320 — so calibration alone was not the answer, and the ensemble is. |
| **F-5** | That 300-epoch CPU training runs are affordable inside a single tool invocation. | They time out; the validated configuration uses `--val-subset 24` and a bounded epoch count (60/130 epochs, 183 s / 481 s). |

### 14.7 G. Passed logical checks

| ID | Check | Result |
|---|---|---|
| **G-1** | Served rainfall == independently recomputed `round(cell, 2)` | **13/13** |
| **G-2** | date → index → value traces (incl. first/last/month boundaries) | **54/54** |
| **G-3** | Out-of-range / malformed dates refused by `/auth/` | 422 for all six probes |
| **G-4** | No double scaling; no negative; no Inf; means agree to 0.01 mm | pass |
| **G-5** | Rounding applied once, at the API boundary; NPZ keeps float32 | pass |
| **G-6** | All ten advisory thresholds behave exactly as documented at T±ε | pass, no gaps/overlaps |
| **G-7** | Crop changes the fired rules (heat threshold is crop-specific) | pass |
| **G-8** | Stage changes the fired rules | pass (with **A-6**) |
| **G-9** | Temperature is causally used; confidence drops near thresholds | pass |
| **G-10** | Soil clay / NDVI / LULC each change the rules in isolation; missing aux lowers confidence without fabricating values | pass |
| **G-11** | No contradictory action combinations (never irrigation + flood warning) | pass |
| **G-12** | Ambiguous name → 409 + candidates; contextual → correct Panchayat | pass, 12/12 |
| **G-13** | No coordinate collapse: 87,735 rows / 87,735 distinct coordinates / max 1 per coordinate | pass |
| **G-14** | No cache or state contamination, sequential or concurrent | 8/8 |
| **G-15** | Advisory enums: `general`, `ripening`, `mustard` accepted on GET and POST | pass |
| **G-16** | Frontend `classifyRisk` bands == backend `_risk_level` bands | pass |
| **G-17** | XAI names only trained channels, declares its weights heuristic (not SHAP), and accurately describes taking the value from N grid cells | pass |
| **G-18** | The UI explicitly and correctly states the product is historical refinement, not a forecast | pass |
| **G-19** | Monsoon temporal structure physical; daily corr vs reference 0.836; no repeated/stale days | pass |
| **G-20** | Spatial discontinuities are properties of the model's own field | pass |
| **G-21** | Mask semantics: 44,243 finite cells/day, constant | pass |
| **G-22** | Served grid is cryptographically bound to the models that produced it | pass (new) |
| **G-23** | Test suites | `pytest` **53 passed**; frontend `tsc --noEmit` + `next build` clean |

---

## 15. Severity Matrix

| Severity | Count | IDs |
|---|---|---|
| **High** | 2 | A-1, A-2 |
| **Medium** | 3 | A-3, A-4, A-5 |
| **Low-Medium** | 1 | A-6 |
| **Low** | 6 | A-7 … A-12 |
| Architectural (not defects) | 6 | B-1 … B-6 |
| Scientific limitations (not defects) | 7 | C-1 … C-7 |
| Documentation risks | 7 | D-1 … D-7 |
| Data-quality limitations | 7 | E-1 … E-7 |
| Test-plan corrections | 5 | F-1 … F-5 |
| Passed checks | 23 | G-1 … G-23 |

Defects by blast radius:

* **User-visible wrong number, reachable today:** A-1 (only via the frontend, and only for exactly
  0.0 mm), A-2 (only if the client supplies a wrong value).
* **User-visible misleading number:** A-3, A-4 (search list), D-2.
* **Wrong advice without an error:** A-2, A-5.
* **No user impact, but corrupts the record:** A-12, D-7, stale `metrics.json` rows.
* **Nothing at all:** A-10 (dead constant / unused fields) — included because it contradicts the
  README's implied claim.

---

## 16. Recommended Fixes

Ordered by value per unit of effort. Items 1–3 close the chain the judges will probe.

1. **Bind the advisory to the Panchayat and date (closes A-2).** Extend `GET /api/advisory` so that
   `rainfall_mm` becomes optional and, when omitted, the route reads the value itself from
   `Store.grid_rainfall` for `panchayat_id` + `date`; when it *is* supplied, compare it to the store
   and return HTTP 409 with both values if they differ by more than a rounding tolerance. Validate
   `date` against `Store.grid_dates` and 422 otherwise. Expose the verified pair in the response as
   `data_date` + `source` taken from the store, not from the query string.
   *Estimated effort: small. Depends on the Store already being in the route module.*

2. **Replace `||` with `??` in the frontend rainfall merge (closes A-1)**, and audit the remaining
   `||` fallbacks in `HomePage.tsx` / `AdvisoryPanel.tsx` for the same falsy-zero pattern. Add a
   frontend unit test for `rainfall_mm === 0`.
   *Estimated effort: trivial.*

3. **Regenerate the Layer-2 summary from the deployed field and stamp it (closes A-3, D-2).** Re-run
   `scripts/layer2_panchayat_mapping.py` and record the Layer-1 fingerprint from
   `outputs/metrics/layer1_manifest.json` into `outputs/layer2/layer2_qc.json`. Optionally add the
   fingerprint check to `backend/tests/test_artifact_consistency.py` so a retrain that forgets Layer 2
   fails a test.
   *Estimated effort: one script run (the geopandas overlay is minutes-to-hours and also writes the
   100 MB `panchayat_weather.csv`, which is gitignored).*

4. **Make the fallback advisory agree with the backend (closes A-5)** by deleting
   `generateLocalAdvisory` and showing an explicit "advisory service unavailable" state, or by
   replicating the tier table and dropping the crop/stage/aux claims from the fallback text.
   *Estimated effort: small.*

5. **Fix the unknown-stage semantics (closes A-6):** either treat `stage is None` as
   `evaluable=False` for R3, or change the `condition` string to state that an unknown stage is
   assumed heat-sensitive. Prefer the former; a missing input should not create a hazard.
   *Estimated effort: trivial.*

6. **Remove the hardcoded `15.9` (closes A-7)** and render "—" when
   `mae_improvement_pct` is null.
   *Estimated effort: trivial.*

7. **Include action bullets in the faithfulness allow-list (closes A-8)** by adding
   `RULE_ACTIONS[fired]` to `allowed_numbers`, or by relaxing the guard to the union of the trace and
   the action list the API is about to serve.
   *Estimated effort: trivial.*

8. **Align the GET/POST stage vocabulary (closes A-9)** by exporting `advisory.Stage` into the GET
   route's `StageQ`, or by documenting the deliberate asymmetry in the API docs.
   *Estimated effort: trivial.*

9. **Delete or wire `SAND_HIGH_G_PER_KG` and label the unused aux fields (closes A-10, C-7).** Either
   implement a sand-based irrigation rule or remove the constant and mark `sand/ocd/ph/bdod` in
   `evidence.aux` as "context only, not used by any rule". Do not let the API imply otherwise.
   *Estimated effort: small.*

10. **Decide the mapping method explicitly (addresses B-1, B-2).** Either (a) serve the area-weighted
    polygon value, matching `mapping_method` for the 60 % of rows already labelled `area_weighted`
    (this is what `panchayat_weather.csv` would give, and it changes served values), or (b) keep
    nearest-cell and re-label the UI to say "5.5 km grid cell at the Panchayat's centre", adjusting
    the XAI text and `mapping_method` accordingly. Option (b) is honest and cheap; option (a) is the
    larger architectural fix.
    *Estimated effort: (b) small, (a) large.*

11. **Refresh `outputs/metrics/metrics.json` and `outputs/layer2/block_rainfall.csv`, or delete the
    latter** (closes A-12, D-7).
    *Estimated effort: one `scripts/evaluate.py` run; `produce_block_rainfall.py` needs a vectorised
    rewrite first.*

12. **Extend the provenance manifest to Layer 2** so the summary and the geojson carry the Layer-1
    hash they were built from (reinforces item 3).

---

## 17. Tests That Could Not Be Executed

| Not executed | Why | Impact on the conclusions |
|---|---|---|
| Groq LLM rewrite path and Sarvam translation, live | No API keys in this environment. The fallbacks were exercised and behave safely (`/api/translate` without a key → 502 `invalid_api_key_error`, server stays up). | The *rule* decisions are unaffected (the LLM never decides). The faithfulness guard was tested directly against the engine instead (§6.6). |
| Browser-level UI interaction (clicking a Panchayat, changing crop, observing the panel) | No headless browser session available; the frontend was validated by `tsc --noEmit`, `next build`, and source inspection against the API contract. | The frontend findings A-1, A-4, A-5, A-7 are **source-and-API-verified**, not click-verified. A-1 in particular is a code-reading conclusion confirmed by a real `0.0 mm` API response; a browser check is still advisable. |
| Full `scripts/layer2_panchayat_mapping.py` regeneration | Needs geopandas + the LGD overlay over 87,735 polygons per date and writes the >100 MB gitignored `panchayat_weather.csv`; far outside the remaining budget. | A-3 is proven by recomputing the same quantity for three Panchayats from the served field; the *cause* of the mismatch (which historical run produced the file) is inferred from the max-value fingerprint, not reproduced. |
| `scripts/evaluate.py` re-run | Writes figures and `prediction/*.npz` for every model; ~6 model × 2 split full-grid passes. | `outputs/metrics/metrics.json` result rows are stale; the block the API actually reads (`meta`) is current. Noted as an open item. |
| Torch-dependent tests in the backend's light venv | `.qa-b1/.venv` has no torch (by design — the backend does not need it). | The ensemble arithmetic was verified in the torch venv instead (`qa/qa_ensemble_smoke.py`, max difference `0.0` against a manual weighted mean). |
| Live-inference route (`ENABLE_LIVE_INFERENCE=1`) end-to-end | Needs torch + `data/raw` in the same interpreter; not available in the backend venv. | The route now shares `scripts/ensemble.py` with `generate_pred.py`, so its model resolution is the same code path that was verified. |
| `models/model_b/c/c_weighted/d` re-evaluation under the new harness | Not needed: `ablation.py` regenerates all rows and was run to completion (82 s). | None. |

---

## 18. Final Logical-Integrity Verdict

> **"If a judge selects a Panchayat, selects a date, receives a rainfall value, and then receives an
> agro-advisory, can we prove that the final advisory is based on the correct Panchayat, correct date,
> correct rainfall prediction, correct contextual inputs, and correct rules — with no silent
> substitution, stale data, wrong spatial mapping, or unused-data claims?"**

**Answer: partially — and the exact boundary is identifiable.** Every link inside Layers 1 and 2 is
proven. The chain **breaks in two specific, reproducible places**, both between the API response and
the advisory call.

### 18.1 Links that ARE proven

| Link | Proof |
|---|---|
| **Name + context → exactly one Panchayat** | A bare duplicate name returns **409 with candidates** (RAMPUR = 61 rows); a contextual request resolves to the intended row, verified for 12 unique-named Panchayats across five states (12/12). No silent pick of `h.iloc[0]`. |
| **Panchayat → its own coordinate → its own fine cell** | 87,735 index rows / 87,735 distinct coordinates, max 1 Panchayat per coordinate (coordinate collapse eliminated). `location_precision: "exact"` only when the LGD own point exists. |
| **Date → NPZ index → value** | 54/54 traces exact, including first date, last date, and every month boundary; six out-of-range/malformed dates refused with 422, never substituted from another day or year. |
| **Value → API** | 13/13 served values byte-equal to `round(NPZ[date_idx, i, j], 2)`, with orientation and transpose controls that differ wherever the field is not locally constant. |
| **Units and scaling** | No double scaling (split mean 9.4134 vs reference 9.4015), no negatives, no Inf, rounding applied exactly once at the boundary. |
| **No stale data inside Layer 1/2** | The served grid's sha256 and the sha256 of both contributing checkpoints are recorded and verified by tests; the API republishes them. |
| **Rules are correct and deterministic** | All ten thresholds exact at T±ε; crop, stage, temperature, soil clay, NDVI and LULC each change the outcome in isolation; no contradictory actions; the decision never comes from the LLM. |
| **No cross-request contamination** | 8/8 requests (sequenced and concurrent) returned their own Panchayat's own value. |
| **Masked data is never fabricated** | A masked cell yields HTTP 422 with an explicit reason; aux absence lowers confidence instead of inventing a zero. |

### 18.2 Links that are NOT proven

| Broken link | What can go wrong | Evidence |
|---|---|---|
| **Frontend → advisory rainfall (A-1)** | A legitimate `0.0 mm` answer is falsy in JavaScript and is replaced by the **previous selection's** rainfall before it is sent to the advisory. The advisory then answers a question about a different Panchayat while the UI still shows the selected one. | `HomePage.tsx:73`; `NARASAPURAM`/`BHEEMUNIPALLI`/`ALUR` return `0.0` on 2022-07-10 |
| **Client → advisory endpoint (A-2)** | `GET /api/advisory` accepts any rainfall for any `panchayat_id` and any `date`, and reports the client's own date back as `data_date`. Nothing in the backend can detect the mismatch. | rainfall 0 → warning, rainfall 999 → alert on the same id/date; `date=1999-01-01` → 200; `panchayat_id=999999999` → 200 with `aux: null` |

Because of these two, the honest answer to "can we prove the final advisory used the correct
rainfall?" is **not yet** — the proof exists for Layers 1–2 and for the rule engine, and it is
*believed* for the frontend by reading its code, but it is not enforced or checkable at the point
where the advisory is computed. Two changes make it provable: `??` instead of `||`, and a
server-side lookup/verification of the rainfall for the supplied `panchayat_id` + `date`.

### 18.3 A second, narrower gap: stale Layer-2 data

The Layer-2 summary path is stale (**A-3**) and its rainfall columns are served by `/api/panchayats`,
but that path is **not** the one the advisory uses (`layer2_value` returns `None` because
`panchayat_weather.csv` is absent, so the served rainfall always comes from
`outputs/prediction_test.npz`). So the stale data does **not** contaminate the advisory — it
contaminates the search-result preview and would contaminate a future build that ships
`panchayat_weather.csv`.

### 18.4 Unused-data claims

One claim in the UI/API is stronger than the implementation: `evidence.aux` exposes five soil
properties and six land-cover fractions, but only **clay**, **NDVI** and **LULC dominant /
cropland-fraction** are read by any rule; `sand`, `ocd`, `ph`, `bdod` and the remaining fractions are
transport only, and `SAND_HIGH_G_PER_KG` is a dead constant (**A-10**). The aux layers that *are* used
are genuinely causal — verified by isolation, not by inspection.

### 18.5 Verdict in one line

**Layers 1 and 2 are logically sound and provably traceable; the rule engine is sound and provably
deterministic; the end-to-end guarantee fails only because the advisory receives its rainfall from the
caller without verification — two small fixes away from being fully provable.**

> **Status: both of those fixes have since been implemented — see §20.** The findings above are
> reported as they stood at `4b2bc46` (no logic was changed while validating).

---

## 19. Evidence Inventory

### 19.1 Executed experiments

| File | Covers | Key results |
|---|---|---|
| `qa/qa_logic_core.py` → `qa/qa_logic_core.json` | L1→L2 traces, orientation controls, units, polygon vs nearest-cell, summary staleness, temporal index, spatial sanity, temporal structure, model-output sanity, admin joins | 13/13 exact; 54/54 exact; area-weighted gaps to 3.44 mm; summary max 163.2806 vs served 193.96; 87,735/87,735 coordinates; ≥100 mm recall 0.005 |
| `qa/qa_logic_advisory.py` → `qa/qa_logic_advisory.json` | every threshold at T±ε, 11 one-input perturbations, action coherence, faithfulness guard | all boundaries exact; crop/stage/temp/soil/NDVI/LULC all causal; `is_faithful("…48 hours")` false |
| `qa/qa_logic_http.py` → `qa/qa_logic_http.json` | identity binding, ambiguity 409, sequence + concurrency isolation, zero-rain reachability, Layer-3 binding, enums/errors | 12/12 identity; 8/8 isolation; 3 Panchayats return exactly 0.0; advisory accepts rainfall 999 and `date=1999-01-01` |
| `qa/qa_f1_50.py` → `qa/qa_f1_50.json` | heavy-rain failure diagnosis | E at 50 mm: precision 0.438, recall 0.076 |
| `qa/qa_eval_f1.py` → `qa/qa_f1_50_variants.json` | per-checkpoint fixed + val-calibrated event metrics | E 0.129 / F 0.320 / EF 0.284 vs baseline 0.234 |
| `qa/qa_blend.py` → `qa/qa_blend.json` | blend/ensemble candidates on val+test | equal-weight E+F dominates the baseline at every metric |
| `qa/qa_ensemble_smoke.py` | ensemble loader correctness | max difference `0.0` vs the manual weighted mean; single-checkpoint override exact |
| `pytest` (`backend/.venv`) | backend behaviour, mapping, aux, contract, new artefact-consistency suite | **53 passed at validation time; 69 after the fixes in §20** |

### 19.2 Artefacts inspected

* Layer 1: `data/processed/{X,Y,M}_{train,val,test}.npy`, `meta.json`, `models/model_{b,c,c_weighted,d,e,f}.pt`, `models/ensemble.json`, `models/best_model.{pt,history.json}`, `models/model_e_history.json`, `models/model_f_history.json`, `outputs/prediction_test.npz`, `outputs/metrics/{ablation.csv,ablation.json,ablation_summary.md,layer1_manifest.json,metrics.json,metrics.csv}`
* Layer 2: `outputs/layer2/{panchayat_summary,panchayat_index,block_rainfall}.csv`, `layer2_qc.json`, `data/raw/administrative/panchayat/LGD_Panchayats.parquet`, `data/aux_data/admin/grid_admin_map_deccan.npz`
* Layer 3 / backend: `backend/{advisory,main,routes_data,data_store,metrics_loader,live_infer,aux_layers,build_panchayat_index}.py`, `backend/tests/*`
* Pipeline: `scripts/{config,train,ensemble,ablation,evaluate,layer2_panchayat_mapping,produce_block_rainfall,infer}.py`, `generate_pred.py`
* Frontend: `src/views/HomePage.tsx`, `src/views/LandingPage.tsx`, `src/components/advisory/AdvisoryPanel.tsx`, `src/components/search/PanchayatSearch.tsx`, `src/components/sidebar/{LeftSidebar,RightSidebar}.tsx`, `src/components/xai/XAIPanel.tsx`, `src/components/modals/ModelModal.tsx`, `src/api/backend.ts`, `src/types/index.ts`
* Docs: `README.md`, `docs/HANDOVER.md`, `docs/COMPREHENSIVE_QA_TEST_REPORT.md`, `docs/END_TO_END_QA_TEST_PLAN.md`, `backend/README.md`

### 19.3 Reproduction commands

```bash
# from the repo root (this branch's root IS the worktree that was validated, so
# the old `cd .qa-b1` prefix is gone; the QA scripts are committed at the root)
# Layer 1 / Layer 2 logic - needs torch + geopandas (root .venv, README section 5)
.venv/Scripts/python.exe qa/qa_logic_core.py
# advisory engine logic - needs the backend deps (fastapi, pydantic)
backend/.venv/Scripts/python.exe qa/qa_logic_advisory.py
# HTTP / end-to-end - backend must be running on 127.0.0.1:8000
backend/.venv/Scripts/python.exe qa/qa_logic_http.py
# test suites
cd backend && .venv/Scripts/python.exe -m pytest -q          # 69 passed
cd Frontend && npm run typecheck && npm run build
```

### 19.4 Build fingerprint at validation time

| Item | Value |
|---|---|
| Branch / commit | `fix/qa-hardening` @ `4b2bc46` (worktree `.qa-b1`) |
| Served grid | `outputs/prediction_test.npz`, sha256 `09410cf635cf5ab94aef61df97fd7cf31f120ca3409b1bf7eab0efd1ce4d5b20` |
| Deployed model | `model_e.pt` ×0.5 + `model_f.pt` ×0.5 (`models/ensemble.json`); `best_model.pt` = `model_e.pt` fallback |
| Checkpoint hashes | `model_e.pt` `75187a83…`, `model_f.pt` `569b3adc…` |
| Parameters | 117,329 (width 16, 5 channels, residual) |
| Grid | 122 days × 285 × 200, 44,243 finite cells/day, max 193.96 mm, mean 9.413 mm |
| Test-split metrics (deployed) | MAE 8.428 / RMSE 14.998 / corr 0.5267; F1 0.601 / 0.442 / 0.284 @ 10/25/50 mm |
| Baseline metrics | MAE 9.603 / RMSE 18.183 / corr 0.385; F1 0.477 / 0.343 / 0.234 |
| Test suites | `pytest` 53 passed; `tsc --noEmit` clean; `next build` clean |

---

## 20. Post-fix status: A-1, A-2, A-5 and A-6 addressed

The two High findings were fixed immediately after this report was written. The findings are kept
above as they were found — nothing was rewritten to look better in hindsight — and this section
records what changed and how it was verified. `pytest` went from 53 to **69 passed**.

### 20.1 A-2 — the advisory now verifies the rainfall (fixed)

`GET /api/advisory` resolves the authoritative Layer-1 value for the requested `panchayat_id` + `date`
via `data_store.Store.panchayat_grid_value()`, which `routes_data.register` publishes on
`app.state.rainfall_resolver`. It never invents a value: when the answer is unobtainable, `rainfall_mm`
is `None` and `reason` names the missing piece. The route then behaves as follows:

| situation | before | after |
|---|---|---|
| `rainfall_mm` matches the stored value (±0.011 mm) | 200, unverifiable | **200**, `verification.rainfall = "verified_against_layer1"` |
| `rainfall_mm` omitted | 422 (required param) | **200**, value resolved server-side, `"resolved_from_layer1"` |
| `rainfall_mm` disagrees (e.g. 999 vs 95.55) | **200 — silently wrong** | **409** with `supplied_mm`, `expected_mm`, `difference_mm` |
| `date` outside the served grid (e.g. `1999-01-01`) | **200**, `data_date` echoed back | **422**, `reason: "date_unavailable"` |
| masked cell (e.g. AMRUTHALUR pid 199960) | 200 if a value was supplied | **422**, `reason: "masked_cell"` |
| `panchayat_id` with no coordinate | **200** with `aux: null` | **422**, `reason: "unknown_panchayat"` |
| no data layer mounted (text-only deployment) | 200, indistinguishable | 200 labelled `unverified_no_data_source`; `rainfall_mm` required |

The response now carries a `verification` block — `panchayat_id`, verified `date`, the rainfall status,
the `source` string, the fine-grid `cell`, `location_precision`, and the status of aux, temperature and
humidity — so a caller can see exactly what was checked. Temperature and humidity are still
**caller-supplied** and are labelled as such rather than being implied to be verified; verifying those
would need the same treatment against `/api/weather`. `POST /api/advisory` is unchanged and remains the
route for deliberate what-if runs.

Reproduced against the live server (`uvicorn app:app`):

```
rainfall_mm=95.55 (ALURU pid 220447)  -> 200  verified_against_layer1  cell [47,67]  aux verified
rainfall_mm=999                       -> 409  supplied 999.0, expected 95.55, difference 903.45
rainfall_mm omitted                   -> 200  resolved_from_layer1, evidence.rainfall_mm 95.55
date=1999-01-01                       -> 422  reason: date_unavailable
date=""                               -> 422  reason: date_unavailable
panchayat 199960 (masked)             -> 422  reason: masked_cell
panchayat 999999999                   -> 422  reason: unknown_panchayat
panchayat 196404, rainfall_mm=0       -> 200  evidence.rainfall_mm 0.0   (zero survives)
```

### 20.2 A-1 — the frontend no longer substitutes a previous value (fixed)

`HomePage.tsx` now merges with `??` instead of `||`, so a legitimate `0.0 mm` survives:

```ts
rainfall_mm: response.prediction.rainfall_mm ?? selected.rainfall_mm,   // was `||`
```

The sibling fields already used `??`, so this was the single inconsistent line. Additionally, the
advisory panel no longer masks a refused request: `api/backend.ts` now throws an `ApiError` carrying
the HTTP status, and `AdvisoryPanel.loadAdvisory` treats **409/422 as a hard failure** — it renders an
explicit "Advisory withheld" state with the server's explanation instead of silently falling back to the
local rule set (now deleted — §20.3). Without that second change the new 409 would have been swallowed by the
old `catch` and converted back into a plausible-looking advisory.

### 20.3 A-5 — the client-side fallback advisory is gone (fixed)

`AdvisoryPanel.tsx` carried its own rainfall rule table — `>= 64.5` alert, `>= 24.5` warning,
`>= 6.5` watch, else info — against the engine's `64.5` / `24.5` / `< 3.0 × window` (`advisory.py`).
The two disagreed in the band a farmer is most likely to see, and the fallback could also not read
temperature, soil, NDVI or land cover at all, so an outage could turn an `alert` into an `info` on the
same numbers.

The table has been deleted. A network failure now renders an explicit **"Advisory unavailable"** state
saying the rules run server-side and are not reproduced in the browser, with a **Retry** action; a
**409/422** still renders **"Advisory withheld"** with the server's own explanation (§20.2). The panel
can therefore no longer show advice that differs from the backend: it shows either the backend's answer
or no advice at all. `backend/tests/test_frontend_contract.py` guards this at source level — the panel
must not re-declare the `64.5` / `24.5` mm tiers and must still carry both states — because this repo has
no frontend test harness.

### 20.4 A-6 — an unknown crop stage no longer fires heat stress (fixed)

`_r_heat` treated `stage = None` as heat-sensitive and merely downgraded the severity, so an unknown
input *increased* the hazard while the rule's own `condition` string said the stage had to be in the
sensitive set. It now returns `evaluable=False`, `fired=False`, `severity="none"` and the flip hint
`"crop stage missing - heat stress could not be evaluated"` when the stage is unknown: a missing input
can no longer manufacture a hazard. `evaluate()` already lowered confidence for a missing stage, so the
uncertainty is reported instead of acted upon. Verified by a stage sweep at wheat / 34.5 °C — `sowing`,
`vegetative`, `maturity` → `low`; `flowering`, `grain_filling` → `high`; unknown → `evaluable=False`,
confidence 0.5 — and covered by two tests in `backend/tests/test_advisory.py`.

### 20.5 Consequences for the findings list

| ID | Status |
|---|---|
| **A-1** (falsy-zero substitution) | **Fixed** — `??` + the panel surfaces refusals |
| **A-2** (advisory not bound to Panchayat/date) | **Fixed** — resolve, check, refuse, and publish what was checked |
| A-3, A-4 (stale Layer-2 summary, season mean labelled with a date) | Open — needs a `layer2_panchayat_mapping.py` re-run and a fingerprint stamp |
| A-5 (divergent fallback advisory) | **Fixed** — the client-side rule table is deleted; an outage shows an explicit "Advisory unavailable" state instead of a second opinion |
| A-6 (unknown stage fired heat stress) | **Fixed** — R3 is unevaluable without a known crop stage |
| A-7 … A-12, B-1, B-2, D-7 | Open — see §16 |

### 20.6 The judge question, re-answered

With A-1 and A-2 fixed, the previously unproven links are now enforced by the server rather than by
client good behaviour:

* **correct Panchayat** — proven at 12/12, and now a mismatched or unverifiable Panchayat is refused
  rather than trusted;
* **correct date** — proven at 54/54 on `/auth/`, and now the advisory refuses any date outside the
  served grid instead of echoing it;
* **correct rainfall prediction** — proven at 13/13 against the NPZ, and now a disagreeing client value
  is rejected with both numbers;
* **no silent substitution** — the substitution path in the frontend is closed and the server refuses
  rather than substitutes;
* **one advisory rule set** — the browser carries no second rule table, so an unreachable backend yields
  "Advisory unavailable" rather than different advice on the same numbers;
* **no unused-data claims** — unchanged, still open for `sand`/`ocd`/`ph`/`bdod` (**A-10**): those are
  transported but read by no rule.

Remaining honest caveats: temperature and humidity are still supplied by the caller (labelled, not
verified), the Layer-2 summary path is still stale (**A-3**, not on the advisory path), and the served
rainfall is still a nearest-cell value rather than a polygon mean (**B-1**).
