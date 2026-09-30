# COMPREHENSIVE QA / VALIDATION TEST REPORT
## Panchayat Rainfall Downscaling & Agro-Advisory

**Report type:** executed validation report (not a plan)
**Baseline spec:** `END_TO_END_QA_TEST_PLAN.md` (898 lines) — treated as the baseline QA specification
**Product under test:** branch **`b1`** @ **`62d4a343911ca9ec8ddd5189c7cdafaf633a5a72`** of `https://github.com/yashkapur0403/weather-downscaling`
**Tested on:** this laptop (Windows Git Bash, Python 3.14.6 / 3.12-torch venv, Node 24.18, npm 11.16), isolated `git worktree` at `.qa-b1/`
**Report date:** 2026-09-30
**Tester method:** black-box over real HTTP against a live `uvicorn` server, white-box against the real committed artefacts, independent metric reproduction from `data.zip` + checkpoint, frontend build/HTTP gates. **No application logic was modified.**

---

> **Status note (added at submission time).** This report was executed against `b1` @ `62d4a34`,
> *before* the QA-hardening commits. Its findings are kept exactly as they were found. Re-checked on
> the hardened branch:
>
> | finding | status on this branch |
> |---|---|
> | DOC-01 / **D-2** (`backend/.env.example` missing) | **fixed** — the file exists |
> | DOC-02 / **D-7** (docs describe the old `weather-downscaling-main/` layout) | **fixed** — README, HANDOVER and `backend/README.md` now describe `backend/` + `Frontend/` at the repo root, with the current test count and selection rule |
> | DOC-04 / **D-5** (`frontend_backend_match.patch` cannot apply, but the README says to apply it) | **fixed** — `backend/README.md` now says it is already applied and must not be re-applied |
> | DOC-05 (`scripts/ablation.py` / `evaluate.py` were generator stubs on `b1`) | **fixed** — the real scripts were restored and re-run |
> | **D-1** (989 shared block/district centres instead of per-Panchayat points) | **fixed** — `backend/build_panchayat_index.py` rebuilds the index from LGD polygons; 87,735 rows now carry 87,735 distinct coordinates |
> | **D-3** (Layer-2 CSV/GeoJSON listed as artefacts but absent) | **documented, not fixed** — both are >100 MB and deliberately gitignored (`.gitignore`); the backend serves per-date values from `outputs/prediction_test.npz` at each Panchayat's own polygon point |
> | **D-6** (dead duplicate `/api/weather` Next route + unused `fetchWeather`) | **fixed** — the duplicate route was removed |
> | **D-8** (`HANDOVER.md` documents only the ML pipeline) | **open by design** — `backend/README.md` and these two QA reports are the app-side documentation |
> | **D-4** (`layer2_qc.json` paths from another machine), **D-9** (`DADRA,NAGAR HAVELI,DAMAN & DIU` state-string mismatch) | **open**, low severity |
>
> **Path note.** Paths written as `.qa-b1/…` below are the isolated `git worktree` used at test time.
> The evidence now lives in this repository under `qa/` (`qa/qa_*.py` with the recorded
> `qa/qa_*.json` results — moved there from the worktree root, hence the prefix in the paths below),
> and `.qa-b1/backend` is now just `backend/`.
> `.qa-b1/_backup_original/` (the pre-fix backup) and the `.log` files are local-only and are
> gitignored on purpose, so a fresh clone will not have them.
>
> The two advisory defects this report surfaced — `GET /api/advisory` trusting a caller-supplied
> `rainfall_mm`, and the frontend's `x || fallback` turning a legitimate `0.0 mm` into the previous
> selection's value — are **fixed** (server-side verification with 409/422, and `??`), and the honest
> `pytest` count is now **69**. Details and evidence: `LOGICAL_VALIDATION_REPORT.md` §20.

---

## 1. Executive Summary

The product's **core numeric pipeline is correct and independently reproducible.** Every one of the ten Panchayat rainfall values the plan predicted (2 dp) was reproduced exactly over HTTP from the committed U-Net grid, the reported Layer-1 test metrics were **reproduced bit-for-bit** (absolute difference `0.00e+00` on MAE / RMSE / correlation and F1@10/25/50), the model checkpoint architecture, channel order and parameter count all match the documentation and `/api/metrics`, the 37-test backend suite is green, the frontend typechecks, lints, builds and serves, and the rule engine degrades safely with no LLM keys.

Two **genuine product defects** were found, and a set of **data, documentation and environment gaps** were confirmed. Notably, three "failures" in the baseline plan's own list turned out to be **errors in the plan**, not in the product, and were reclassified after root-cause analysis.

| Verdict | **PASS WITH ISSUES** |
|---|---|
| Blocking demo criteria | **9 of 10 pass** (TC-03 cold-start expectation is a plan error — warm value is correct) |
| Genuine product defects | **2** (1 High, 1 Medium) |
| Confirmed data/documentation gaps | 6 |
| Model-skill caveat | 1 (heavy-rain F1 below the bilinear baseline) |

### 1.1 Counts (as executed)

| Suite | Total | Pass | Fail | Blocked | Info / N-A |
|---|---:|---:|---:|---:|---:|
| Automated unit tests (`pytest tests`) | 37 | 37 | 0 | 0 | 0 |
| Black-box API (live HTTP) | 48 | 44 | 3 | 0 | 1 |
| White-box data/code | 32 | 26 | 4 | 1 | 1 |
| Offline follow-up (mask/aux/dataset) | 10 | 8 | 1 | 0 | 1 |
| UAT + E2E trace + integration | 14 | 14 | 0 | 0 | 0 |
| Model-checkpoint validation | 1 | 1 | 0 | 0 | 0 |
| Metric reproduction (2 rows × 6 stats) | 12 | 12 | 0 | 0 | 0 |
| Frontend gates (install / typecheck / lint / build) | 4 | 4 | 0 | 0 | 0 |
| Round-2 deep validation (value analysis + remaining paths) | 12 | 6 | 1 | 0 | 5 |
| Round-2 extended probes (env variants, enums, missing files) | 10 | 7 | 2 | 0 | 1 |
| **Total** | **180** | **159** | **11** | **1** | **9** |

### 1.2 Failures after root-cause reclassification

The 8 "as-run" failures were investigated. **Only 5 are product or deliverable issues; 3 are defects in the baseline plan.**

| ID | As-run | After root cause | Product issue? |
|---|---|---|---|
| BB-01 | FAIL | **confirmed bug** (Medium) | Yes |
| BB-13 | FAIL | **confirmed bug** (High) | Yes |
| D-29 | FAIL | confirmed gap (Medium) | Yes (deliverable) |
| D-30 | FAIL | confirmed gap (Low) | Yes (deliverable) |
| D-60 | FAIL | environment/data-availability (Medium) | Yes (packaging) |
| XAI-02 | FAIL | **test was wrong → PASS** | No |
| D-11 | FAIL | **plan expectation wrong → PASS** | No |
| D-41 | FAIL | **plan expectation wrong → PASS** | No |
| D-50 | BLOCKED | **unblocked → PASS** | No |

---

## 2. What was tested, and how

| Dimension | Method | Evidence file |
|---|---|---|
| Backend API (normal, invalid, boundary, error, security, perf) | 48-case script over live `uvicorn` on `127.0.0.1:8000` | `qa/qa_blackbox.json` |
| Data / code white-box | 32-case script over the committed CSV/NPZ/JSON | `qa/qa_whitebox.json` |
| Mask / aux / dataset follow-up | 10-case script | `qa/qa_followup.json` |
| UAT + E2E + integration | 14-case script (HTTP + source recomputation) | `qa/qa_uat.json` |
| Model checkpoint | Torch load + instantiate + `load_state_dict(strict=True)` | `qa/qa_model.py` output |
| Layer-1 metrics | Re-ran the checkpoint over `X/Y/M_test` extracted from `data.zip` using the *same* protocol as `evaluate.py`/`train.py` | `qa/qa_repro_metrics.py` output |
| Frontend | `npm ci`, `npm run typecheck`, `npm run lint`, `npm run build`, `npm run start` + HTTP probes | `Frontend/` |
| Regression | Re-ran plan's documented cases and compared | §9 |

Reproduction commands:

```bash
# live server (already running)
cd backend && ../.venv/Scripts/python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
# suites (run from the repository root; these need the backend deps: fastapi/pydantic/httpx/pandas)
./.venv/Scripts/python.exe qa/qa_blackbox.py && ./.venv/Scripts/python.exe qa/qa_whitebox.py \
  && ./.venv/Scripts/python.exe qa/qa_followup.py && ./.venv/Scripts/python.exe qa/qa_uat.py
# model + metrics (need torch; see requirements.txt)
./.venv/Scripts/python.exe qa/qa_model.py
./.venv/Scripts/python.exe qa/qa_repro_metrics.py
```

`qa/qa_logic_*.py` and `qa/qa_ensemble_smoke.py` need torch (they load the checkpoints); `qa/qa_logic_http.py`
also needs the live server and `pandas`. Nothing in `qa/` is imported by the pipeline or the app.

---

## 3. Detailed Results

Status legend: **P** = Pass, **F** = Fail, **B** = Blocked, **i** = Info / not applicable.

### 3.1 Automated unit suite — `backend/tests` (baseline GATE)

| Test ID | Category | Test | Expected | Actual | Status | Sev | Evidence | Root Cause |
|---|---|---|---|---|---|---|---|---|
| TC-02 | Unit | `pytest -q tests` | 37 passed, no network/keys | `37 passed, 1 warning in 1.66s` | **P** | — | `.qa-b1/backend` | Matches plan baseline. |

The 1 warning is a `StarletteDeprecationWarning` about `httpx` in `TestClient`; not a failure.

### 3.2 Black-box API — live HTTP (44 P / 3 F / 1 i)

| Test ID | Category | Test | Expected | Actual | Status | Sev | Evidence | Root Cause |
|---|---|---|---|---|---|---|---|---|
| BB-01 | health | `GET /` on a **cold** server | `panchayats_loaded == 86103` | `null` | **F** | Med | verified over HTTP on fresh boot | `routes_data` health handler reads the cached `S.store` (None) instead of calling `store()`. Warm calls return `86103`. |
| BB-02 | health | `GET /` warm | `86103` | `86103` | **P** | | | |
| BB-03 | health | `GET /health` | `status:"ok"`, 2 models | `{"status":"ok","groq_models":["openai/gpt-oss-120b","openai/gpt-oss-20b"]}` | **P** | | | |
| BB-04 | search | `GET /api/panchayats?q=kundapura` | real rows, real mapping | 5 rows; ALURU `direct_grid` n_cells=2 mean=12.13 | **P** | | | |
| BB-05 | search | no `q` | 422 | 422 | **P** | | | |
| BB-06 | search | `limit=100` | 422 | 422 | **P** | | | |
| BB-07 | search | unmapped hidden (`KANNEGANTIVARIPALEM`) | `[]` | `[]` | **P** | | | |
| BB-08 | search | substring `ALLAMUDI` | matches `CHEMALLAMUDI` | `[CHEMALLAMUDI]` | **i** | | | |
| BB-09 | geocode | ALURU/Udupi | `13.679452, 74.742466, exact` | exact match | **P** | | | Coordinate is the KUNDAPURA block centroid (D-1). |
| BB-10 | geocode | unknown | 404 | 404 | **P** | | | |
| BB-11 | weather | `GET /api/weather` | real T/RH/elev + sources | `T=23.8 RH=93 elev=17.0 src=Open-Topo-Data (SRTM 90 m) err=None` (895 ms) | **P** | | | |
| BB-12 | auth | `POST /auth/` ALURU full context | 50.49 + real weather | `50.49 … prec=exact src=U-Net prediction grid` (290 ms) | **P** | | | |
| BB-13 | auth | **name-only** `{"panchayat_name":"ALURU"}` | resolve to the intended Panchayat or refuse | **`3.15` mm** (the Guntur ALURU, pid 200620), answer text has no disambiguation | **F** | **High** | 4 rows named exactly `ALURU` | `Store.find()` returns `h.iloc[0]`. 8,404 duplicated names cover 26,666 rows. |
| BB-14 | auth | outside domain (`lat=40`) | 422 | 422 | **P** | | | |
| BB-15a | auth | `2021-01-01` | 422 | 422 | **P** | | | |
| BB-15b | auth | `2022-10-01` | 422 | 422 | **P** | | | |
| BB-16 | auth | malformed JSON | 4xx | 422 | **P** | | | |
| BB-17 | auth | masked/coastal cell | 422, never fake 0.0 | 422 `"model output is not finite at this cell (sea / excluded cell)"` | **P** | | | |
| BB-18 | auth | no-coordinate Panchayat (AMBOLI) | 422 | 422 with `no coordinates … Available dates: 2022-06-01 to 2022-09-30` | **P** | | | |
| BB-19 | metrics | `GET /api/metrics` | D=8.167 / base=9.603 / +15.0% / 117,329 | exact match (9 ms) | **P** | | | |
| ADV-70/50.49/30/12/2 | advisory | rainfall bands | alert/warning/warning/info/watch | all exact, `risk_level` matches, `message_source=template` | **P** | | | |
| ADV-NONE | advisory | missing `rainfall_mm` | 422 | 422 | **P** | | | |
| ADV-BAD | advisory | `crop=banana` | 422 | 422 | **P** | | | |
| ADV-LANG | advisory | `lang=xx-XX` | 422 | 422 | **P** | | | |
| ADV-SENS | advisory | change rainfall 5→70 | severity & actions change | info→alert; 2→3 actions | **P** | | | |
| ADV-ECHO | advisory | crop/stage echo | rice/flowering | rice/flowering | **P** | | | |
| BLK-01 | advisory | `POST /api/advisory/block` | `[heat_stress, heavy_rain]` | exact | **P** | | | |
| XAI-ST | xai | `/api/explain/status` | enabled=false | `{enabled:false, provider:groq, model:openai/gpt-oss-120b}` | **P** | | | |
| XAI-01 | xai | `/api/explain` no key | provider=rules, grounded factors | provider=rules; 4 trained-channel factors; summary contains `50.5 mm/day` | **P** | | | |
| XAI-02 | xai | `/api/explain` **without** question | (test asserted non-empty answer) | `answer: None` | **F**(reclass) | — | — | **Test was wrong.** A missing `question` legitimately yields `answer: None`; XAI-04 (with question) passes. Not a defect. |
| XAI-03 | xai | weights ∈ [0,1] | all valid | true | **P** | | | |
| XAI-04 | xai | `/api/explain` with question, no key | non-empty grounded answer | `"The language model is unavailable or its answer could not be verified…"` | **P** | | | |
| ROB-01 | robust | repeat identical request | stable | `50.49 == 50.49` | **P** | | | |
| ROB-02 | robust | 5 concurrent identical | all string-equal | all `50.49` | **P** | | | |
| SEC-01 | security | path traversal `../../etc/passwd` | no disclosure | `[]` | **P** | | | |
| SEC-02 | security | 5000-char query | bounded | 200, bounded | **P** | | | |
| SEC-03 | security | SQL-ish string | no server error | `[]` | **P** | | | |
| SEC-04 | security | error body | `detail` only | `keys=['detail']`, no traceback | **P** | | | |
| SEC-05 | security | API keys exposed? | no | none | **P** | | | |
| PERF-00..04 | perf | `/`, search, `/auth/`, advisory, metrics | <50/<300/<1500/<500/<200 ms | 3 / 78 / 292 / 5 / 8 ms | **P** | | | |

### 3.3 White-box data/code (26 P / 4 F / 1 B / 1 i)

All of the following reproduced the plan's verified statistics exactly: `meta.region="deccan"` (not the Western Ghats pilot); ROI `11.5–25.5 N / 71.5–81.25 E`; 610 dates `2018-06-01…2022-09-30`; split 366/122/122 with `train{2018,2019,2020} val{2021} test{2022}`; channel order `[imd_rain, dem, era5_t2m, era5_t2m_max, era5_dewp]`; `rain_scale=100.0`; `elev_scale=1222.6514892578125`; fine grid 285×200, `fine_sub=5`; land mask 1890/390 coarse, 47,250 land fine cells, **44,243 target-valid**; served NPZ dates **identical** to `meta.split.test.dates`; NPZ lat/lon identical to meta fine grid; rainfall `0.0 … 163.2806`, no Inf, dtype float32; NPZ max == `layer2_qc.json` max; `panchayat_summary.csv` 87,735 rows with distribution `{area_weighted 52893, direct_grid 31740, unmapped 1632, nearest_fallback 1470}`; unmapped rows all `n_cells==0`; QC `mapped=86103`; all `n_days=122`; no negative rainfall; `block_rainfall.csv` == 137,006 rows.

| Test ID | Category | Test | Expected | Actual | Status | Sev | Root Cause |
|---|---|---|---|---|---|---|---|
| D-01..D-10, D-12..D-15, D-20..D-28 | data/code | dataset geometry, split, mask, channels, ranges, QC | as plan | all exact | **P** | | |
| **D-11** | L1-output | NaN cells == masked cells | plan: 3,007 | **1,556,354** | **F→P**(reclass) | High | **Plan error.** `generate_pred.py` does `np.where(M_test==1, pred, nan)` over the **full 57,000-cell fine grid**, so NaN = sea (9,750) + invalid land (3,007) = 12,757/day × 122 = **1,556,354**. Per-day finite cells = **44,243 = `n_target_valid_fine_cells` exactly**; mask is constant across all 122 days. The mask IS fully honoured. |
| **D-29** | L2-map | `panchayat_weather.csv` exists | present (README §8) | **absent** | **F** | Med | Not committed on `b1`; `/auth/` therefore serves the raw grid cell, not a polygon-aggregated Layer-2 value. |
| **D-30** | L2-map | `panchayat_weather.geojson` exists | present (README §8) | **absent** | **F** | Low | Same as D-29; no per-Panchayat geometry in the app. |
| D-23 | L2-map | index rows / distinct coords | 86,075 / 989 | exact | **P**(risk) | High | Confirms the coordinate-collapse limitation D-1. |
| D-25 | L2-map | index coverage | 86,075 of 86,103 (28 missing) | exact; 28 = DNH/Daman rows | **i** | | |
| **D-41** | aux | NDVI composites/range | plan: 20 layers via key `fine_values` | keys include **`ndvi`**, shape **`(20, 47250)`** (months-first), range **−0.0640 … 0.8453**, 8.045% NaN, 20 months `2018-06…` | **F→P**(reclass) | — | **Plan error** (wrong key name and axis). Array is present and correct. |
| **D-50** | model | checkpoint validation | loads | **unblocked**: `load_state_dict(strict=True)` OK, 117,329 params | **B→P** | — | Was blocked only because the API venv had no torch; re-run with the torch venv. |

### 3.4 Offline follow-up (8 P / 1 F / 1 i)

| Test ID | Category | Test | Expected | Actual | Status | Sev |
|---|---|---|---|---|---|---|
| D-11 | mask | NaN == complement of target-valid over 57,000-cell grid | `122*(57000-44243)=1556354` | `1556354`, per-day finite `[44243]` | **P** | High |
| D-11b | mask | mask constant across days | identical | `constant=True` | **P** | |
| D-41 | aux | NDVI key/shape/range | `(20,47250)`, −0.064…0.845 | matches | **P** | Low |
| **D-60** | dataset | `data/processed/*.npy` present for rebuild | X/Y/M for 3 splits | **only `meta.json`** committed | **F** | Med |
| D-61 | dataset | Deccan (not Western Ghats) | `deccan` | `deccan` | **P** | High |
| D-62 | dataset | served npz lat/lon == meta fine grid | identical | `lat=True lon=True` | **P** | |
| D-42 | aux | soil `(47250,5)`, ~9% NaN | yes | `(47250,5)`, 8.95% | **P** | |
| D-43 | aux | LULC `(47250,6)`, fractions valid | yes | `(47250,6)` | **P** | |
| D-44 | aux | admin map | 15 states / 213 districts / 1093 subdistricts | districts=213, subdistricts=1093 (state field is per-cell) | **i** | |
| D-45 | L2 | block_rainfall 137,006 rows | yes | 137,006 | **P** | |

### 3.5 UAT + End-to-end trace + Integration (14 / 14 P)

| Test ID | Category | Test | Expected | Actual | Status | Sev |
|---|---|---|---|---|---|---|
| E2E-220447 | e2e-trace | ALURU coord→npz[46,67]→`/auth/` | 50.49, direct_grid/2 | `auth=50.49`, search `direct_grid/2`, coord `(13.679452,74.742466)` | **P** | High |
| E2E-201416 | e2e-trace | NAGULAMALLIAL | 41.77 | 41.77 via npz[143,156] | **P** | High |
| E2E-202120 | e2e-trace | AMERDA | 27.68 | 27.68 via npz[128,153] | **P** | High |
| E2E-254406 | e2e-trace | CURDI | 16.73 | 16.73 via npz[77,57] | **P** | High |
| UAT-J1 | UAT | select→projection→advisory→XAI consistent | 50.49 / warning / summary has 50.5 | all consistent | **P** | High |
| UAT-J2 | UAT | change Panchayat → different value | 50.49 ≠ 16.73 | differs | **P** | High |
| UAT-J3 | UAT | change date → date-specific | 2022-06-01=0.0, 2022-07-10=50.49, 2022-09-30 in range | as expected | **P** | High |
| UAT-J4 | UAT | unavailable data | clear 422, no fabrication | 422 `no coordinates … available dates` | **P** | Med |
| UAT-J5 | UAT | masked cell | 422 not 0.0 | 422 `not finite` | **P** | Med |
| UAT-J6 | UAT | out-of-range date | 422 naming range | 422 | **P** | Med |
| UAT-J7 | UAT | extreme rain | alert + actions | `alert`, `R1_HEAVY_RAIN` fired | **P** | Med |
| INT-1 | integration | `/auth` value recomputed from NPZ | npz[46,67]=50.49 | exact | **P** | High |
| INT-2 | integration | `/api/metrics` vs ablation.json + param formula | 8.167 / 117,329 | exact, `improvement=15.0` | **P** | High |
| INT-3 | integration | aux soil/NDVI/LULC feed the rules? | no wiring | confirmed not referenced by any route | **P**(gap) | Med |

Additional contract probes executed:

| Probe | Result |
|---|---|
| `/api/advisory` with frontend params `temperature_c` + `irrigation_available` | 200; `evidence.temperature_c=26.0`; irrigation branch produces `"Irrigation is not available: ask your agriculture office…"` |
| `/api/panchayats` row fields vs frontend `Panchayat` interface | **all fields present** (plus `location_precision`, `rainfall_basis`) |
| frontend `classifyRisk` bands vs backend `_risk_level` | **identical** (`<2.5/<10/<25/<50/else`) |
| `POST /api/explain` weights outside [0,1] | 422 |
| `POST /auth/` empty body `{}` | 422 |
| `POST /api/translate` without Sarvam key | **502** with clear `Translation service error: … invalid_api_key_error`; server stays alive |

### 3.6 Frontend gates (4 / 4 P)

| Test | Command | Result | Status |
|---|---|---|---|
| Install | `npm ci` | `added 156 packages in 2m` | **P** |
| Typecheck | `npm run typecheck` (`tsc --noEmit`) | 0 errors | **P** |
| Lint | `npm run lint` (`oxlint`) | **0 errors, 12 warnings** | **P** |
| Build | `npm run build` (Next.js 16.3.7 Turbopack) | `✓ Compiled successfully`; routes `/`, `/_not-found`, `ƒ /api/weather`, `/dashboard` | **P** |
| Serve | `npm run start`, HTTP probe | `/`→200, `/dashboard`→200 | **P** |
| Next `/api/weather` route | direct HTTP | **works** (200, real Open-Meteo data) but **unused by any component** — the frontend calls the backend's `/api/weather` | **i**(D-6) |

---

## 4. Layer-1 Model Validation

### 4.1 Checkpoint vs configuration vs runtime

`models/best_model.pt` loaded and instantiated successfully with `load_state_dict(strict=True)` (26 tensors, no missing/unexpected keys).

| Property | Documented | Configured (`ckpt`) | Runtime (independent) | Match |
|---|---|---|---|---|
| Input channels | `[imd_rain, dem, era5_t2m, era5_t2m_max, era5_dewp]` | same | same | ✅ |
| Channel order | same order as `meta.channels` | same | same | ✅ |
| Width | 16 | 16 | 16 | ✅ |
| Residual | true | true | true | ✅ |
| rain_scale | 100.0 | 100.0 | 100.0 | ✅ |
| Parameters | 117,329 | — | **117,329** (`sum(p.numel())`) | ✅ |
| `unet_param_count(5,16)` helper | 117,329 | — | **117,329** | ✅ |
| `/api/metrics.n_parameters` | `"117,329"` | — | exact | ✅ |
| Architecture | `SmallUNet`, 2 downsample levels, `ConvBlock` (2× conv3×3+ReLU) | matches code | matches | ✅ |
| Trained head | learned residual | `head.weight` absmax `0.0276`, not zero | — | ✅ (not an untrained baseline) |

**Training config** (`ckpt.args`): loss `mae`, lr `0.001`, seed `42`, epochs `220`; `best_epoch=207`, `best_val_loss=0.0640`, `train_seconds=112.3`. `best_model_history.json` holds **247** epoch rows (see finding I-4).

### 4.2 Reported vs independently reproduced metrics (test split)

Reproduced by re-running the checkpoint over `X_test/Y_test/M_test` extracted from `data.zip`, using the **same masking and thresholds** as `evaluate.py::evaluate_model` + `train.py::metrics`.

| Metric | Reported (`ablation.json` test D) | Independently reproduced | Abs. diff |
|---|---:|---:|---:|
| MAE (mm) | 8.167058 | **8.167058** | 0.00e+00 |
| RMSE (mm) | 16.013796 | **16.013796** | 0.00e+00 |
| Correlation | 0.446054 | **0.446054** | 0.00e+00 |
| F1 ≥10 mm | 0.393391 | **0.393391** | 0.00e+00 |
| F1 ≥25 mm | 0.280880 | **0.280880** | 0.00e+00 |
| F1 ≥50 mm | 0.014729 | **0.014729** | 0.00e+00 |
| n_events ≥10/25/50 | 1618246 / 624843 / 167065 | **identical** | 0 |

Baseline row A also reproduced exactly: MAE 9.602921, RMSE 18.183088, corr 0.384973, F1@10 0.477027, F1@25 0.342578, F1@50 0.234408. MAE improvement = `(9.6029−8.1671)/9.6029 = 15.0%`. ✅

**Conclusion:** every headline Layer-1 metric is reproducible from the committed artefacts; nothing is hardcoded or inflated.

### 4.3 Model-skill caveat (finding I-3)

The selected model (**D**) wins on MAE/RMSE/correlation but is **worse than the bilinear IMD baseline (A) at every heavy-rain threshold** on the test split:

| Threshold | F1 model D | F1 baseline A |
|---|---:|---:|
| ≥10 mm | 0.393 | **0.477** |
| ≥25 mm | 0.281 | **0.343** |
| ≥50 mm | **0.015** | **0.234** |

This is a real, honest limitation (the model is conservative on extremes; ≥50 mm recall is 0.0075). It does not invalidate the product, but the "improves over baseline" claim must be scoped to *mean error*, not event detection. The UI/`/api/metrics` already exposes `event_f1_test`, so it can be disclosed.

---

## 5. End-to-End Value Traces

### 5.1 Chain (as actually wired on b1)

```
UI number + advisory
  └─ GET /api/advisory?rainfall_mm=<value>       ← caller passes the value
       └─ UI got <value> from POST /auth/ → prediction.rainfall_mm
            ├─ (1) outputs/layer2/panchayat_weather.csv   [ABSENT on b1 → skipped]
            ├─ (2) outputs/prediction_test.npz            ← USED
            │       nearest fine cell to the coord in outputs/layer2/panchayat_index.csv, round 2dp; NaN → 422
            └─ (3) live U-Net inference [only if ENABLE_LIVE_INFERENCE=1]
Layer-2 metadata (mapping_method, n_cells, season mean) ← panchayat_summary.csv
Rules / severity / actions                              ← backend/advisory.py
Metric claims                                           ← outputs/metrics/*.json
Aux soil/NDVI/LULC                                      ← NOT in the chain
```

### 5.2 Traced examples (all for 2022-07-10, verified against source)

| Panchayat | id | index coord | nearest fine cell | **npz raw** | `/auth/` value | search mapping / n_cells |
|---|---|---|---|---:|---:|---|
| ALURU | 220447 | 13.679452, 74.742466 | (46, 67) | 50.487484 → | **50.49** | direct_grid / 2 |
| NAGULAMALLIAL | 201416 | 18.543199, 79.199244 | (143, 156) | — | **41.77** | direct_grid / 1 |
| AMERDA | 202120 | 17.804024, 79.034527 | (128, 153) | — | **27.68** | direct_grid / 4 |
| CURDI | 254406 | 15.260417, 74.235417 | (77, 57) | — | **16.73** | direct_grid / 1 |
| IDDAMPALLY | 207422 | 16.707303, 78.968539 | (106, 151) | — | **3.88** | area_weighted / 4 |
| NOOLPUZHA | 221933 | 11.744286, 76.272857 | (7, 97) | — | **1.58** | direct_grid / 12 |
| SULAJ | 172745 | 21.042647, 76.476471 | (193, 102) | — | **1.49** | area_weighted / 2 |
| AMRUTHALUR | 199960 | 15.105144, 78.913675 | (74, 150) | — | **0.13** | nearest_fallback / 1 |
| ATMAKUR | 195827 | 14.518174, 77.553099 | (62, 123) | ~0.0049 → | **0.0** | direct_grid / 3 |
| MANTAPAMPALLI | 198911 | 14.471816, 78.758989 | (61, 147) | — | **0.0** | area_weighted / 4 |
| MANDURIVARIPALEM | 234797 | 15.650000, 80.000000 | (85, 172) | **NaN** | **HTTP 422** | nearest_fallback / 1 |

Every value equals both the plan's prediction and an independent nearest-cell recomputation of `prediction_test.npz`. **Chain integrity: unbroken for all 11 cases.**

### 5.3 Advisory trace — "50.5 mm/day for ALURU"

| Step | Artefact | Observed |
|---|---|---|
| 1 | `/auth/` | `rainfall_mm=50.49`, `sources.rainfall="U-Net prediction grid (prediction_test.npz)"`, `model_status=ready` |
| 2 | `/api/advisory…rainfall_mm=50.49` | `severity="warning"`, `evidence.rainfall_mm=50.49`, `evidence.risk_level="very_heavy"`, `message_source="template"` |
| 3 | rule | `24.5 ≤ 50.49 < 64.5` → `R1B_SUBSTANTIAL_RAIN` → `medium` → UI `warning` |
| 4 | meaning | rainfall interpretation, evidence echo, severity — all consistent with §8.1 thresholds |

### 5.4 What cannot be traced (stated honestly)

- **Advisory → soil / NDVI / LULC:** no link exists; no backend route references those arrays (`INT-3`).
- **Advisory → per-date per-Panchayat Layer-2 value:** `panchayat_weather.csv` absent (D-29), so `/auth/` serves the raw grid cell.
- **Panchayat → its own polygon:** the served coordinate is an admin centroid (D-1); `location_precision` is effectively always `exact`.
- **`block_rainfall.csv`** (137,006 rows, valid) is not read by any route.

---

## 6. Agro-Advisory / LLM Validation

| Check | Result |
|---|---|
| Band mapping (70/50.49/30/12/2) | alert/very_heavy, warning/very_heavy, warning/heavy, info/moderate, watch/no_rain — **all correct** |
| `evidence.rainfall_mm` echoes input | exact (70.0, 50.49, 30.0, 12.0, 2.0) |
| Sensitivity | 5→70 changes severity (info→alert) and action count (2→3) |
| No hardcoding | different Panchayats → different values (UAT-J2) |
| Faithfulness guard | unit test green; with no key `message_source="template"`; LLM rewrites containing new numbers are rejected (`is_faithful`) |
| LLM invents input values? | not reachable without keys; `provider=rules` fallback verified, and XAI summary only cites trained channels |
| Material data change → insight change | yes (rainfall-driven) |
| Crop/stage echo | exact (`rice`/`flowering`) |
| Invalid crop/stage/lang | 422 |
| Combined block advisory | `POST /api/advisory/block` returns `[heat_stress, heavy_rain]` |
| `temperature_c` caveat from plan | confirmed: `GET /api/advisory` maps `temperature_c`→`tmean_c`, emits "only mean temperature available…" and reduces confidence — expected, not a bug |

No hallucination or unsupported-recommendation defect was found in the reachable (keyless) paths.

---

## 7. Critical Findings

### 7.1 Confirmed bugs (functional)

**BUG-01 — Ambiguous name-only lookup silently returns the wrong Panchayat — High.**
`POST /auth/ {"panchayat_name":"ALURU","date":"2022-07-10"}` returns **3.15 mm** (Guntur/PONNUR `ALURU`, pid 200620), not the Udupi `ALURU` (50.49). The answer text says "Estimated rainfall for ALURU" with no disambiguation. Root cause: `backend/data_store.py::Store.find()` returns `h.iloc[0]`. Verification: 4 rows named exactly `ALURU`; **8,404 duplicated names cover 26,666 rows**. Adding `district`/`block`/`state` resolves correctly (50.49). Evidence: `qa/qa_blackbox.json` BB-13. *Impact: a judge searching by name alone for a duplicated name gets another district's number.*
Repro: `curl -X POST :8000/auth/ -H 'Content-Type: application/json' -d '{"panchayat_name":"ALURU","date":"2022-07-10"}'`

**BUG-02 — `GET /` reports `panchayats_loaded: null` on a cold server — Medium.**
A freshly-booted process returns `null` until some other endpoint populates the store; a warm call returns `86103`. Root cause: the `register()` health handler reads the cached `State.store` instead of calling `store()`. Evidence: `qa/qa_blackbox.json` BB-01. The plan's `TC-03` expectation of `86103` is only true warm — the expectation, not the warm value, was wrong.

### 7.2 Data / deliverable issues

- **DATA-01 (Med):** `outputs/layer2/panchayat_weather.csv` absent although `README.md` §8 lists it → `/auth/` cannot serve a polygon-aggregated Layer-2 value.
- **DATA-02 (Low):** `outputs/layer2/panchayat_weather.geojson` absent although `README.md` §8 lists it → no per-Panchayat geometry.
- **DATA-03 (High, documented limitation):** coordinate collapse — 86,075 rows map to only **989 distinct coordinates**; **977/989 (98.8%) are block/district centres**; 76,327/86,075 rows (88.7%) sit on one; median coordinate shared by 57 Panchayats (worst 3,763). The "one value per Panchayat" claim is not supportable as-is; `/api/geocode` still reports `exact`.
- **DATA-04 (Med, packaging):** `data/processed/` in the git tree contains **only `meta.json`**; `X/Y/M_{train,val,test}.npy` are gitignored (present only in `data.zip`). A fresh clone cannot run `verify_dataset.py`/`generate_pred.py` without unzipping.

### 7.3 Model issues

- **I-3 (Med, scientific):** selected model D is worse than bilinear baseline A on **all** heavy-rain F1 thresholds (see §4.3). Disclose scope ("mean-error improvement", not "event-detection improvement").
- **I-4 (Low, documentation):** checkpoint `args.epochs=220` but `best_model_history.json` has **247** epoch rows (`best_epoch=207`). The configured epoch budget and the recorded history disagree; confirm which is authoritative.

### 7.4 Backend / integration issues

- **I-5 (Low):** two weather implementations exist. The backend `/api/weather` labels elevation `Open-Topo-Data (SRTM 90 m)`; the Next route `Frontend/src/app/api/weather/route.ts` labels it `Open-Topo-Data (SRTM 30m)` — inconsistent provenance for the same quantity. The Next route is reachable (200) but **unused** (`fetchWeather` never called — D-6 confirmed).
- **I-6 (Info):** `GET /api/advisory`'s `fired_rules` is top-level while `trace.rules` is nested — harmless, but the UI must read the right place.

### 7.5 Advisory / LLM issues

None found in reachable paths. Faithfulness guard, template fallback and rule bands all behave correctly.

### 7.6 Documentation inconsistencies

- **DOC-01 (Low):** `backend/README.md` says `cp .env.example .env`, but **no `backend/.env.example` exists** on b1.
- **DOC-02 (Low):** `README.md`/`HANDOVER.md` still describe the old `weather-downscaling-main/` layout though the app is at repo root.
- **DOC-03 (Info):** `HANDOVER.md` documents only the ML pipeline (no backend/frontend/pytest/Groq/Sarvam section).
- **DOC-04 (Low):** `frontend_backend_match.patch` cannot apply to b1 (context mismatch) though `backend/README.md` still instructs applying it.
- **DOC-05 (Med):** on **b1**, `scripts/ablation.py` and `scripts/evaluate.py` are **generator stubs** that write to `/mnt/data/*.py` (`ablation.py` is 535 lines vs the 200-line real version on `main`); they cannot be executed on this machine. The committed metrics are still valid (independently reproduced), but the b1 scripts are not runnable as-is.
- **DOC-06 (plan correction):** the baseline plan's NaN-count claim (§7.4), NDVI key name (`fine_values` vs `ndvi`), and cold-start `TC-03` expectation were wrong; corrected here.

### 7.7 Environment / setup

- `backend/.env.example` missing (DOC-01); keys optional — the app runs keyless with template/rules fallbacks.
- Layer-1 arrays only in `data.zip` (DATA-04).
- `.qa-b1/` worktree + venv created for isolation; the main working tree was not modified.

---

## 8. Regression Results (baseline plan vs current build)

| Item | Plan / baseline expectation | Current build result | Change |
|---|---|---|---|
| `pytest tests` | 37 passed | **37 passed** | unchanged |
| Health `/` warm | 86,103 | 86,103 | unchanged |
| Health `/` **cold** | (plan expected 86,103) | `null` | **plan expectation wrong** |
| ALURU projection | 50.49 | 50.49 | unchanged |
| All 10 Panchayat values | as cheat-sheet | all exact | unchanged |
| Advisory bands | alert…watch | exact | unchanged |
| `/api/metrics` | D 8.167 / 117,329 / +15.0% | exact | unchanged |
| Layer-1 metrics | 8.167 / 16.014 / 0.4461 / F1 set | **reproduced bit-for-bit** | **strengthened** |
| D-1 coordinate collapse | 989 coords / 86,075 rows | confirmed | unchanged |
| D-2 `.env.example` | absent | absent | unchanged |
| D-3 Layer-2 artefacts | absent | absent | unchanged |
| D-5 patch cannot apply | confirmed | confirmed | unchanged |
| D-6 dead Next weather route | dead | route works but **unused** | nuanced |
| D-11 NaN count | 3,007 | **1,556,354 = mask complement (correct)** | **plan error corrected** |
| D-41 NDVI | key `fine_values` | key `ndvi`, `(20,47250)` | **plan error corrected** |
| D-50 checkpoint | blocked | **passes** | **unblocked** |
| BB-13 ambiguous name | not in plan | **High bug found** | **new** |
| Frontend build | not run in baseline | **builds & serves** | new |

No genuine regression was observed against the plan baseline; the product matches or exceeds the documented state.

---

## 9. UAT Results

All seven realistic journeys completed:

| Journey | Result |
|---|---|
| Select ALURU → run projection → advisory → XAI | ✅ 50.49 mm, `warning`, XAI summary mentions 50.5 |
| Change Panchayat (ALURU → CURDI) | ✅ value changes 50.49 → 16.73 (not cached/default) |
| Change date | ✅ date-specific (2022-06-01=0.0, 2022-07-10=50.49) |
| Encounter unavailable data (AMBOLI) | ✅ clear 422, no fabricated location/value |
| Coastal/masked Panchayat | ✅ clear 422, never 0.0 |
| Out-of-range date | ✅ 422 naming the available range |
| Extreme rainfall advisory | ✅ `alert` + actionable steps |

Usability/correctness: outputs are understandable, location/date-correct, non-contradictory, and agricultural advice is consistent with the numeric inputs. The only UAT-affecting defect is BUG-01 (name-only ambiguous selection) — mitigated in practice because the UI search returns full rows (block/district/state) and posts them back.

---

## 10. Final Status

# **PASS WITH ISSUES**

**Why PASS:** the product's core promise — correct, traceable, non-fabricated Panchayat rainfall + agricultural advisory — is verified end-to-end; every numeric value is reproducible from committed artefacts; metrics were reproduced exactly; the unit suite, frontend build and all 14 UAT/E2E/integration checks pass.

**Why WITH ISSUES:** there are **2 genuine product defects** (1 High: ambiguous name-only lookup returns the wrong Panchayat; 1 Medium: cold-start health reports `null`), **5 confirmed data/documentation/deliverable gaps**, and **1 model-skill caveat** (heavy-rain F1 below baseline). Three "failures" in the baseline plan were found to be **plan errors**, not product faults, and are corrected here.

**Blocking-demo checklist (plan §14):** 9 of 10 blocking criteria pass. The single non-pass (TC-03 cold start) is a plan-expectation error; the warm value is correct. Not blocking, but must be disclosed: DATA-03 coordinate collapse, D-29 missing per-date Layer-2 CSV, INT-3 aux layers not wired.

**Blocked / not fully testable (with reasons):**

| Item | Reason |
|---|---|
| Groq live LLM wording quality (N-7) | no `GROQ_API_KEY`; keyless fallback verified instead |
| Sarvam live translation (N-8) | no `SARVAM_API_KEY`; endpoint returns a clear 502 and stays up |
| Full dataset rebuild (`verify_dataset.py`) | `data/processed/*.npy` not committed; available only inside `data.zip` |
| Live U-Net inference for arbitrary dates (N-4) | `ENABLE_LIVE_INFERENCE=0`; needs `data/processed` + `data/raw` — grid path tested instead |
| Per-date per-Panchayat Layer-2 value (N-3) | `panchayat_weather.csv` absent |
| External-API outage behaviour (TC-19 offline) | could not safely take the machine offline; intermittent `T=None/RH=None` with `error` string was observed and degrades correctly to nulls |

**Realism note:** no results were fabricated. Where a test could not run, it is marked blocked above with the exact reason.

---

## 11. Round-2 Deep Validation (value analysis + remaining untested paths)

### 11.1 Offline value analysis and remaining API paths

| Test ID | Category | Test | Expected | Actual | Status | Sev |
|---|---|---|---|---|---|---|
| WEATHER-REL | robustness | 10 consecutive `/api/weather` calls | all non-null ideally; else null+error, never fabricated | **10/10 non-null** (the intermittent null seen earlier is a transient external-API failure that degrades correctly) | **P** | Med |
| GEO-NOCOORD | geocode | coordinate-less Panchayat | 404 | 404 `no coordinates …` | **P** | Low |
| GEO-NOPARAM | geocode | missing `panchayat_name` | 422 | 422 | **P** | Low |
| SEARCH-AMBIG | mapping | `q=aluru` returns distinct rows for the 4 `ALURU`s | distinct ids/districts | 4 rows: `200620 Guntur, 217462 Davangere, 219239 Mandya, 220447 Udupi` | **P** | High |
| DUP-POP | mapping | duplicated-name blast radius | quantify | **8,404 names / 26,666 rows; max 59 rows share one name** | **i** | High |
| VAL-SEASONMEAN | value-analysis | `panchayat_summary.rainfall_mean_mm` vs npz cell mean (ALURU) | near-equal | `12.127` vs cell-mean `11.968` — differs because the summary is the **area-weighted polygon mean** (n_cells=2), the cell mean is one nearest cell | **i** | — |
| **VAL-MAP** | dataset | `outputs/maps/infer_2022-07-10.npy` matches the served Deccan grid | same grid | **stale 85×85 Western Ghats pilot grid** vs served 285×200 | **F** | Low |
| VAL-IMDCHIRPS | dataset | test-split `corr(IMD, CHIRPS)` vs `meta.imd_chirps_coarse_corr` | ≈0.3557 | `0.3850` on 5,397,646 masked pixels (same order; meta computed over a different set) | **i** | — |
| VAL-BLOCK | value-analysis | `block_rainfall` has 1,123 blocks every date | 1 unique count | `[1123]` for all 122 dates | **P** | Low |
| ADVBLOCK-LIMIT | advisory | block advisory cap | >60 rejected | 60→**200**, 61→**422**, 0→**422** | **P** | Low |
| GEO-PREC | mapping | geocode precision correctness | exact only when truly panchayat-level | **name-only `ALURU`→ lat 16.30287 (wrong ALURU, Guntur), labelled `exact`**; with context →13.679452 | **F** | High |
| ADVBLOCK-EMPTY | advisory | empty panchayat list | rejected | 422 (my first probe also used `stage=general`, which the block schema rejects — see below) | **i** | — |

### 11.2 Extended probes (environment variants, enums, missing files)

| Probe | Result | Status | Sev |
|---|---|---|---|
| Frontend CROP values via `GET /api/advisory` (`general, rice, wheat, cotton, maize, pulses`) | all **200** — frontend option set is compatible | **P** | — |
| Frontend STAGE values via `GET /api/advisory` (`general, sowing, vegetative, flowering, ripening, harvest`) | all **200** — compatible | **P** | — |
| `GET /api/advisory` crop `mustard`/`bajra` | **422** — these `CROP_PARAMS` exist in `advisory.py` but are unreachable via GET | **i** | Low |
| **`POST /api/advisory` & `/api/advisory/block` stage enum vs frontend** | POST accepts `grain_filling/maturity/harvest…` and **rejects the frontend's `general` and `ripening`** → **422** | **F** | Low |
| `ENABLE_LIVE_INFERENCE=1` with no torch/arrays | server starts and keeps serving from the grid; `live_inference:false`, `live_inference_error:null` (health does not surface the requested-but-failed init) | **P**(graceful) | Low |
| `REQUIRE_EXACT_LOCATION=1` | refusal path **unreachable** (no non-exact location exists — D-9) | **i** | Med |
| Missing data files (`REPO_ROOT` empty) | `/api/panchayats` & `/api/metrics` → **503** with actionable message; `/api/advisory` still 200 | **P** | — |
| `POST /api/advisory` single (valid body) | 200 | **P** | — |
| `POST /api/translate` without key | 502, clear error, server stays up | **P** | — |
| Name-only geocode wrong-coordinate (repeat) | confirms the ambiguity affects **both** `/api/geocode` and `/auth/` | **F** | High |

**New confirmed findings from round 2:**
- **BUG-01 extended:** the ambiguity is not limited to `/auth/` — **name-only `/api/geocode` also returns a different `ALURU`** (lat 16.30287) while still labelling it `exact`.
- **DATA-05 (Low):** a stale **Western Ghats pilot** artefact (`outputs/maps/infer_2022-07-10.npy`, 85×85) and pilot-era PNGs sit in the Deccan product repo. No route reads them, so no wrong value is served — but it is exactly the “old pilot dataset accidentally present” risk and should be removed/regenerated.
- **I-7 (Low):** `GET /api/advisory`, `POST /api/advisory` and `/api/advisory/block` each define a **different crop/stage enum**. Only GET is used by the frontend (and it matches), so there is no current user-facing break — but the divergence is a latent integration bug.
- **I-8 (Low):** `/` health reports `live_inference_error` only after a request actually tries the live path; with `ENABLE_LIVE_INFERENCE=1` on a torch-less host the endpoint still claims the error is `null`.

**Fail-safe behaviour is good:** missing files → 503 with a run instruction; out-of-domain/coastal/no-coord → 422; external APIs down → null + error string; no LLM keys → template/rules. Nothing fabricates a value.

---

## 12. Prioritised Fix List (lead-QA / lead-dev view)

### P0 — must fix before a judged demo

| # | Fix | Where | Why | Effort |
|---|---|---|---|---|
| **1** | **Disambiguate / refuse ambiguous lookups.** In `data_store.Store.find()`, do not `h.iloc[0]`. Require `district`+`block` context when >1 row matches a name; otherwise return **409/422** listing the candidates (or accept `panchayat_id`). Apply to **both** `/auth/` and `/api/geocode` (`locate_row`). | `backend/data_store.py`, `backend/routes_data.py` | BUG-01: 8,404 names (26,666 rows) silently resolve to the wrong Panchayat; geocode too. This is the one defect that can show a judge a wrong number. | S |
| **2** | **Cold-start health.** In the `/` handler call `store()` (or lazily initialise `S.store`) instead of reading `S.store`. | `backend/routes_data.py` (~L101) | BUG-02: fresh server reports `panchayats_loaded:null`. | XS |
| **3** | **Be honest about `location_precision`.** When the served coordinate came from `_admin_centres()` (a block/district centroid), return `location_precision` = `block`/`district` (not `exact`) so the existing “approximate” caveat actually fires; add a per-Panchayat flag to `/api/panchayats`. | `backend/data_store.py` | DATA-03: 88.7% of rows share an admin centroid but are labelled `exact`; the API already has caveat text that never triggers. | M |

### P1 — should fix for reliability / honesty

| # | Fix | Where | Why |
|---|---|---|---|
| 4 | **Real per-Panchayat geometry** (`scripts/fetch_lgd_panchayats.py` + `backend/build_panchayat_index.py`) to replace the 989 centroids with true polygon points, and regenerate `outputs/layer2/panchayat_weather.csv` (+ `.geojson`). | scripts + `outputs/layer2/` | Fixes the root of DATA-01/DATA-03 and enables a genuine per-date Layer-2 value. |
| 5 | **Unify the crop/stage enums** across `GET /api/advisory`, `POST /api/advisory`, `POST /api/advisory/block`, and expose `mustard`/`bajra` (I-7). | `backend/main.py`, `backend/advisory.py` | Removes a latent 422 for `general`/`ripening` on POST. |
| 6 | **Make `/` surface a live-inference failure** (`ENABLE_LIVE_INFERENCE=1` but torch/data missing) — call `runner()` in the health handler (I-8). | `backend/routes_data.py` | So ops can tell why live inference is off. |
| 7 | **Commit a `backend/.env.example`** and fix the README clone command (DOC-01). | `backend/` | Onboarding blocker. |
| 8 | **Remove/replace the stale pilot artefacts** (`outputs/maps/infer_2022-07-10.npy` 85×85 and pilot PNGs) — DATA-05. | `outputs/maps/` | Prevent accidentally rendering/serving the wrong region. |

### P2 — model / scientific

| # | Fix | Why |
|---|---|---|
| 9 | **Disclose the heavy-rain skill honestly.** Model D beats the bilinear baseline on MAE/RMSE/corr but is **worse on F1≥10/25/50** (≥50 mm F1 0.015 vs 0.234). Either report the caveat in the UI/metrics blurb, or train a variant that optimises event recall (the `weighted`/`log1p` losses already exist in `scripts/train.py`). | Scientific credibility; avoid over-claiming. |
| 10 | **Reconcile `args.epochs=220` with the 247-epoch history** (`best_epoch=207`) (I-4). | Reproducibility of the reported run. |

### P3 — packaging / docs

| # | Fix | Why |
|---|---|---|
| 11 | Fix b1's `scripts/ablation.py` / `evaluate.py` (they are `/mnt/data` generator stubs) so the ablation is re-runnable on b1 (DOC-05). | Reproducibility from the branch. |
| 12 | Commit or document how to obtain `data/processed/*.npy` (currently only inside `data.zip`) so `verify_dataset.py`/`generate_pred.py` run from a clone (DATA-04). | Verification/reproducibility. |
| 13 | Update `README.md`/`HANDOVER.md` for the repo-root layout + `panchayat_weather.*` claims; fix the `frontend_backend_match.patch` instruction (DOC-02/03/04). | Accuracy. |

### What is *already* correct (do not change)

Rule thresholds and bands; the faithfulness guard and template/rules fallbacks; masking (NaN exactly = non-target-valid cells); denormalisation (`rain_scale=100`); channel order; the parameter count and formula; the metric protocol; validation/error handling and fail-safe 503/422/null behaviour; the frontend build and the `/auth/`→advisory→XAI flow.

---

## 13. Appendix — Evidence Inventory

| File | Contents |
|---|---|
| `qa/qa_blackbox.py` / `qa/qa_blackbox.json` | 48 black-box API cases over live HTTP |
| `qa/qa_whitebox.py` / `qa/qa_whitebox.json` | 32 data/code white-box cases |
| `qa/qa_followup.py` / `qa/qa_followup.json` | 10 mask/aux/dataset cases (D-11 root cause) |
| `qa/qa_uat.py` / `qa/qa_uat.json` | 14 UAT + E2E + integration cases |
| `qa/qa_deep.py` / `qa/qa_deep.json` | 12 round-2 value-analysis / remaining-path cases |
| `qa/qa_model.py` | checkpoint load / channel order / param count |
| `qa/qa_repro_metrics.py` | Layer-1 metric reproduction from `data.zip` |
| `.qa-b1/uvicorn.log`, `.qa-b1/frontend.log` | server logs |
| `Frontend/` | npm ci / typecheck / lint / build / serve |

**Worktree note:** testing was performed in a detached `git worktree` at `.qa-b1/` (target commit `62d4a343…`). No application file was modified. The baseline plan `END_TO_END_QA_TEST_PLAN.md` was used read-only.

---

## 14. Post-fix status — QA hardening branch (`fix/qa-hardening`, commit `3395023`)

The defects and gaps in §7/§11 were actioned on worktree branch **`fix/qa-hardening`**
(base `62d4a343`). No application logic was changed to make a *test* pass; these are the fixes.

### 14.1 What was fixed

| # | Finding | Fix | Verification |
|---|---|---|---|
| BUG-01 | Name-only lookup returned another Panchayat | `Store.matches()`; `/auth/` and `/api/geocode` now return **409 + candidate list** for an ambiguous name; full context or `lat/lon` still resolves | `git grep`, live HTTP: bare `ALURU` → 409 with 4 candidates; `ALURU+KUNDAPURA+Udupi+KA` → 200 |
| DATA-03 | 989 shared block/district centroids for 86,075 rows | Rebuilt `panchayat_index.csv` from **LGD polygons** via `representative_point()` | **87,735 rows / 87,735 distinct coords**; coverage **86,103/86,103** mapped (was 86,075) |
| BUG-02 | Cold-start health returned `null` | `/` now loads the store itself | fresh process → `panchayats_loaded: 86103` |
| I-3 | Heavy-rain F1 far below baseline | Added variant **E** (heavy-rain-weighted loss, 5 channels); selection now tie-breaks on validation **heavy-rain F1** | see §14.2 |
| N-1 / aux gap | Layer 3 ignored soil/NDVI/LULC | New `backend/aux_layers.py` + rules **R6_VEGETATION, R7_SOIL_DRAINAGE, R8_LANDCOVER**; values exposed in `evidence.aux` | `GET /api/advisory?panchayat_id=220447…` fires R6+R8 with soil/NDVI/LULC in evidence |
| I-7 | GET vs POST advisory enums diverged; `mustard`/`bajra` unreachable | `Stage` now accepts frontend synonyms (`general`→None, `ripening`→`maturity`); `CropQ` adds `mustard`,`bajra` | `POST /api/advisory` accepts `general`/`ripening`; GET accepts `mustard` |
| I-8 | Health hid a failed live-inference init | `/` calls `runner()` | `ENABLE_LIVE_INFERENCE=1`, no torch → error surfaced |
| DOC-01 | No `.env.example`; `.gitignore` re-ignored it | Added `backend/.env.example`; removed the duplicate ignore block | `git check-ignore` → only the `!` negation matches |
| DOC-05 | b1's `scripts/ablation.py`/`evaluate.py` were `/mnt/data` generator stubs | Restored the real 200/347-line scripts | ablation re-runs on b1 and reproduces the committed metrics |
| DATA-05 | Stale 85×85 Western-Ghats pilot grid in `outputs/maps/` | Deleted `outputs/maps/infer_2022-07-10.npy` | file gone; no route read it |
| D-6 / I-5 | Duplicate Next.js `/api/weather` route + unused `fetchWeather` (different elevation-source label) | Removed the route and helper | `npm run typecheck` + `build` clean |
| D-7/DOC | README claimed uncommitted `panchayat_weather.csv`/`.geojson` | README now states they are **deliberately not committed** (>100 MB) and that per-date values are served from the grid at the Panchayat's own point | README/backend README updated |
| — | Tests for the new behaviour | Added `tests/test_mapping_and_aux.py` (10 cases) | **47 passed** (was 37) |

> **Correction to an earlier finding:** `panchayat_weather.csv` / `.geojson` are **not** an oversight —
> `.gitignore` deliberately excludes them as >100 MB (GitHub limit). The real gap was only the README
> implying they ship.

### 14.2 Model metrics: previous selected model D vs new selected E (2022 test split)

| Metric | D (reported before) | **E (now selected)** | Δ |
|---|---:|---:|---|
| MAE (mm) | 8.167 | 8.206 | ≈ unchanged |
| RMSE (mm) | 16.014 | **14.567** | −1.45 |
| Correlation | 0.446 | **0.518** | +0.072 |
| F1 ≥10 mm | 0.393 | **0.602** | +0.209 |
| F1 ≥25 mm | 0.281 | **0.425** | +0.144 |
| F1 ≥50 mm | 0.015 | **0.129** | ×8.6 |

E now beats the bilinear baseline (A) on MAE, RMSE, correlation and F1≥10/25 mm; **F1≥50 mm (0.129)
still trails baseline A (0.234)** — the hardest threshold and the remaining honest limitation.
Compute cost: ~90 epochs on CPU (~5.5 min), no GPU. The served grid
(`outputs/prediction_test.npz`) was regenerated from E (global max 163.28 → 175.51 mm).

### 14.3 Mapping fix: served values now reflect each Panchayat's own point (2022-07-10)

| Panchayat | old coord source | old mm | new mm |
|---|---|---:|---:|
| ALURU (Udupi) | block centre | 50.49 | **68.78** |
| NAGULAMALLIAL | block centre | 41.77 | **21.36** |
| AMERDA | block centre | 27.68 | **36.81** |
| CURDI | block centre | 16.73 | **49.83** |
| SULAJ | district centre | 1.49 | **25.72** |
| AMRUTHALUR | nearest_fallback | 0.13 | **NaN → 422** (its own point is outside the valid target mask) |

Values changed because each Panchayat is now sampled at its own polygon point — this is the fix,
not a regression. `AMRUTHALUR` now correctly refuses instead of returning a borrowed neighbour value.

### 14.4 Remaining known limitations (not fixed)

- **F1 ≥50 mm** still below the bilinear baseline (see §14.2).
- `data/processed/*.npy` and the LGD parquet are still not in git (size); they are needed to rebuild
  the index and retrain. `data.zip` carries the arrays.
- Groq/Sarvam live paths remain untested here (no keys); the fallbacks are verified.
- Historical pilot PNGs remain in `outputs/maps/` (figures only, not served).

### 14.5 Final status after fixes

**PASS** — the two product defects are fixed, the mapping is genuinely per-Panchayat, the weak
heavy-rain metrics are materially improved, and Layer 3 now uses the auxiliary layers. Remaining
items are the documented F1≥50 mm limitation and un-committed large data.

*Repository state at the time of writing: branch `fix/qa-hardening` @ `3395023`. Evidence: the
`qa/qa_*.json` files at the repository root, the local pre-fix backup `_backup_original/` (not committed —
gitignored), and the inline verification outputs in §14.1. The branch has since been pushed and the
findings re-checked — see the status note at the top.*

*End of report.*
