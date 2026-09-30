# END-TO-END QA TEST PLAN — Panchayat Rainfall Downscaling & Agro-Advisory

**Document type:** executable test plan (for a teammate who did **not** build the project)
**Target build:** branch **`b1`** of `https://github.com/yashkapur0403/weather-downscaling`
**Target commit (pin this):** `62d4a343911ca9ec8ddd5189c7cdafaf633a5a72`
**Author of this plan:** automated repository inspection of the `b1` tree (no tests were executed while writing it)
**Reference date of plan:** 2026-09-30

> **Status note (added at submission time).** This plan targets `b1` @ `62d4a34`. Its D-1 … D-9
> findings were re-checked against the hardened branch: **D-1** (centroid collapse), **D-2** (missing
> `backend/.env.example`), **D-5** (the patch cannot apply), **D-6** (dead duplicate `/api/weather`
> route) and **D-7** (docs describing the old `weather-downscaling-main/` layout) are **fixed**;
> **D-3** is now documented rather than fixed, because the >100 MB Layer-2 files are gitignored on
> purpose; **D-8** is open by design; **D-4** and **D-9** are open and low. The concrete changes and
> their evidence are in the status table at the top of `COMPREHENSIVE_QA_TEST_REPORT.md` and in
> `LOGICAL_VALIDATION_REPORT.md` §20.

> ⚠️ **Read this first — the branch matters.**
> The repository's **default branch `main` does NOT contain the backend or the frontend**.
> It holds only the Layer-1/Layer-2 ML pipeline. The complete product (FastAPI backend +
> Next.js frontend + tests) currently lives on the branch **`b1`**.
> If you clone and stay on `main`, almost every test below is unrunnable. Always check out `b1`.

---

## 0. How to use this document

### 0.1 Conventions

| Marker | Meaning |
|---|---|
| **GATE-n** | A blocking checkpoint. Do not continue past a failed gate; report it. |
| `☐ Pass ☐ Fail` | Fill in on your machine. Never leave a case unmarked. |
| `Actual result:` | Fill with the **exact** observed value/JSON, not "works". |
| `Evidence:` | File name you saved (see §0.3). |
| **[NOT TESTABLE]** | The current implementation does not expose this. Do not invent a procedure. |

### 0.2 Rules for the tester

1. **Do not modify application logic.** This plan is read-only against the codebase, except for creating `backend/.env` and saving evidence files.
2. **Do not "fix" a failure while testing it.** Record actual vs expected, mark Fail, continue.
3. When a case depends on the network (Open-Meteo, Open-Topo-Data, Groq, Sarvam), record the network state in `Notes:`.
4. Copy every value verbatim. A rounded value where the plan expects 2 decimals is a **Fail** — it usually means a fallback path was taken.
5. Estimated hands-on time: **2.5–4 hours** for the full plan on a fresh laptop.

### 0.3 Evidence

Create a folder next to the repo: `qa-evidence/`. One file per case, e.g.:
`qa-evidence/TC-07-aluru-projection.json`, `qa-evidence/TC-20-ui-flow.png`.
For API cases save the **raw JSON body**; for UI cases save a screenshot showing the selected Panchayat, the rainfall number, and the advisory text together.

---

## 1. What you are actually testing (verified build inventory)

### 1.1 Architecture (as implemented on `b1`)

```
                 ┌────────────────────────── Next.js frontend (Frontend/, :3000) ──────────────────────────┐
                 │  search box → POST /auth/ → advisory panel → XAI panel                                   │
                 └───────────────┬──────────────────────────────────────────────────────────────────────────┘
                                 │  HTTP, NEXT_PUBLIC_API_URL (default http://localhost:8000)
                 ┌───────────────▼──────────────── FastAPI backend (backend/, :8000) ───────────────────────┐
                 │  routes_data.py   : /  /api/panchayats  /api/geocode  /auth/  /api/metrics  /api/weather │
                 │  main.py          : owns the app + CORS; /api/advisory  /api/explain  /api/translate     │
                 │  advisory.py      : THE rule engine (Layer 3) — decides severity/action                  │
                 │  data_store.py    : reads the CSV/NPZ artefacts (never invents a value)                  │
                 │  weather_lookup.py: Open-Meteo (T/RH) + SRTM DEM (elevation)                            │
                 │  live_infer.py    : optional real U-Net inference (OFF unless ENABLE_LIVE_INFERENCE=1)    │
                 │  metrics_loader.py: real numbers from outputs/metrics/*.json                             │
                 └───────────────┬──────────────────────────────────────────────────────────────────────────┘
                                 │
      outputs/prediction_test.npz │ (U-Net fine grid, 122 days × 285 × 200 @ 0.05°)
      outputs/layer2/*.csv        │ (panchayat_summary, panchayat_index, block_rainfall)
      data/aux_data/*.npz         │ (admin IS read for block/district centres; soil/NDVI/LULC are NOT — §13)
      models/best_model.pt        │ (used only by live inference / metrics param count)
```

### 1.2 API routes → where each value really comes from

| Route | Method | Value it returns | Real backer (verified) |
|---|---|---|---|
| `/` | GET | health + `panchayats_loaded` | `outputs/layer2/panchayat_summary.csv` |
| `/health` | GET | `{status, groq_models}` | process state |
| `/api/panchayats?q=&limit=` | GET | Panchayat search rows | `panchayat_summary.csv` (+ `panchayat_index.csv` for coords) |
| `/api/geocode?panchayat_name=…` | GET | `{lat, lon, location_precision}` | `panchayat_index.csv`, else district/block centre from `grid_admin_map_deccan.npz` |
| `/api/weather?lat=&lon=&date=` | GET | T / RH / elevation + source labels | Open-Meteo archive; SRTM DEM local if present, else Open-Topo-Data |
| `/auth/` | POST | `prediction.rainfall_mm` + context | `panchayat_weather.csv` **if present**, else `prediction_test.npz`, else live inference |
| `/api/metrics` | GET | model comparison metrics | `outputs/metrics/ablation.json` + `metrics.json` + `models/best_model_history.json` |
| `/api/advisory` | GET | advisory text/severity/actions/evidence | `advisory.py` rule engine (**all inputs are query parameters**) |
| `/api/advisory` | POST | full audit trace | `advisory.py` + optional Groq rephrase + Sarvam translate |
| `/api/advisory/block` | POST | batch advisory | `advisory.py` per panchayat (max 60) |
| `/api/explain`, `/api/explain/status` | POST/GET | XAI explanation | deterministic `ui_facts()` in `main.py` + optional Groq |
| `/api/translate` | POST | translated text | Sarvam |

### 1.3 Committed data artefacts — verified statistics

| File | Size / shape | Verified content |
|---|---|---|
| `outputs/prediction_test.npz` | 122 × 285 × 200 float32 | dates `2022-06-01` … `2022-09-30`; lat `11.4`→`25.6`; lon `71.4`→`81.35`; rainfall min/mean/max = `0.0 / 3.93 / 163.28`; **NaN where the land mask M=0 (sea/excluded)** |
| `outputs/layer2/panchayat_summary.csv` | 87,735 rows | mapping: `area_weighted` 52,893 · `direct_grid` 31,740 · `unmapped` 1,632 · `nearest_fallback` 1,470 |
| `outputs/layer2/panchayat_index.csv` | 86,075 rows | `panchayat_id,lat,lon` — **only 989 distinct coordinates for 86,075 Panchayats**; **977 of those 989 (98.8%) are exactly a block/district centre** derived from `grid_admin_map_deccan.npz`, and **76,327 of 86,075 rows (88.7%)** sit on such a centre (see §15 D-1) |
| `outputs/layer2/block_rainfall.csv` | 137,006 rows | `date,state,district,block,subdistrict_idx,rainfall_mm` |
| `outputs/layer2/layer2_qc.json` | — | Layer-1 source `outputs/prediction_test.npz`; 87,735 total / 86,103 mapped (98.1%) / 1,632 unmapped; rainfall 0–163.28 |
| `outputs/metrics/ablation.json` | — | selected row **D** ("U-Net + DEM + ERA5-Land"); test MAE A=`9.603`, D=`8.167` |
| `data/aux_data/{admin,soil_soilgrids,ndvi_monthly,lulc_fractions}_deccan.npz` | committed | **not read by any backend route** (§13) |

**Backend-visible Panchayat count:** the store drops `unmapped` rows → `87,735 − 1,632 = **86,103**`. This number must appear in the health response.

---

## 2. GATE-1 — Fresh-clone setup on a different laptop

### 2.1 Prerequisites

| Requirement | Notes |
|---|---|
| Git, Python 3.11–3.13, Node 20+ | `backend/Dockerfile` uses `python:3.11-slim`; backend deps are pure-Python wheels |
| ~1 GB free disk | repo + `node_modules` |
| Internet | needed for Open-Meteo / Open-Topo-Data, and for Groq/Sarvam if testing those |
| Groq API key + Sarvam API key | **optional**. Without them the product still runs (advisory falls back to a deterministic template; explain falls back to `provider: "rules"`) |

### 2.2 Clone and pin the build

```bash
git clone https://github.com/yashkapur0403/weather-downscaling.git
cd weather-downscaling

# IMPORTANT: the app is NOT on main.
git checkout b1
git checkout 62d4a343911ca9ec8ddd5189c7cdafaf633a5a72   # pin so evidence is reproducible
git rev-parse HEAD        # must print 62d4a343911ca9ec8ddd5189c7cdafaf633a5a72
git branch --show-current # must print b1 (or HEAD detached at b1)
```

**Expected tree at the repo root** (verify with `ls`):
`Frontend/  backend/  data/  docs/  generate_pred.py  legacy/  models/  outputs/  qa/  README.md  requirements.txt  scripts/`

> As of the QA-hardening merge, `main` carries the app too, so `backend/` and `Frontend/` at the repo
> root are the real ones. If you see no `backend/` at all, you are on an older branch.
> `legacy/old-app-snapshot/` (formerly `weather-downscaling-main/`) is the superseded archive kept only for the unique data it
> holds — it is not a second application.

### 2.3 Backend install

```bash
cd backend
python -m venv .venv
# Windows Git Bash:
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
# macOS/Linux:
# source .venv/bin/activate && pip install -r requirements.txt
```

### 2.4 Create the environment file (manual — there is no template)

`backend/README.md` tells you to run `cp .env.example .env`, **but `backend/.env.example` does not exist in the `b1` tree** (see §15 D-2). Create it by hand:

```bash
cd backend
cat > .env <<'EOF'
# Leave the keys empty if you are testing the no-LLM fallback path.
GROQ_API_KEY=
SARVAM_API_KEY=
# Defaults below are read by main.py / routes_data.py:
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
ENABLE_LIVE_INFERENCE=0
REQUIRE_EXACT_LOCATION=0
EOF
```

Keep this file: several cases change one variable and restart the API.

### 2.5 Frontend install

```bash
cd ../Frontend
npm install            # or: npm ci  (package-lock.json is committed)
cat > .env.local <<'EOF'
NEXT_PUBLIC_API_URL=http://localhost:8000
EOF
```

> `frontend_backend_match.patch` exists at the repo root and `backend/README.md` tells you to apply it.
> **Do not apply it on `b1` blindly** — `Frontend/src/api/backend.ts` on `b1` already sends the new
> `/api/advisory` parameters. Confirm before applying:
> ```bash
> cd ..
> git apply --check --directory=Frontend frontend_backend_match.patch
> ```
> If it reports errors, skip it (expected: the fixes are already in the tree by different edits).

### 2.6 GATE-1 checklist

| # | Check | Expected |
|---|---|---|
| 1 | `git rev-parse HEAD` | `62d4a343…` |
| 2 | `backend/`, `Frontend/` exist at root | yes |
| 3 | backend `pip install` | exits 0 |
| 4 | `Frontend/npm install` | exits 0, `node_modules/` created |
| 5 | `backend/.env` exists | created manually in §2.4 |

---

## 3. GATE-2 — Backend startup and health

### 3.1 Start

```bash
cd backend
.venv/Scripts/python.exe -m uvicorn app:app --reload --port 8000
```

Expected startup log: a warning `GROQ_API_KEY or SARVAM_API_KEY is missing` when keys are empty, then `Uvicorn running on http://127.0.0.1:8000`.
(`app:app` → `backend/app.py` → imports `main.app` and calls `routes_data.register(app)`.)

### 3.2 Health checks

```bash
curl -s http://localhost:8000/ | python -m json.tool
curl -s http://localhost:8000/health
```

Expected `/`:

```json
{
  "message": "Panchayat downscaling API",
  "repo_root": "<absolute path to your clone>",
  "panchayats_loaded": 86103,
  "live_inference": false,
  "live_inference_error": null
}
```

Expected `/health`: `{"status":"ok","groq_models":[...]}` (two models by default).

| # | Check | Expected |
|---|---|---|
| 1 | `GET /` | HTTP 200, `panchayats_loaded == 86103` |
| 2 | `GET /health` | HTTP 200, `status == "ok"` |
| 3 | CORS | a browser request from `http://localhost:3000` is not blocked |

> **Gate failure meaning:** `panchayats_loaded` of `null` ⇒ `outputs/layer2/panchayat_summary.csv` missing ⇒ you are on the wrong branch/commit. `live_inference_error` non-null ⇒ harmless (torch/data absent).

---

## 4. GATE-3 — Frontend startup

```bash
cd Frontend
npm run dev
# → http://localhost:3000
```

Optional static checks: `npm run typecheck`, `npm run lint`, `npm run build`.

Open the dashboard and confirm the map renders and the search box is present. If the search box shows `Backend not connected`, the API is not on `http://localhost:8000` (check `NEXT_PUBLIC_API_URL`).

| # | Check | Expected |
|---|---|---|
| 1 | page loads at `:3000` | no console error loop |
| 2 | `npm run typecheck` | exits 0 |
| 3 | search box present | yes |
| 4 | Leaflet map tiles render | yes (needs internet) |

---

## 5. Automated test baseline (`backend/tests/`)

```bash
cd backend
.venv/Scripts/python.exe -m pytest -q tests
```

Expected: **37 passed** (files `test_advisory.py`, `test_routes_data.py`, `test_frontend_contract.py`). These need **no API keys and no network** (Groq/Sarvam are monkeypatched; the data-repo tests build a synthetic `tmp_path`).

**What these 37 tests already cover** (do not re-derive by hand, just confirm green):
rule-engine logic (heavy rain, dual-condition irrigation, heat stage gating, non-evaluable rules, margin/flip hints), the downscaling-residual explanation, the LLM faithfulness guard, advisory API contract + severity mapping + 422 on bad input, explain contract + hallucinated-number rejection, routes validation, Layer-2-CSV priority over the grid, "no data is an error, not a fake number", and the parameter-count formula.

**What they do NOT cover** (this is why this plan exists):
the **real** committed data (`prediction_test.npz` / `panchayat_summary.csv` / `panchayat_index.csv`), the real coordinate collapse, real Land-Mask NaN cells, live Open-Meteo/DEM values, the browser end-to-end flow, latency, and the auxiliary soil/NDVI/LULC layers.

| # | Check | Expected |
|---|---|---|
| 1 | `pytest -q tests` | `37 passed` |
| 2 | no network/keys required | confirmed |

---

## 6. The end-to-end user flow (manual, with exact API calls)

Work through **A → E** once with the UI, and repeat each step with `curl` so values can be compared byte-for-byte. Use the Panchayat **ALURU / KUNDAPURA / Udupi / KARNATAKA**.

### Step A — Select location (search + Layer-2 mapping + coordinates)

UI: type `kundapura` in the search box.

```bash
curl -s "http://localhost:8000/api/panchayats?q=kundapura&limit=5" | python -m json.tool
curl -s "http://localhost:8000/api/geocode?panchayat_name=ALURU&block_name=KUNDAPURA&district=Udupi&state=KARNATAKA" | python -m json.tool
```

Expected: rows carrying the pipeline's own `mapping_method`, `n_cells`, `rainfall_basis: "season_mean_2022"`, `date: "2022-07-10"`, and `location_precision: "exact"` for the geocode. See TC-05/TC-06.

### Step B — Retrieve Layer-1 rainfall (projection)

UI: click **Run Projection →**.

```bash
curl -s -X POST http://localhost:8000/auth/ -H "Content-Type: application/json" -d '{
  "panchayat_name":"ALURU","block_name":"KUNDAPURA","district":"Udupi","state":"KARNATAKA",
  "date":"2022-07-10","requested_metrics":["rainfall","temperature","humidity","elevation"],
  "raw_location_text":"ALURU, Udupi"}' | python -m json.tool
```

Expected `prediction.rainfall_mm = 50.49`, `sources.rainfall = "U-Net prediction grid (prediction_test.npz)"`, `model_status = "ready"`, `sources.location_precision = "exact"`. See TC-07 and §12 traceability.

### Step C — Layer-2 Panchayat mapping / aggregation

Observe in the search response and in the UI selection summary that `mapping_method` and `n_cells` are the **real** join outputs (`direct_grid` / `area_weighted` / `nearest_fallback`), not constants. See TC-05 and TC-11.

> **[NOT TESTABLE — per-date per-Panchayat Layer-2 value]** `outputs/layer2/panchayat_weather.csv` is **not present on `b1`**, so the backend cannot serve a per-date Layer-2 Panchayat value and `/auth/` falls back to the fine grid. Producing that file requires the LGD parquet (`python scripts/fetch_lgd_panchayats.py` + `scripts/layer2_panchayat_mapping.py`) — see §13.

### Step D — Auxiliary values (SoilGrids / NDVI / LULC) and risk/conditions

**[NOT TESTABLE in the running product]** No backend route reads `data/aux_data/soil_soilgrids_deccan.npz`, `ndvi_monthly_deccan.npz` or `lulc_fractions_deccan.npz`. The advisory engine consumes only `rainfall_mm`, `tmean_c`/`tmax_c`, `humidity_pct`, `wind_kmh`, `soil_moisture` — and `soil_moisture` is only ever supplied by the API caller, never read from the aux files.
Offline **data-integrity** checks on those npz files ARE testable — see TC-22.

What IS testable for "risk/conditions": the deterministic classification bands (§8) and the advisory severity.

### Step E — Final agro-advisory

UI: the advisory panel loads automatically once a projection has run.

```bash
curl -s "http://localhost:8000/api/advisory?panchayat_id=220447&crop=wheat&stage=general&rainfall_mm=50.49&temperature_c=26.0&humidity_pct=80&date=2022-07-10&panchayat_name=ALURU" | python -m json.tool
```

Expected: `severity: "warning"` (50.49 lies in `[24.5, 64.5)`), `evidence.rainfall_mm == 50.49`, `evidence.risk_level == "very_heavy"`, `crop == "wheat"`, `stage == "general"`. See TC-14/TC-15.

---

## 7. Data-correctness cross-checks (the core of this plan)

### 7.1 The rule being verified

> Every number the backend returns must be **traceable to a committed file**. A value that is identical for every Panchayat, or identical for every date, is a defect (or a documented fallback).

### 7.2 Reproduction snippet — verify a projection against the source NPZ

Run this from the **repo root** with the backend's venv. It re-computes exactly what `data_store.Store.grid_rainfall()` does (nearest fine cell, `round(v, 2)`).

```python
import numpy as np, pandas as pd
z = np.load("outputs/prediction_test.npz")
rain, la, lo, dates = z["rainfall_mm"], z["latitude"], z["longitude"], [str(d) for d in z["dates"]]
idx = pd.read_csv("outputs/layer2/panchayat_index.csv")
coord = {int(r.panchayat_id): (r.lat, r.lon) for r in idx.itertuples()}

def expected(pid, date):
    L, O = coord[pid]
    i = int(np.abs(la - L).argmin()); j = int(np.abs(lo - O).argmin())
    v = rain[dates.index(date), i, j]
    return ("NaN -> HTTP 422" if not np.isfinite(v) else round(float(v), 2)), i, j

print(expected(220447, "2022-07-10"))   # ('ALURU')          -> (50.49, 46, 67)
```

Verified values for the plan's test Panchayats (all for **2022-07-10**):

| Panchayat | id | Coordinate (lat, lon) | fine (i,j) | Expected `rainfall_mm` |
|---|---|---|---|---|
| ALURU (Kundapura, Udupi, KA) | 220447 | 13.679452, 74.742466 | 46, 67 | **50.49** |
| NAGULAMALLIAL (Kothapally, Karimnagar, TG) | 201416 | 18.543199, 79.199244 | 143, 156 | **41.77** |
| AMERDA (Aswapuram, Bhadradri Kothagudem, TG) | 202120 | 17.804024, 79.034527 | 128, 153 | **27.68** |
| CURDI (Sanguem, South Goa, GA) | 254406 | 15.260417, 74.235417 | 77, 57 | **16.73** |
| IDDAMPALLY (Devarakonda, Nalgonda, TG) | 207422 | 16.707303, 78.968539 | 106, 151 | **3.88** |
| NOOLPUZHA (Sulthan Bathery, Wayanad, KL) | 221933 | 11.744286, 76.272857 | 7, 97 | **1.58** |
| SULAJ (Jalgaon Jamod, Buldhana, MH) | 172745 | 21.042647, 76.476471 | 193, 102 | **1.49** |
| AMRUTHALUR (Amruthalur, Bapatla, AP) | 199960 | 15.105144, 78.913675 | 74, 150 | **0.13** |
| ATMAKUR (Atmakur, Anantapur, AP) | 195827 | 14.518174, 77.553099 | 62, 123 | **0.0** |
| MANTAPAMPALLI (Vontimitta, Y.S.R., AP) | 198911 | 14.471816, 78.758989 | 61, 147 | **0.0** |
| MANDURIVARIPALEM (Ongole, Prakasam, AP) | 234797 | 15.650000, 80.000000 | 85, 172 | **NaN → HTTP 422** |

### 7.3 Reproduction snippet — verify the metrics endpoint

```python
import json
ab = json.load(open("outputs/metrics/ablation.json"))
print(ab["selection"])                       # {'criterion': 'val MAE', 'row': 'D', ...}
print(ab["results"]["test"]["A"]["MAE"])     # 9.602921485900879
print(ab["results"]["test"]["D"]["MAE"])     # 8.167057991027832
```

Expected `/api/metrics`: `selected_model.test_mae_mm = 8.167`, `test_rmse_mm = 16.014`, `test_correlation = 0.4461`, `baseline.test_mae_mm = 9.603`, `mae_improvement_pct = 15.0`, `n_parameters = "117,329"`, `evaluation_period = "2022-06-01 to 2022-09-30 (monsoon test season)"`, `variant_key = "D"`.

### 7.4 Units, dates and spatial rules

| Quantity | Unit / rule | Verified in |
|---|---|---|
| `rainfall_mm` | mm/day, 2 decimals | `prediction_test.npz`, `round(v,2)` in `sample_grid` |
| `rainfall_mean_mm` (search) | mm/day, **season mean of the 2022 monsoon**, labelled `rainfall_basis: "season_mean_2022"` | `panchayat_summary.csv` |
| `temperature_c` | °C, 1 decimal | Open-Meteo |
| `humidity_pct` | %, integer | Open-Meteo |
| `elevation_m` | m, 1 decimal | SRTM DEM / Open-Topo-Data |
| Default date | `2022-07-10` when the request omits `date` | `data_store.DEFAULT_DATE` |
| Valid date range | `2022-06-01` … `2022-09-30` (122 days) | `prediction_test.npz["dates"]` |
| Fine grid | 0.05°, 285 × 200, lat 11.4→25.6, lon 71.4→81.35 | `prediction_test.npz` |
| `/auth/` sampling | nearest fine cell to the Panchayat coordinate; outside the grid → **422** | `live_infer.sample_grid` |
| Masked cells | M=0 (sea/coastal-excluded) are **NaN** → sampling them raises → **422** | `generate_pred.py` writes NaN where M≠1 |

**Date traps to test:** `"2022-07-10"` (default), `"2022-06-01"` (first day), `"2022-09-30"` (last day), `"2021-01-01"` (before range → 422), `"2022-10-01"` (after range → 422), and an empty/missing date (→ default).

**Spatial traps to test:** duplicate coordinates (§15 D-1), coordinates outside the grid, `nearest_fallback` rows, and the fact that `location_precision: "exact"` is **nominal** — the coordinate it returns is usually a block/district centre shared with thousands of other Panchayats (§15 D-1/D-9).

---

## 8. Layer-3 advisory correctness (rule engine)

### 8.1 The rules, exactly as implemented in `backend/advisory.py`

| Rule | Fires when | Severity | UI label |
|---|---|---|---|
| `R1_HEAVY_RAIN` | `rainfall_mm >= 64.5` | high | **alert** |
| `R1B_SUBSTANTIAL_RAIN` | `24.5 <= rainfall_mm < 64.5` | medium | **warning** |
| `R2_IRRIGATION` | rain-only fallback: `rainfall_mm < 3.0 × window_days` → severity *low*; with soil moisture: also `soil_moisture < crop.dry_sm` → severity *medium* | low/medium | watch/warning |
| `R3_HEAT_STRESS` | `temperature >= crop.heat` **and** stage is heat-sensitive (or stage unknown) | medium/high | warning/alert |
| `R4_DISEASE` | `humidity >= 85` **and** `22 <= temperature <= 32` | medium | warning |
| `R5_LODGING` | `wind_kmh >= 40` **and** tall crop | medium | warning |

Crop heat thresholds: general 36 · wheat 34 · rice 35 · maize 35 · mustard 32 · cotton 38 · bajra 40 · pulses 35.
Severity→UI map: `none→info`, `low→watch`, `medium→warning`, `high→alert`.
Rainfall band labels (`_risk_level`): `<2.5 no_rain`, `<10 light`, `<25 moderate`, `<50 heavy`, else `very_heavy`.

### 8.2 What to verify

1. **Consistency with numeric inputs** — `evidence.rainfall_mm` must equal the `rainfall_mm` you sent, and `severity` must match the band in §8.1.
2. **No invention** — the advisory must not contain numbers absent from the deterministic trace. `main.farmer_message()` rejects an LLM rewrite that introduces new numbers and silently falls back to `message_source: "template"`.
3. **Sensitivity** — changing only `rainfall_mm` across a threshold must change `severity`/`actions`.
4. **Not hardcoded** — two different Panchayats with different rainfall must not produce the same `evidence.rainfall_mm`.

> **Caveat to record in `Notes:`** `GET /api/advisory` passes `temperature_c` into `tmean_c` (not `tmax_c`), so the trace includes `"only mean temperature available; heat stress may be under-detected"` and `confidence` is reduced by 0.1. This is expected, not a bug.

---

## 9. API contract, validation, error handling and latency

### 9.1 Validation matrix

| Request | Expected |
|---|---|
| `GET /api/panchayats` (no `q`) | **422** (`q` has `min_length=1`) |
| `GET /api/panchayats?q=kundapura&limit=100` | **422** (`limit ≤ 50`) |
| `GET /api/panchayats?q=kundapura` | **200**, JSON array |
| `GET /api/geocode?panchayat_name=NOPE` | **404** `"panchayat not found"` |
| `GET /api/geocode` for a Panchayat with no coordinates | **404** `"no coordinates for this panchayat…"` |
| `POST /auth/` with `date:"2021-01-01"` | **422**, message lists the available range |
| `POST /auth/` with `lat:40, lon:10` (outside domain) | **422** `…is outside the model domain` |
| `POST /auth/` for a Panchayat with no coordinates | **422** containing `coordinates` |
| `GET /api/advisory` **without** `rainfall_mm` | **422** (required param) |
| `GET /api/advisory?crop=banana` | **422** (not in the enum) → the frontend then uses its local fallback |
| `GET /api/advisory?lang=xx-XX` | **422** `unsupported lang` |
| `POST /api/explain` with `weights` outside 0–1 | **422** (pydantic) |
| `GET /api/metrics` | **200** (503 if `ablation.json` missing) |

Empty-body / malformed JSON on `POST /auth/` must be a 4xx, never a 500.

### 9.2 Latency budget (record actual ms in `Notes:`)

| Request | Target (local, CPU) |
|---|---|
| `GET /` | < 50 ms |
| `GET /api/panchayats` | < 300 ms first call, then < 150 ms |
| `POST /auth/` (grid path) | < 1.5 s incl. Open-Meteo + elevation |
| `GET /api/advisory` (no keys → template) | < 500 ms |
| `GET /api/advisory` (Groq enabled) | < 8 s typical |
| `GET /api/metrics` | < 200 ms |
| `POST /api/explain` (no key → rules) | < 300 ms |

> NOTE: no explicit per-route instrumentation exists in the code — measure wall-clock with `curl -w '%{time_total}\n'`.

---

## 10. Frontend ↔ backend consistency

The frontend must display **exactly** what the API returned for the same selection.

| UI element | Must equal | Source of truth |
|---|---|---|
| search dropdown rainfall | `rainfall_mm` from `/api/panchayats` (season mean) | `panchayat_summary.csv` |
| selection summary "Rainfall" after projection | `prediction.rainfall_mm` from `/auth/` | `prediction_test.npz` |
| selection "Mapping" / "Grid Cells" | `mapping_method` / `n_cells` from `/api/panchayats` | `panchayat_summary.csv` |
| risk badge | `classifyRisk(rainfall_mm)` — same bands as backend `_risk_level` | frontend `types/index.ts` |
| advisory text / severity / actions / `evidence` | `/api/advisory` response | `advisory.py` |
| evidence footer mm / temp / RH / date | `evidence.rainfall_mm` / `temperature_c` / `humidity_pct` / `data_date` | `/api/advisory` |

**Known duplicate code path to check:** `Frontend/src/app/api/weather/route.ts` is a Next.js route handler that proxies Open-Meteo/Open-Topo-Data. The frontend actually calls the **backend** `/api/weather` (absolute `NEXT_PUBLIC_API_URL`). Confirm at runtime which one answers (Network tab) and note whether the Next route is reachable/dead — do not silently assume it works.

---

## 11. Representative Panchayat test cases

Each case below uses one field per required column. Fill `Actual result`, `Pass/Fail`, `Evidence`, `Notes`.

> **Preconditions for every TC-** case: backend running (§3), `backend/.env` with empty keys unless stated, working directory = repo root.

---

### TC-01 — Fresh clone installs and pins the correct build
* **Objective:** prove another laptop can reproduce the build.
* **Steps:** follow §2.1–§2.5; then
  ```bash
  git rev-parse HEAD
  ls backend Frontend scripts outputs/layer2
  ```
* **Expected result:** HEAD = `62d4a343911ca9ec8ddd5189c7cdafaf633a5a72`; `backend/` and `Frontend/` exist at the root; `outputs/layer2/` contains `panchayat_summary.csv`, `panchayat_index.csv`, `block_rainfall.csv`, `layer2_qc.json`.
* **Actual result:** ______
* **Pass/Fail:** ☐ Pass ☐ Fail
* **Evidence:** ______ (paste `git rev-parse HEAD` + `ls` output)
* **Notes:** ______

### TC-02 — Automated backend test suite is green
* **Objective:** baseline regression suite passes with no keys/network.
* **Steps:** `cd backend && .venv/Scripts/python.exe -m pytest -q tests`
* **Expected result:** `37 passed` (0 failed, 0 errors).
* **Actual result:** ______  **Pass/Fail:** ☐ ☐  **Evidence:** ______  **Notes:** ______

### TC-03 — Backend health reports the real Panchayat count
* **Objective:** the service loaded the committed Layer-2 summary.
* **Steps:** `curl -s http://localhost:8000/ | python -m json.tool`
* **Expected result:** `panchayats_loaded == 86103`; `message == "Panchayat downscaling API"`; `repo_root` = your clone path.
* **Actual result:** ______  **Pass/Fail:** ☐ ☐  **Evidence:** ______  **Notes:** ______

### TC-04 — `/health` returns process state
* **Objective:** confirm the service-level health probe reports a usable state without touching the data files.
* **Steps:** `curl -s http://localhost:8000/health`
* **Expected result:** `status == "ok"`, `groq_models` is a non-empty list.
* **Actual result:** ______  **Pass/Fail:** ☐ ☐  **Evidence:** ______  **Notes:** ______

### TC-05 — Search returns real Layer-2 mapping fields (not hardcoded)
* **Objective:** prove `mapping_method` / `n_cells` come from the data file.
* **Steps:**
  ```bash
  curl -s "http://localhost:8000/api/panchayats?q=kundapura&limit=5" | python -m json.tool
  grep -m3 -i kundapura outputs/layer2/panchayat_summary.csv
  ```
* **Expected result:** API rows match the CSV rows (same `panchayat_id`, `mapping_method`, `n_cells`, `rainfall_mean_mm`); `rainfall_basis == "season_mean_2022"`; validation: searching `ALLAMUDI` (an `unmapped` row) returns `[]`.
* **Actual result:** ______  **Pass/Fail:** ☐ ☐  **Evidence:** ______  **Notes:** ______

### TC-06 — Geocode returns coordinates and their precision
* **Objective:** confirm the Panchayat resolves to a real stored coordinate and that the endpoint declares how precise that location is.
* **Steps:**
  ```bash
  curl -s "http://localhost:8000/api/geocode?panchayat_name=ALURU&block_name=KUNDAPURA&district=Udupi&state=KARNATAKA"
  ```
* **Expected result:** `{"lat":13.679452,"lon":74.742466,"location_precision":"exact"}`.
  **Record in `Notes:`** that this coordinate is *identical* to the mean centre of all fine-grid cells in the KUNDAPURA block (verified), i.e. `"exact"` means "present in `panchayat_index.csv`", not "a real per-Panchayat polygon point" (D-1).
* **Actual result:** ______  **Pass/Fail:** ☐ ☐  **Evidence:** ______  **Notes:** ______

### TC-07 — Projection returns the U-Net grid value (heavy-ish rain)
* **Objective:** prove the headline rainfall number is the committed model output at the Panchayat's own coordinate — not a default, average or cached constant.
* **Preconditions:** none.
* **Steps:**
  ```bash
  curl -s -X POST http://localhost:8000/auth/ -H "Content-Type: application/json" -d '{"panchayat_name":"ALURU","block_name":"KUNDAPURA","district":"Udupi","state":"KARNATAKA","date":"2022-07-10","requested_metrics":["rainfall","temperature","humidity","elevation"]}' | python -m json.tool
  ```
* **Expected result:** `prediction.rainfall_mm == 50.49`; `sources.rainfall == "U-Net prediction grid (prediction_test.npz)"`; `model_status == "ready"`; `sources.location_precision == "exact"`; temperature/humidity/elevation non-null.
  Cross-check with the §7.2 snippet: `(50.49, 46, 67)`.
* **Actual result:** ______  **Pass/Fail:** ☐ ☐  **Evidence:** ______  **Notes:** ______

### TC-08 — Projection returns a near-zero value (zero-rain Panchayat)
* **Objective:** confirm a legitimately dry Panchayat returns a real `0.0` value, distinguishing "no rain" from "no data".
* **Steps:** POST `/auth/` with `{"panchayat_name":"ATMAKUR","block_name":"ATMAKUR","district":"Anantapur","state":"ANDHRA PRADESH","date":"2022-07-10"}`.
* **Expected result:** `prediction.rainfall_mm == 0.0` (raw cell value `0.0049` rounded to `0.0`); source = U-Net grid. Must **not** be `null` and must not raise.
* **Actual result:** ______  **Pass/Fail:** ☐ ☐  **Evidence:** ______  **Notes:** ______

### TC-09 — Date handling and boundaries
* **Objective:** only the 122 committed dates are servable; the default date is applied.
* **Steps:**
  ```bash
  # in range (first day)
  curl -s -X POST http://localhost:8000/auth/ -H "Content-Type: application/json" -d '{"panchayat_name":"ALURU","date":"2022-06-01"}'
  # last day
  curl -s -X POST http://localhost:8000/auth/ -H "Content-Type: application/json" -d '{"panchayat_name":"ALURU","date":"2022-09-30"}'
  # before range
  curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:8000/auth/ -H "Content-Type: application/json" -d '{"panchayat_name":"ALURU","date":"2021-01-01"}'
  # no date -> default 2022-07-10
  curl -s -X POST http://localhost:8000/auth/ -H "Content-Type: application/json" -d '{"panchayat_name":"ALURU"}'
  ```
* **Expected result:** `2022-06-01` → **0.0** (verified: ALURU cell is 0.0 on that date); `2022-09-30` → 200 with a value in range; `2021-01-01` → **422** whose message names the available range (`2022-06-01` … `2022-09-30`); no-date → same as `2022-07-10` → **50.49**.
* **Actual result:** ______  **Pass/Fail:** ☐ ☐  **Evidence:** ______  **Notes:** ______

### TC-10 — Coastal / masked-cell Panchayat fails loudly (no fake number)
* **Objective:** a Panchayat whose sampled cell is NaN (sea/excluded, M=0) must error, not invent.
* **Steps:**
  ```bash
  curl -s -w '\nHTTP %{http_code}\n' -X POST http://localhost:8000/auth/ -H "Content-Type: application/json" -d '{"panchayat_name":"MANDURIVARIPALEM","block_name":"ONGOLE","district":"Prakasam","state":"ANDHRA PRADESH","date":"2022-07-10"}'
  ```
* **Expected result:** **HTTP 422** with a message that the model output is not finite at this cell (sea/excluded). `rainfall_mm` must never be `0.0` or a guessed value here.
* **Actual result:** ______  **Pass/Fail:** ☐ ☐  **Evidence:** ______  **Notes:** ______

### TC-11 — Duplicate coordinates ⇒ identical rainfall (documented limitation)
* **Objective:** expose that `panchayat_index.csv` collapses coordinates, so distinct Panchayats repeat one value.
* **Steps:**
  ```bash
  curl -s "http://localhost:8000/api/panchayats?q=swapuram&limit=3" | python -m json.tool   # AMERDA, AMMAGARIPALLI, ANANDAPURAM
  curl -s -X POST http://localhost:8000/auth/ -H "Content-Type: application/json" -d '{"panchayat_name":"AMERDA","date":"2022-07-10"}' | python -m json.tool
  curl -s -X POST http://localhost:8000/auth/ -H "Content-Type: application/json" -d '{"panchayat_name":"ANANDAPURAM","district":"Bhadradri Kothagudem","date":"2022-07-10"}' | python -m json.tool
  .venv/Scripts/python.exe -c "import pandas as pd; d=pd.read_csv('outputs/layer2/panchayat_index.csv'); print(len(d), d.groupby(['lat','lon']).ngroups)"
  ```
* **Expected result:** both Panchayats return the **same** `rainfall_mm` (**27.68**), and the coordinate count prints `86075 989`. Mark **Pass** if the behaviour matches this documented limitation; record D-1 in `Notes:` and flag it as a **demo risk** (it breaks the "one value per Panchayat" claim).
* **Actual result:** ______  **Pass/Fail:** ☐ ☐  **Evidence:** ______  **Notes:** ______

### TC-12 — Unmapped Panchayat is never searchable
* **Objective:** confirm Panchayats with no Layer-2 mapping are excluded from search and from the served dataset (they must never receive a fabricated value).
* **Steps:**
  ```bash
  curl -s "http://localhost:8000/api/panchayats?q=ALLAMUDI"
  grep -c ',unmapped,' outputs/layer2/panchayat_summary.csv
  ```
* **Expected result:** search returns `[]`; the CSV contains `1632` `unmapped` rows; the health count (86,103) already excludes them.
* **Actual result:** ______  **Pass/Fail:** ☐ ☐  **Evidence:** ______  **Notes:** ______

### TC-13 — Panchayat with no coordinates fails cleanly (does not guess)
* **Objective:** verify that a searchable Panchayat that has no row in `panchayat_index.csv` is refused explicitly rather than given a made-up location or value.
* **Steps:**
  ```bash
  curl -s -w '\nHTTP %{http_code}\n' "http://localhost:8000/api/geocode?panchayat_name=AMBOLI&block_name=Dadra%20Nagar%20Haveli&district=Dadra%20And%20Nagar%20Haveli&state=DADRA%2CNAGAR%20HAVELI%2CDAMAN%20%26%20DIU"
  curl -s -w '\nHTTP %{http_code}\n' -X POST http://localhost:8000/auth/ -H "Content-Type: application/json" -d '{"panchayat_name":"AMBOLI","date":"2022-07-10"}'
  ```
* **Expected result (verified by replaying the storage logic):**
  * `/api/geocode` → **HTTP 404**, `"no coordinates for this panchayat. Run build_panchayat_index.py to add them"`.
  * `/auth/` → **HTTP 422**, message `Cannot produce a rainfall value: no coordinates for this panchayat (run build_panchayat_index.py). Available dates: 2022-06-01 to 2022-09-30.`
  * No location and no rainfall value may be fabricated.

  > Why not block/district? All 28 coordinate-less searchable Panchayats belong to the state `DADRA,NAGAR HAVELI,DAMAN & DIU`, whose name never matches a single state entry in `grid_admin_map_deccan.npz` (which stores `DadraandNagarHaveli` / `DamanandDiu` separately). The admin-centre fallback therefore returns `unknown` — see **D-9**. Record this; it means the `REQUIRE_EXACT_LOCATION=1` refusal path is not reachable from the committed data.
* **Actual result:** ______  **Pass/Fail:** ☐ ☐  **Evidence:** ______  **Notes:** ______

### TC-14 — Advisory severity follows the rainfall bands
* **Objective:** rule engine maps inputs → severity deterministically.
* **Steps:**
  ```bash
  for R in 70 50.49 30 12 2; do
    curl -s "http://localhost:8000/api/advisory?panchayat_id=220447&crop=wheat&stage=general&rainfall_mm=$R&date=2022-07-10&panchayat_name=ALURU" \
      | .venv/Scripts/python.exe -c "import sys,json;d=json.load(sys.stdin);print(d['evidence']['rainfall_mm'],d['severity'],d['evidence']['risk_level'],d['message_source'])"
  done
  ```
* **Expected result:**
  | `rainfall_mm` | `severity` | `evidence.risk_level` |
  |---|---|---|
  | 70 | `alert` | `very_heavy` |
  | 50.49 | `warning` | `very_heavy` |
  | 30 | `warning` | `heavy` |
  | 12 | `info` | `moderate` |
  | 2 | `watch` (R2 rain-only fallback, severity *low*) | `no_rain` |
  `message_source` is `template` (no key) or `llm:<model>`.
* **Actual result:** ______  **Pass/Fail:** ☐ ☐  **Evidence:** ______  **Notes:** ______

### TC-15 — Advisory is consistent with, and sensitive to, its inputs
* **Objective:** the advisory text/evidence must reflect the actual numbers and change when they change.
* **Steps:**
  1. Call `GET /api/advisory` with `rainfall_mm=70, crop=rice, stage=flowering, temperature_c=36, humidity_pct=90, lang=en-IN`.
  2. Repeat with `rainfall_mm=5` (all else identical).
  3. Repeat with `crop=banana`.
  4. Compare `evidence.rainfall_mm` to what you sent; diff the two response bodies.
* **Expected result:** (1) `severity == "alert"`, `evidence.rainfall_mm == 70`, `crop == "rice"`, `stage == "flowering"`, `actions` non-empty, `fired_rules` includes `R1_HEAVY_RAIN`; (2) `severity == "info"`, `fired_rules` no longer contains `R1_HEAVY_RAIN`, actions differ; (3) **422**; (4) no number appears in `advisory_text` that is absent from `trace`/`evidence`.
* **Actual result:** ______  **Pass/Fail:** ☐ ☐  **Evidence:** ______  **Notes:** ______

### TC-16 — LLM rewrite cannot invent numbers (faithfulness guard)
* **Objective:** verify the guard documented in `advisory.py::is_faithful`.
* **Steps:**
  ```bash
  cd backend && .venv/Scripts/python.exe -m pytest -q tests/test_advisory.py -k hallucinated -v
  # and, if you have a GROQ_API_KEY:
  curl -s -X POST http://localhost:8000/api/advisory -H "Content-Type: application/json" -d '{"data":{"panchayat":"X","crop":"wheat","stage":"grain_filling","rainfall_mm":70,"tmax_c":30,"humidity_pct":60,"soil_moisture":0.25,"wind_kmh":10},"output_language":"en-IN"}' | python -m json.tool
  ```
* **Expected result:** the unit test passes; and for the live call, `message_source` is `template` **or** `llm:<model>` — and if `llm:*`, `message_en` contains **no** number absent from `trace`. If a rewrite was rejected, an `attempts` entry with `"error": "unfaithful_numbers"` appears.
* **Actual result:** ______  **Pass/Fail:** ☐ ☐  **Evidence:** ______  **Notes:** ______

### TC-17 — Metrics equal the committed evaluation files
* **Objective:** verify the performance numbers shown to judges come from the evaluation artefacts rather than hardcoded constants.
* **Steps:** `curl -s http://localhost:8000/api/metrics | python -m json.tool` + §7.3 snippet.
* **Expected result:** `selected_model.variant_key == "D"`, `test_mae_mm == 8.167`, `test_rmse_mm == 16.014`, `test_correlation == 0.4461`, `baseline.test_mae_mm == 9.603`, `mae_improvement_pct == 15.0`, `n_parameters == "117,329"`.
* **Actual result:** ______  **Pass/Fail:** ☐ ☐  **Evidence:** ______  **Notes:** ______

### TC-18 — XAI explain endpoint is grounded and degrades cleanly
* **Objective:** verify the explanation only cites factors the model was actually trained on, and that it still answers usefully when the LLM is unavailable.
* **Steps:**
  ```bash
  curl -s http://localhost:8000/api/explain/status
  curl -s -X POST http://localhost:8000/api/explain -H "Content-Type: application/json" -d '{
    "panchayat_name":"ALURU","block_name":"KUNDAPURA","district":"Udupi","state":"KARNATAKA","date":"2022-07-10",
    "lat":13.679452,"lon":74.742466,
    "prediction":{"rainfall_mm":50.49,"risk_level":"very_heavy","temperature_c":26.0,"humidity_pct":80,"elevation_m":20.0},
    "mapping":{"method":"direct_grid","n_cells":2,"fallback_distance_m":null},
    "model":{"name":"U-Net + DEM + ERA5-Land","channels":["imd_rain","dem","era5_t2m","era5_t2m_max","era5_dewp"],
             "test_mae_mm":8.167,"baseline_mae_mm":9.603,"mae_improvement_pct":15.0,"reference_product":"CHIRPS v2.0"},
    "language":"en"}' | python -m json.tool
  ```
* **Expected result:** `provider` is `rules` (no key) or `groq` (key present); `summary` contains `50.5 mm/day`; `factors` names only the trained channels (with all 5 channels you should see IMD rainfall, Elevation, Moisture, Temperature); every `weight` in `[0,1]`; with no key, `fallback_reason` explains why rules were used; a `question` with no key returns a non-empty `answer` that explicitly says follow-up questions are unavailable.
* **Actual result:** ______  **Pass/Fail:** ☐ ☐  **Evidence:** ______  **Notes:** ______

### TC-19 — Weather lookup and external-API failure behaviour
* **Objective:** verify temperature/humidity/elevation are fetched from real sources with provenance labels, and that an external outage yields nulls rather than invented values.
* **Steps:**
  ```bash
  curl -s "http://localhost:8000/api/weather?lat=13.679452&lon=74.742466&date=2022-07-10" | python -m json.tool
  # then disable networking (or point to a dead host) and repeat
  ```
* **Expected result:** with network: `temperature_c`, `humidity_pct`, `elevation_m` non-null and each carrying a `*_source` string (`Open-Meteo Archive (ERA5 reanalysis)`, and elevation either `SRTM DEM (model training grid)` or `Open-Topo-Data (SRTM 90 m)`); `error` is `null`.
  Without network: the route still returns **200** with `null` values and a non-empty `error` string — it must **never** fabricate numbers and must not 500.
* **Actual result:** ______  **Pass/Fail:** ☐ ☐  **Evidence:** ______  **Notes:** ______

### TC-20 — Full UI flow is internally consistent
* **Objective:** verify the browser shows exactly what the API returns across search → projection → advisory → explanation.
* **Steps:** in the browser: search `aluru` → select **ALURU (KUNDAPURA, Udupi)** → click **Run Projection →** → wait for the advisory panel → open the XAI panel.
* **Expected result:** selected name/block/district shown; rainfall ≈ **50.49 mm**; risk badge consistent with `classifyRisk(50.49)`; advisory severity `warning`; evidence row shows `50.5 mm`; XAI explanation present. All displayed numbers must equal the `curl` responses from TC-07/TC-14/TC-18.
* **Actual result:** ______  **Pass/Fail:** ☐ ☐  **Evidence:** ______  **Notes:** ______

### TC-21 — Latency within the §9.2 budget
* **Objective:** confirm the interactive flow is fast enough for a live demo on a typical laptop and connection.
* **Steps:**
  ```bash
  curl -s -o /dev/null -w 'root %{time_total}\n' http://localhost:8000/
  curl -s -o /dev/null -w 'search %{time_total}\n' "http://localhost:8000/api/panchayats?q=kundapura"
  curl -s -o /dev/null -w 'auth %{time_total}\n' -X POST http://localhost:8000/auth/ -H "Content-Type: application/json" -d '{"panchayat_name":"ALURU","date":"2022-07-10"}'
  curl -s -o /dev/null -w 'advisory %{time_total}\n' "http://localhost:8000/api/advisory?panchayat_id=220447&rainfall_mm=50.49&crop=wheat&stage=general"
  ```
* **Expected result:** each within the §9.2 budget; `auth` may exceed 1.5 s on a slow link (network calls) — record the reason.
* **Actual result:** ______  **Pass/Fail:** ☐ ☐  **Evidence:** ______  **Notes:** ______

### TC-22 — Auxiliary layers (soil / NDVI / LULC) are internally correct
**[Offline data check only — the app does not read these files; see §13]**
* **Objective:** verify the committed auxiliary arrays are structurally sound and match their documented statistics, even though no route consumes them yet.
* **Steps:** run from the repo root:
  ```bash
  .venv/Scripts/python.exe - <<'PY'
  import numpy as np
  s = np.load("data/aux_data/soil_soilgrids_deccan.npz")
  n = np.load("data/aux_data/ndvi_monthly_deccan.npz")
  l = np.load("data/aux_data/lulc_fractions_deccan.npz")
  print("soil keys:", list(s.keys()));       print("soil fine_values:", s["fine_values"].shape)
  print("ndvi keys:", list(n.keys()));       print("ndvi:", n["fine_values"].shape if "fine_values" in n else "?")
  print("lulc keys:", list(l.keys()));       print("lulc:", l["fractions"].shape)
  PY
  ```
* **Expected result:** soil is `(47250, 5)` (sand/clay/ocd/phh2o/bdod, ~8.8–9.4% NaN per property); NDVI is 20 monthly composites with ~8.0% missing and values within `-0.064 … 0.845`; LULC is `(47250, 6)` with ~97.14% cell coverage. Cross-check against `data/aux_data/build_summary_deccan.json`.
* **Actual result:** ______  **Pass/Fail:** ☐ ☐  **Evidence:** ______  **Notes:** ______

---

## 12. Traceability — how to walk a displayed advisory back to the source

### 12.1 The chain (as actually wired on `b1`)

```
UI advisory text + severity + evidence.rainfall_mm
  └─ GET /api/advisory?rainfall_mm=<value>        ← value is passed by the UI, not looked up
       └─ the UI obtained <value> from
            POST /auth/  →  prediction.rainfall_mm
              ├─ (1) outputs/layer2/panchayat_weather.csv     [ABSENT on b1 → path skipped]
              ├─ (2) outputs/prediction_test.npz              ← used
              │        sampled at the Panchayat coordinate from
              │        outputs/layer2/panchayat_index.csv
              │        (nearest fine cell, round 2dp; NaN → 422)
              └─ (3) live U-Net inference                     [only if ENABLE_LIVE_INFERENCE=1]
                       └─ models/best_model.pt over data/processed/X_test.npy
   Layer-2 metadata (mapping_method, n_cells, season mean) ← outputs/layer2/panchayat_summary.csv
   Rules / severity / actions                              ← backend/advisory.py
   Metric claims (MAE etc.)                                ← outputs/metrics/ablation.json
   Aux layers (soil/NDVI/LULC)                             ← data/aux_data/*.npz  [NOT in the chain]
```

### 12.2 Worked example — "50.5 mm/day for ALURU"

| Step | Artefact | Evidence to capture |
|---|---|---|
| 1 | UI shows `~50.5 mm` and advisory `warning` | screenshot (TC-20) |
| 2 | `/api/advisory…rainfall_mm=50.49` → `severity:"warning"`, `evidence.rainfall_mm:50.49` | raw JSON (TC-14) |
| 3 | `/auth/` → `prediction.rainfall_mm:50.49`, `sources.rainfall:"U-Net prediction grid (prediction_test.npz)"` | raw JSON (TC-07) |
| 4 | `panchayat_index.csv` → `220447 → 13.679452, 74.742466` | grep output |
| 5 | `prediction_test.npz[2022-07-10, 46, 67] = 50.487484` → `round → 50.49` | §7.2 snippet output |
| 6 | `panchayat_summary.csv` → `ALURU, direct_grid, n_cells 2, rainfall_mean_mm 12.13` | grep row |
| 7 | rule `R1B_SUBSTANTIAL_RAIN` (24.5 ≤ 50.49 < 64.5) → `warning` | `fired_rules` in JSON |

A chain is **broken** if any of these are unexplained: the UI number ≠ step 3 ≠ step 5; or `sources.rainfall` names a file that does not exist; or `severity` contradicts §8.1.

### 12.3 What CANNOT be traced today (state honestly)

* Advisory → **soil / NDVI / LULC**: no link exists; the rules never read those arrays.
* Advisory → **per-date per-Panchayat Layer-2 value**: `panchayat_weather.csv` is absent, so `/auth/` serves the raw grid sample instead of a polygon-aggregated value.
* Layer-2 aggregate → **geometry**: `panchayat_weather.geojson` is claimed in `README.md` §8 but is **not in the `b1` tree**; `panchayat_weather_map.png` is present, but there is no per-Panchayat geometry in the app.
* Panchayat → **its own polygon**: the served coordinate is an admin-level centroid (D-1), and the `location_precision` field that is supposed to flag this always says `exact` (D-9). The trace therefore stops at "a coordinate in `panchayat_index.csv`", not at "this Panchayat's boundary".
* Sub-district/block aggregation → **`block_rainfall.csv`** exists (137,006 rows) but **no route reads it**.

---

## 13. Register — "Not testable with current implementation"

| # | Item | Why | What you *can* do instead |
|---|---|---|---|
| N-1 | SoilGrids / NDVI / LULC values **in the product flow** | No file under `backend/` references `soil_soilgrids_*`, `ndvi_monthly_*` or `lulc_fractions_*` (verified with `git grep` over the whole backend tree); `advisory.py` never consumes them. **Nuance:** the *admin* aux file `grid_admin_map_deccan.npz` **is** read, by `data_store.Store._admin_centres()`, to build block/district centres — only the soil / vegetation / land-cover layers are unwired | TC-22 offline integrity check |
| N-2 | Soil-moisture (SMAP / ERA5-Land 0–7 cm) | `soilmoisture_daily_deccan.npz` was never materialised; `soil_moisture` is only a caller-supplied field | POST `/api/advisory` passing `soil_moisture` manually |
| N-3 | Per-date per-Panchayat Layer-2 value via `/auth/` | `outputs/layer2/panchayat_weather.csv` absent on `b1` | regenerate (needs LGD parquet) or verify against the grid path (TC-07) |
| N-4 | Live U-Net inference for arbitrary dates | `ENABLE_LIVE_INFERENCE=0`; needs `data/processed/*.npy` + `data/raw/` (not in git) | TC-07 grid path |
| N-5 | Block-rainfall route | `block_rainfall.csv` is committed but unused by any route | offline row-count check |
| N-6 | Wind / lodging rule through the UI | no wind source in the app; `/api/advisory` (GET) does not pass `wind_kmh` | unit test / POST `/api/advisory` with `wind_kmh` |
| N-7 | Groq LLM wording quality | needs `GROQ_API_KEY` + network | template fallback path (TC-14/TC-16) |
| N-8 | Sarvam translation (`lang=hi-IN` etc.) | needs `SARVAM_API_KEY` + network | confirm the endpoint returns the English text unchanged when translation fails |
| N-9 | Layer-1 dataset rebuild (`verify_dataset.py`) | `data/processed/*.npy` are not in git (`data.zip` only) | out of scope for this plan |
| N-10 | `panchayat_weather.geojson`, per-Panchayat geometry | claimed in README §8, **not present** in the tree | flag as a documentation/deliverable gap (D-4) |

---

## 14. Acceptance criteria — is the demo ready?

The product is **demo-ready** only when **all** of the following hold:

**Blocking (must all pass)**
1. GATE-1, GATE-2, GATE-3 pass on a clean laptop.
2. TC-02: `37 passed`.
3. TC-03: `panchayats_loaded == 86103`.
4. TC-05: search returns the pipeline's real `mapping_method` / `n_cells`; an `unmapped` Panchayat returns `[]`.
5. TC-07: the projection value is reproducible from `prediction_test.npz` via §7.2 (exact match to 2 dp).
6. TC-09: in-range dates work; out-of-range dates return **422** with the available range (never a fabricated number).
7. TC-10: masked/coastal Panchayats return **422**, never `0.0` or a guess.
8. TC-14/TC-15: advisory severity follows the rainfall bands, `evidence.rainfall_mm` equals the input, and the advisory changes when the rainfall changes.
9. TC-20: the UI numbers equal the API numbers for the same selection.
10. `curl` timings within the §9.2 budget for `/`, search, and advisory.

**Non-blocking but must be recorded and disclosed in the demo**
11. D-1 (coordinate collapse: 989 distinct coordinates for 86,075 Panchayats) — the "one value per Panchayat" claim is **not** supportable as-is.
12. D-3 (`panchayat_weather.csv` absent ⇒ `/auth/` serves the raw grid cell, not a polygon-aggregated Layer-2 value).
13. N-1/N-2: soil/NDVI/LULC/soil-moisture are **not** driving the advisory.

**Explicitly out of scope for "demo-ready"**
14. Groq/Sarvam live behaviour (N-7/N-8) — the demo must work with `provider: "rules"` and `message_source: "template"`.
15. Layer-2 regeneration, live inference, and the Deccan dataset rebuild.

---

## 15. Defects and risks found while writing this plan

> These were found by reading the `b1` tree and the committed artefacts. **They are not test results** — confirm each with the referenced case before reporting it as a bug.
> **Verification status:** D-1 was re-checked over the **full population** (all 86,075 rows, not a sample); D-5 by comparing the patch hunk against the real file; D-6 by `git grep` over the whole `Frontend/` tree. D-2/D-3/D-4/D-7/D-8/D-9 are direct tree/content reads. One earlier claim was **withdrawn**: the initial draft said no backend code reads any `data/aux_data/*.npz` — that is false for the *admin* layer (`data_store._admin_centres()` reads `grid_admin_map_deccan.npz`); it is only true for soil/NDVI/LULC (see N-1).

| ID | Severity | Finding (evidence) | Reproduce with |
|---|---|---|---|
| **D-1** | **High** | `outputs/layer2/panchayat_index.csv` has 86,075 rows but only **989 distinct `(lat, lon)` pairs**. Full-population verification: **977 of the 989 coordinates (98.8%) are exactly a block/district centre** computed from `grid_admin_map_deccan.npz`, and **76,327 of 86,075 rows (88.7%)** carry such a centre — i.e. the file stores **admin-level centroids, not per-Panchayat polygon points** (e.g. ALURU = the KUNDAPURA block centre; ATMAKUR = the Anantapur district centre). The **median coordinate is shared by 57 Panchayats**; 219 coordinates are shared by ≥100; the worst holds **3,763**. `/api/geocode` still reports `"exact"` and `/auth/` then returns one identical rainfall value for all of them — the "panchayat-level" output is not actually Panchayat-level. | full-population replay of `data_store.Store._admin_centres()` vs the index; `python -c "import pandas as pd;d=pd.read_csv('outputs/layer2/panchayat_index.csv');print(len(d), d.groupby(['lat','lon']).ngroups)"` → `86075 989`; TC-11 |
| **D-2** | Low | `backend/README.md` instructs `cp .env.example .env`, but **no `backend/.env.example` exists** in the `b1` tree (the only path matching `*env*` is `Frontend/next-env.d.ts`). A fresh teammate hits `cp: cannot stat`. Onboarding blocker, not a functional defect (the app still runs keyless). | `ls backend/.env.example` → not found; §2.4 |
| **D-3** | Medium | `README.md` §8 lists `outputs/layer2/panchayat_weather.csv` and `panchayat_weather.geojson` as produced artefacts, but **neither is in the tree**. Consequently `/auth/` cannot serve per-date Layer-2 values and falls back to the raw grid. | `git ls-tree -r --name-only b1 \| grep layer2` |
| **D-4** | Low | `outputs/layer2/layer2_qc.json` `outputs.*` paths point at a different machine (`C:\Users\5717a\Downloads\…`), so the QC record is not portable. | read `layer2_qc.json` |
| **D-5** | Low | **Verified by binary comparison:** `frontend_backend_match.patch` **cannot apply to `b1`**. Its hunk expects the context `stage: CropStage,` → `irrigationAvailable?: boolean,`, but `b1`'s `backend.ts` has `rainfallMm: number,` at that position and a different comment. `backend/README.md` still instructs applying it, so a teammate following the README will get a patch failure. | compare `git show origin/b1:frontend_backend_match.patch` with `Frontend/src/api/backend.ts` lines 70–88; or `git apply --check --directory=Frontend frontend_backend_match.patch` |
| **D-6** | Low | **Verified dead code:** `Frontend/src/app/api/weather/route.ts` (Next server route → Open-Meteo/Open-Topo-Data) duplicates the backend's `/api/weather`. `fetchWeather` is *exported* in `Frontend/src/api/backend.ts` but **never called** anywhere in `Frontend/`, and no component issues a relative `/api/weather` request — so neither that Next route nor the UI path is exercised. The backend `/api/weather` route itself remains reachable and is still worth testing directly (TC-19). | `git grep -n "fetchWeather" origin/b1 -- Frontend` → only the definition; `git grep "'/api/weather'"` → none |
| **D-7** | Low | `README.md`, `backend/README.md`, and `HANDOVER.md` on `b1` still describe the app as `weather-downscaling-main/…` even though the app now lives at the repo root (`backend/`, `Frontend/`). Misleading for a new tester. | compare `README.md` §0 file map with `ls` |
| **D-8** | Info | `HANDOVER.md` (486 lines) documents **only** the ML pipeline — there is no backend/frontend/`pytest`/Groq/Sarvam section. This document is the first end-to-end ops guide for the app. | `grep -niE "backend\|frontend\|groq\|pytest" HANDOVER.md` |
| **D-9** | Medium | The `location_precision` `block`/`district` fallback path is **unreachable with the committed data**. The only searchable Panchayats missing from `panchayat_index.csv` are the 28 `DADRA,NAGAR HAVELI,DAMAN & DIU` rows, and that state string never matches `grid_admin_map_deccan.npz` state names (`DadraandNagarHaveli`, `DamanandDiu`), so `locate_row()` returns `unknown` → **404 / 422** instead of a block centre. Consequently `REQUIRE_EXACT_LOCATION=1` can never trigger, and `location_precision` is effectively always `exact` (misleading, see D-1). | TC-13; `data_store.locate_row` |

---

## 16. Appendix

### 16.1 Verified expected-value cheat-sheet (2022-07-10 unless stated)

| Panchayat | id | State / District / Block | mapping_method | n_cells | season mean (search) | `rainfall_mm` from `/auth/` |
|---|---|---|---|---|---|---|
| ALURU | 220447 | KARNATAKA / Udupi / KUNDAPURA | direct_grid | 2 | 12.13 | **50.49** |
| NAGULAMALLIAL | 201416 | TELANGANA / Karimnagar / Kothapally | direct_grid | 1 | 5.48 | **41.77** |
| AMERDA | 202120 | TELANGANA / Bhadradri Kothagudem / ASWAPURAM | direct_grid | 4 | — | **27.68** |
| CURDI | 254406 | GOA / South Goa / SANGUEM | direct_grid | 1 | 6.83 | **16.73** |
| IDDAMPALLY | 207422 | TELANGANA / Nalgonda / DEVARAKONDA | area_weighted | 4 | 1.38 | **3.88** |
| NOOLPUZHA | 221933 | KERALA / Wayanad / SULTHAN BATHERY | direct_grid | 12 | 2.14 | **1.58** |
| SULAJ | 172745 | MAHARASHTRA / Buldhana / JALGAONJAMOD | area_weighted | 2 | 3.02 | **1.49** |
| AMRUTHALUR | 199960 | ANDHRA PRADESH / Bapatla / AMRUTHALUR | nearest_fallback | 1 | — | **0.13** |
| ATMAKUR | 195827 | ANDHRA PRADESH / Anantapur / ATMAKUR | direct_grid | 3 | 0.92 | **0.0** |
| MANTAPAMPALLI | 198911 | ANDHRA PRADESH / Y.S.R. / VONTIMITTA | area_weighted | 4 | 0.80 | **0.0** |
| MANDURIVARIPALEM | 234797 | ANDHRA PRADESH / Prakasam / ONGOLE | nearest_fallback | 1 | — | **422 (NaN cell)** |

Global (all 122 days): rainfall `0.0 … 163.28`, mean `3.93`. On `2022-07-10`: `0.0 … 83.71`, mean `11.95`. No committed Panchayat coordinate samples a value ≥ 64.5 on that date — use `GET /api/advisory` directly to exercise the `alert` band.

### 16.2 Copy-paste case template

```markdown
### TC-XX — <title>
* **Objective:**
* **Preconditions:**
* **Steps:**
* **Expected result:**
* **Actual result:**
* **Pass/Fail:** ☐ Pass ☐ Fail
* **Evidence:**
* **Notes:**
```

### 16.3 Cleanup after testing

```bash
# stop uvicorn and `npm run dev` (Ctrl-C).
git status           # must show no tracked file modified
# Remove only what this plan created:
#   backend/.env, Frontend/.env.local, node_modules/, qa-evidence/
```
If `git status` shows a modified tracked file, a failure was "fixed" during testing — revert it (`git checkout -- <file>`) and re-run that case.

---

*End of plan. Target: branch `b1` @ `62d4a343911ca9ec8ddd5189c7cdafaf633a5a72`.*
