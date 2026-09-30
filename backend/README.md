# Backend: weather-downscaling (single FastAPI app, port 8000)

Put this folder at the repo root as `backend/` (next to `outputs/`, `models/`, `scripts/`).

```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # optional: add GROQ_API_KEY / SARVAM_API_KEY
python -m pytest -q tests                              # 69 tests, no network or keys needed
uvicorn app:app --reload --port 8000
```
Frontend: `NEXT_PUBLIC_API_URL=http://localhost:8000` (already the default). The
`legacy/frontend_backend_match.patch` is already applied on this branch, so do **not** re-apply it
(`git apply --check` will fail because the fixes are present by different edits).

## Routes (all match Frontend/src/types/index.ts)
| Route | Source of the data |
|---|---|
| GET `/` | health. Loads the store itself, so `panchayats_loaded` is correct even on a cold process |
| GET `/api/panchayats?q=` | `outputs/layer2/panchayat_summary.csv` (real `mapping_method`, `n_cells`; unmapped rows hidden) |
| GET `/api/geocode` | `panchayat_index.csv` (an LGD polygon point) if present (exact), else district/block centre from the admin grid (flagged). A bare name shared by several Panchayats returns **409** with the candidate list |
| POST `/auth/` | rainfall: per-date Layer-2 CSV if present, else the U-Net grid `outputs/prediction_test.npz` (2022-06-01..09-30), else live inference (`ENABLE_LIVE_INFERENCE=1`); temperature/humidity: Open-Meteo; elevation: training DEM or Open-Topo-Data. An ambiguous name-only request returns **409** with candidates instead of silently picking one |
| GET `/api/metrics` | `outputs/metrics/ablation.json` + `metrics.json`; parameter count from the checkpoint architecture |
| GET `/api/weather` | Open-Meteo + DEM |
| GET `/api/advisory` | rule engine (`advisory.py`) **plus the committed soil/NDVI/land-cover aux layers**; the rainfall is **verified against the Layer-1 field for that `panchayat_id` + `date`** (409 if a supplied `rainfall_mm` disagrees, 422 if it cannot be verified — see below); Groq only rephrases; Sarvam translates (`lang=hi-IN`) |
| POST `/api/explain`, GET `/api/explain/status` | rule-derived factors; Groq rewrites; falls back to `provider: "rules"` |

Nothing is invented: if a real source is unavailable the route returns a 4xx with the reason
(rainfall) or `null` (weather).

## Panchayat placement (fixed)
`outputs/layer2/panchayat_index.csv` is built from the **LGD Panchayat polygons**
(`backend/build_panchayat_index.py`, using `representative_point()` inside each polygon), so every
one of the 86,103 mapped Panchayats has its **own** coordinate (87,735 distinct points — no more
shared block/district centroids). If the index is missing, the backend falls back to the
district/block centre and flags it via `sources.location_precision` (exact | block | district |
given); set `REQUIRE_EXACT_LOCATION=1` to refuse those.

```bash
python scripts/fetch_lgd_panchayats.py                 # needs the LGD parquet
python backend/build_panchayat_index.py --repo .       # writes outputs/layer2/panchayat_index.csv
```

## Auxiliary layers (Layer 3)
The committed soil / NDVI / land-cover arrays under `data/aux_data/` are read by the advisory engine
(`backend/aux_layers.py`). `GET /api/advisory` looks up the cell for `panchayat_id` and adds three
data-driven rules:

* **R6_VEGETATION** — low satellite NDVI (sparse/stressed vegetation; escalated when rain is also low)
* **R7_SOIL_DRAINAGE** — waterlogging risk on wet days over clay-rich soil
* **R8_LANDCOVER** — flags cells whose dominant land cover is not cropland (advisory is indicative)

The looked-up values are returned in `evidence.aux` and inside each rule's `inputs`.

### Why the advisory verifies the rainfall

The rules are only meaningful if they run on the Layer-1 value belonging to the Panchayat and date
that were asked about. `GET /api/advisory` therefore resolves that value itself through
`data_store.Store.panchayat_grid_value()` (published on `app.state.rainfall_resolver` by
`routes_data.register`) and:

| situation | response |
|---|---|
| `rainfall_mm` matches the stored value (within 0.011 mm, half a rounding step) | 200, `verification.rainfall = "verified_against_layer1"` |
| `rainfall_mm` omitted | 200, the stored value is used: `verification.rainfall = "resolved_from_layer1"` |
| `rainfall_mm` disagrees | **409** with `supplied_mm`, `expected_mm`, `difference_mm` |
| the date is outside the served grid | **422**, `reason: "date_unavailable"` |
| the Panchayat's cell is masked (sea/excluded) | **422**, `reason: "masked_cell"` |
| `panchayat_id` has no coordinate in the index | **422**, `reason: "unknown_panchayat"` |
| no data layer mounted (text-only) | `rainfall_mm` required; 200 is labelled `unverified_no_data_source` |

`data_date` is the date that was verified, and the response carries a `verification` block (including
the source string, the fine-grid cell and whether aux was resolved) so a caller — or a judge — can
see exactly what was checked. Temperature and humidity remain **caller-supplied** and are labelled as
such. Use `POST /api/advisory` for deliberate what-if runs that are not tied to the stored field.

## Configuration
See `.env.example`. Everything is optional — with empty keys the advisory uses the deterministic
template and `/api/explain` uses `provider: "rules"`.
