# Backend: weather-downscaling (single FastAPI app, port 8000)

Put this folder at the repo root as `backend/` (next to `outputs/`, `models/`, `scripts/`).

```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # add GROQ_API_KEY and SARVAM_API_KEY
python -m pytest -q tests                              # 37 tests, no network or keys needed
uvicorn app:app --reload --port 8000
```
Frontend: `NEXT_PUBLIC_API_URL=http://localhost:8000` (already the default). Apply the frontend fixes from the repo root:
`git apply --directory=Frontend frontend_backend_match.patch`

## Routes (all match Frontend/src/types/index.ts)
| Route | Source of the data |
|---|---|
| GET `/` | health |
| GET `/api/panchayats?q=` | `outputs/layer2/panchayat_summary.csv` (real `mapping_method`, `n_cells`; unmapped rows hidden) |
| GET `/api/geocode` | `panchayat_index.csv` if present (exact), else district/block centre from the admin grid (flagged) |
| POST `/auth/` | rainfall: per-date Layer-2 CSV if present, else the U-Net grid `outputs/prediction_test.npz` (2022-06-01..09-30), else live inference (`ENABLE_LIVE_INFERENCE=1`); temperature/humidity: Open-Meteo; elevation: training DEM or Open-Topo-Data |
| GET `/api/metrics` | `outputs/metrics/ablation.json` + `metrics.json`; parameter count from the checkpoint architecture |
| GET `/api/weather` | Open-Meteo + DEM |
| GET `/api/advisory` | rule engine (`advisory.py`); Groq only rephrases; Sarvam translates (`lang=hi-IN`) |
| POST `/api/explain`, GET `/api/explain/status` | rule-derived factors; Groq rewrites; falls back to `provider: "rules"` |

Nothing is invented: if a real source is unavailable the route returns a 4xx with the reason (rainfall) or `null` (weather).

## IMPORTANT: panchayat placement
The repo has no panchayat coordinates. Without them the backend places a panchayat at its DISTRICT (or block) centre, so every
panchayat in that district gets the same rainfall, which defeats panchayat-level output. Fix once, on a machine that has the LGD file:
```bash
python scripts/fetch_lgd_panchayats.py
python backend/build_panchayat_index.py --repo .        # writes outputs/layer2/panchayat_index.csv (restart the API)
```
Better still, run `python scripts/layer2_panchayat_mapping.py` to produce `outputs/layer2/panchayat_weather.csv`
(per date, area-weighted). The API then serves those exact values first. Set `REQUIRE_EXACT_LOCATION=1` to refuse approximate answers.
`/auth/` returns `sources.location_precision` (exact | block | district | given) so the UI can show it.
