# QA evidence

One script per investigation, each with the JSON result it produced when it was run. These are the
files `docs/LOGICAL_VALIDATION_REPORT.md` §19 and `docs/COMPREHENSIVE_QA_TEST_REPORT.md` cite as
evidence, so they are kept in the repository instead of a scratch directory: every number and every
finding in those reports traces back to one of these runs. Nothing here is imported by the pipeline,
the backend or the frontend — it is evidence, not production code.

## What each script covers

| Script | Output | Covers |
|---|---|---|
| `qa_logic_core.py` | `qa_logic_core.json` | Layer-1 → Layer-2 numeric integrity: 13/13 traces exact, orientation/transpose controls, units, polygon-mean vs the nearest-cell value actually served, `panchayat_summary` staleness, temporal index (54/54), spatial sanity, admin joins |
| `qa_logic_advisory.py` | `qa_logic_advisory.json` | The Layer-3 rule engine: every threshold at T±ε, 11 one-input perturbations, action coherence, the LLM faithfulness guard |
| `qa_logic_http.py` | `qa_logic_http.json` | Live HTTP: identity binding (12/12), ambiguous names, sequence + concurrency isolation (8/8), zero-rain reachability, Layer-3 input binding, enums/errors |
| `qa_f1_50.py` | `qa_f1_50.json` | Why the first deployed checkpoint failed the ≥50 mm threshold (precision 0.438, recall 0.076) |
| `qa_eval_f1.py` | `qa_f1_50_variants.json` | Per-checkpoint event F1 at 10/25/50 mm (E, F, EF vs the baseline) |
| `qa_blend.py` | `qa_blend.json` | Ensemble / blend candidates on the val + test splits |
| `qa_ensemble_smoke.py` | — | `scripts/ensemble.py` arithmetic: max difference 0.0 against a hand-computed weighted mean |
| `qa_blackbox.py` | `qa_blackbox.json` | 48 black-box API cases over live HTTP |
| `qa_whitebox.py` | `qa_whitebox.json` | 32 data / code white-box cases |
| `qa_followup.py` | `qa_followup.json` | 10 mask / aux / dataset cases |
| `qa_uat.py` | `qa_uat.json` | 14 UAT + E2E + integration cases |
| `qa_deep.py` | `qa_deep.json` | 12 round-2 value-analysis / remaining-path cases |
| `qa_model.py` | — | Checkpoint load, channel order, parameter count |
| `qa_repro_metrics.py` | — | Layer-1 metric reproduction from `data.zip` |

## Running them

Run them **from the repository root** (they read `outputs/`, `models/` and `data/` by relative path).
They need no API keys, but two environments are involved:

```bash
# Layer 1 / Layer 2 / model logic needs torch + geopandas (root .venv, README section 5)
.venv/Scripts/python.exe qa/qa_logic_core.py
.venv/Scripts/python.exe qa/qa_ensemble_smoke.py

# the backend / HTTP ones need fastapi (backend venv, backend/README.md)
backend/.venv/Scripts/python.exe qa/qa_logic_advisory.py
backend/.venv/Scripts/python.exe qa/qa_logic_http.py
backend/.venv/Scripts/python.exe qa/qa_blackbox.py
```

The HTTP suites need a live server, so start one first (it is also how the recorded evidence was
produced):

```bash
cd backend && uvicorn app:app --host 127.0.0.1 --port 8000
```

Note that re-running a script **overwrites** its JSON. The committed JSON is the record of the run
quoted in the reports, so unless you mean to replace that evidence, read the committed file rather
than re-running.
