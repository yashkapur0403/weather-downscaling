# `weather-downscaling-main/scripts/` — what is live vs. legacy

This folder belongs to the **app** (`weather-downscaling-main/`, Aanya's
backend + frontend). It is *not* the ML pipeline root.

## The ML pipeline root is `../../scripts/`
`weather-downscaling/scripts/` is the **ACTIVE, final (Deccan) pipeline**:
285×200 fine grid, 610 days, 366/122/122 split, 5 channels. Run everything
from the repo root (`weather-downscaling/`):

```
download_or_export.py --region deccan   ->  build_aux.py --region deccan
->  preprocess.py  ->  verify_dataset.py  ->  train.py  ->  infer.py
```

## This folder = Western-Ghats PILOT (legacy, kept for reference)
All scripts here **except the three Layer-2 ones below** are the old
Western-Ghats **pilot** snapshot (85×85, 122-day). They are preserved for
historical reference and are **superseded by `../../scripts/`**. Do not run
them as the current pipeline. The pilot `config.py` here (ROI 13–17 N,
74.25–78.25 E) is the pilot config and is kept as-is.

## Still live here: Layer 2 (panchayat mapping)
Three scripts here are part of the live pipeline and run against the **current
root Deccan config/data** via the `layer2_config.py` shim:

- `layer2_panchayat_mapping.py` — maps Layer-1 rainfall onto Gram Panchayats
- `check_blocks.py` — Layer-2 coverage/edge diagnostics
- `export_pickle.py` — packages Layer-1 model / field / Layer-2 table as `.pkl`

They import `layer2_config` (not the pilot `config`), so they read the current
root pipeline (`repo-root/prediction/infer_*.npz`, `outputs/maps/`, deccan ROI
and grid) while still writing their Layer-2 output to this app folder:

    weather-downscaling-main/outputs/layer2/panchayat_weather.csv   <- backend reads this

Layer-2 also needs the LGD panchayat polygons
(`data/raw/administrative/panchayat/LGD_Panchayats.parquet`) which have **not
been acquired yet** — that tier is documented as future work
(see `HANDOVER.md` §4.2).
