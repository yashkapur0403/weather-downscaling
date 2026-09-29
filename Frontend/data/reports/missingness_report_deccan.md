# Missingness report

## Soil (missing % per property)
- sand: 8.84%
- clay: 8.84%
- ocd: 8.84%
- phh2o: 8.84%
- bdod: 9.42%
## NDVI
- land-month values missing: 8.04%
- NDVI range: [-0.064, 0.845]
- per-month valid-pixel fraction (cloud-driven):
  - 2018-06: ['2018-06', 0.7671]
  - 2018-07: ['2018-07', 0.7721]
  - 2018-08: ['2018-08', 0.7763]
  - 2018-09: ['2018-09', 0.7757]
  - 2019-06: ['2019-06', 0.7763]
  - 2019-07: ['2019-07', 0.771]
  - 2019-08: ['2019-08', 0.7741]
  - 2019-09: ['2019-09', 0.7661]
  - 2020-06: ['2020-06', 0.7736]
  - 2020-07: ['2020-07', 0.7734]
  - 2020-08: ['2020-08', 0.7733]
  - 2020-09: ['2020-09', 0.771]
  - 2021-06: ['2021-06', 0.7756]
  - 2021-07: ['2021-07', 0.7726]
  - 2021-08: ['2021-08', 0.7756]
  - 2021-09: ['2021-09', 0.7739]
  - 2022-06: ['2022-06', 0.7745]
  - 2022-07: ['2022-07', 0.7655]
  - 2022-08: ['2022-08', 0.777]
  - 2022-09: ['2022-09', 0.7766]
## LULC
- cells with class data: 97.14%
- cells with all-nodata stencils (WorldCover ocean class 0): 1349 of 47250
- cells whose whole 0.05-deg stencil is WorldCover nodata (class 0, ocean); sea fringe of coastal IMD grid boxes, never valid training targets (M=0)
- these cells are excluded from training targets by the model mask M, so no imputation is performed or needed
## Admin
- PIP agreement (nearest-point vs polygon): 0.68 (200-cell spot check, original build)
- panchayat/village tier: not available (LGD); documented
## Rainfall pipeline
- dates dropped by alignment/quality filter: 0
- sea cells excluded from Y/M: 390 coarse cells
- IMD input NaN pixels imputed 0 (sea bleed): documented in meta.json
## Dataset decisions (data freeze)
- wind: raw ERA5 cache complete (0% missing, 2018-2022, 2018 re-fetched in place); NOT a model channel - frozen baseline is imd_rain, dem, era5_t2m, era5_t2m_max, era5_dewp
- ESA WorldCereal crop type: evaluated, NOT included - only no-auth distribution is Zenodo 7875105 (global multi-GB ZIPs of 106 AEZ GeoTIFFs, AEZ-specific seasons); a Layer-3 standalone task
- SMAP L4 soil moisture: replaced by ERA5-Land daily volumetric water (same model as our era5_* channels) - OPTIONAL / PENDING: quota-blocked by the free Open-Meteo DAILY limit; resume with build_aux.py --region deccan --skip-admin --skip-soil --skip-ndvi --skip-lulc (per-batch/year caches make each attempt additive) - see HANDOVER.md section 7