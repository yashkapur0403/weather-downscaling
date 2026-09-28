# Coverage report - region `deccan`

Region bbox: lat 11.5..25.5, lon 71.5..81.25
Study window: 2018-06-01 .. 2022-09-30 (monsoon only, 2018,2019,2020,2021,2022)
Days: 610 (per year 122)

## Administrative (block tier = GADM L3 subdistricts)
States in bbox: 17 (of - in India)
Districts in bbox: 235 (of -)
Subdistricts (block tier) in bbox: 1201
Hit by land grid cells: 15 states / 214 districts / 1123 subdistricts
PIP spot-check agreement (200 cells): 0.68 (200-cell spot check, original build)
Gram Panchayats / villages: NOT included (LGD bulk download unavailable) - LGD-joinable schema provided.

## Grids
Land coarse cells: 1890/2280
Land fine cells: 47250/57000
Soil sample points: 1890 (batches ok 30/30)
NDVI months: 20 (valid-pixel fractions per month stored)
LULC cells with class data: 97.14%
Soil-moisture days: - on a -x- ERA5 lattice (Layer-3 aux; 0-7/7-28 cm)

## Model-ready (after preprocess.py, if present)
Samples: 610 days
Split: years {'train': [2018, 2019, 2020], 'val': [2021], 'test': [2022]}
train/val/test days: 366/122/122
Channels: ['imd_rain', 'dem', 'era5_t2m', 'era5_t2m_max', 'era5_dewp']
Land fraction of box: 0.8289
IMD-CHIRPS coarse corr: 0.356
