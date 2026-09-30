# Data dictionary - region `deccan`

## Model grid (Layer 1)
| name | meaning |
|---|---|
| coarse grid | IMD 0.25-deg cell centers (meta.grid.imd_lat/imd_lon) |
| fine grid | 5x5 area-tiling sub-cells of each coarse cell = 0.05 deg (meta.grid.fine_lat/fine_lon) |
| X | (n, C, Hf, Wf) input channels: imd_rain (bilinear 5x, NaN->0), dem, era5_* |
| Y | (n, 1, Hf, Wf) CHIRPS v2.0 daily rainfall (mm/day) on the fine grid, IMD-land cells only |
| M | (n, 1, Hf, Wf) 1 where Y is a valid land pixel (sea pixels are 0 by design) |
| land mask | IMD cell = land iff >=50% of aligned days valid; sea cells excluded from Y/M/metrics |

## Auxiliary layers (data/aux/) - NOT U-Net inputs
| file | contents | units |
|---|---|---|
| admin/admin_master_deccan.geojson | block-tier (GADM L3 subdistrict) polygons + state/district IDs, names, centroids | deg (EPSG:4326) |
| admin/grid_admin_map_deccan.npz | fine land cell -> (state, district, subdistrict) nearest-representative-point mapping | indices/IDs |
| soil_soilgrids_deccan.npz | sand, clay, ocd, phh2o, bdod at 5-15cm: sampled on land 0.25-deg cells; each fine pixel inherits its parent coarse-cell value (no bilinear smoothing across the land lattice) | raw SoilGrids mapped units (x10 factors; see units_note inside the file) |
| ndvi_monthly_deccan.npz | NDVI monthly composites Jun-Sep 2018,2019,2020,2021,2022 on land fine cells | unitless -1..1 |
| lulc_fractions_deccan.npz | 6 per-class area fractions + dominant class (WorldCover 2021) | 0..1 |
| soilmoisture_daily_deccan.npz | OPTIONAL/PENDING (quota-blocked): daily ERA5-Land volumetric soil moisture 0-7 cm on the ERA5 lattice (bilinear to fine grid like the era5_* channels); resume via build_aux.py | m3/m3 |

### Soil unit conversion (SoilGrids d_factor=10)
sand/clay: g/kg -> % = value/10. ocd: dg/dm3 -> g/dm3 = value/10. phh2o: pH x10 -> pH = value/10.
bdod: kg/dm3 -> kg/m3 = value * 1000.

### LULC class legend (ESA WorldCover v200)
10 tree cover; 20 shrubland; 30 grassland; 40 cropland; 50 built-up; 60 bare/sparse;
70 snow/ice; 80 water; 90 wetland; 95 mangroves; 100 moss/lichen.
Fractions: forest = 10+95; grassland = 30+20; barren = 60+70+100.
Note: the 6 fractions do NOT sum to 1 in cells containing wetland (class 90)
pixels - wetland is tracked in `dominant_class` but has no fraction column
(104 of 47,250 covered cells, wetland share 0.02..0.32 in the Deccan build).
All-nodata stencils (ocean) are all-zero rows, so covered cells are
identifiable as fractions.sum(axis=1) > 0.

### NDVI composite
Per-pixel mean of valid (non-fill, in-range) values from 2 sampled days
(day 5 and 25 of each month); NaN = no valid
observation (persistent monsoon cloud). Per-month valid-pixel fractions are stored
in the file for QC.

### LULC processing note
ESA WorldCover v200 COG tiles (36000x36000 px at 10 m) are read with spatial
windowing (zarr chunked reads of only the 1024x1024 COG chunks overlapping the
region), 4096-pixel row strips; per-cell class fractions use the 600x600-pixel
stencil around each 0.05-deg cell. Cells whose whole stencil is WorldCover
nodata (class 0, ocean) get all-zero fractions - these are the sea fringe of
coastal IMD grid boxes and are never valid training targets (M = 0).

### Admin mapping caveat
Grid cells are assigned to the nearest subdistrict representative point
(KD-tree). A 200-cell point-in-polygon spot check quantifies the agreement
(see coverage report). Panchayat/village polygons are not available in GADM;
LGD provides them without a bulk API - the ID schema is LGD-joinable.
