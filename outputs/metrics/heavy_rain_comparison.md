# Heavy-Rain Preservation Experiments

Baseline: **Model D** (U-Net + DEM + ERA5, plain MAE loss).
All models share the same architecture (SmallUNet, width=16, residual=True)
and channel set (imd_rain, dem, era5_t2m, era5_t2m_max, era5_dewp).
Test split: 2022 Jun–Sep (122 days). Metrics over all valid land pixels.

| Row | Model | Val MAE | Test MAE | Test RMSE | Test Corr | F1≥10mm | F1≥25mm | F1≥50mm | ΔMAE vs D | ΔF1≥25 vs D | ΔF1≥50 vs D |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MODEL_D | D  U-Net+DEM+ERA5 (baseline, mae loss) | 6.93 | 8.17 | 16.01 | 0.446 | 0.393 | 0.281 | 0.015 | +0.00 | +0.000 | +0.000 |
| MODEL_E | E  U-Net+DEM+ERA5 (log1p loss) | 6.39 | 7.52 | 15.24 | 0.499 | 0.562 | 0.105 | 0.000 | -0.64 | -0.176 | -0.014 |
| MODEL_F | F  U-Net+DEM+ERA5 (combined gentle ramp) | 6.79 | 7.76 | 14.72 | 0.518 | 0.558 | 0.279 | 0.026 | -0.41 | -0.002 | +0.011 |
| MODEL_G | G  U-Net+DEM+ERA5 (extreme-rain spike) | 26.94 | 26.98 | 34.88 | 0.396 | 0.563 | 0.307 | 0.128 | +18.81 | +0.026 | +0.113 |
| MODEL_H | H  U-Net+DEM+ERA5 (log1p+combined ramp) | 6.69 | 7.53 | 14.27 | 0.548 | 0.598 | 0.386 | 0.034 | -0.63 | +0.105 | +0.019 |
| MODEL_I | I  U-Net+DEM+ERA5 (log1p+combined+extreme) | 26.51 | 26.99 | 35.10 | 0.400 | 0.569 | 0.306 | 0.131 | +18.82 | +0.025 | +0.116 |
| MODEL_J | J  U-Net+DEM+ERA5 (weighted MAE) | 6.91 | 7.67 | 14.17 | 0.552 | 0.610 | 0.430 | 0.127 | -0.49 | +0.149 | +0.113 |

## Notes on loss design

| ID | Loss | Design goal |
|---|---|---|
| D | `mae` | Baseline: minimize masked MAE in normalized space |
| E | `log1p` | Train in log1p(mm) space; penalises relative under-prediction of heavy events |
| F | `combined` | α·plain_MAE + (1-α)·ramped_weighted_MAE; gentle linear ramp 10→50 mm |
| G | `extreme` | plain_MAE + spike term (×8) for pixels ≥ 50 mm |
| H | `log1p_combined` | log1p space with the same ramp as F; combines E+F |
| I | `log1p_combined_extreme` | H plus the explicit extreme-event term in log space |
| J | `weighted` | hard 3x weighting at >=25 mm, using all D channels |

Loss constants (config.py): COMBINED_ALPHA=0.6, COMBINED_MULT_LO=1.5,
COMBINED_MULT_HI=4.0, EXTREME_THR_MM=50, EXTREME_MULT=8.

## Reproducibility

Run `python scripts/run_heavy_rain_experiments.py --skip-existing` from the Layer-1 project root.
All candidates use the D channels, residual SmallUNet width 16, seed 42, batch 16,
patch 48, learning rate 0.001, up to 300 epochs, and patience 40.
Per-checkpoint recorded arguments:

| Model | Loss | Seed | Epochs | Patience | Best validation loss |
|---|---|---:|---:|---:|---:|
| MODEL_D | mae | 42 | 220 | 40 | 0.0544243 |
| MODEL_E | log1p | 42 | 300 | 40 | 0.876895 |
| MODEL_F | combined | 42 | 300 | 40 | 0.0827794 |
| MODEL_G | extreme | 42 | 300 | 40 | 1.90132 |
| MODEL_H | log1p_combined | 42 | 300 | 40 | 0.932087 |
| MODEL_I | log1p_combined_extreme | 42 | 300 | 40 | 3.61507 |
| MODEL_J | weighted | 42 | 300 | 40 | 0.0928902 |
