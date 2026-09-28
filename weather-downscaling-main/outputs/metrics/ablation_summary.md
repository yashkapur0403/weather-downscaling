# Layer-1 ablation (same dates/split for every row)

Reference: CHIRPS 0.05-deg (a reference product, not ground truth).
Split: years {'train': [2019, 2020], 'val': [2021], 'test': [2022]}. Event F1 pooled over all valid fine pixels.

| Row | Model | val MAE | val RMSE | val corr | test MAE | test RMSE | test corr |
|---|---|---|---|---|---|---|---|
| A | Bilinear IMD baseline | 6.72 / 15.51 / 0.257 | 7.67 / 16.93 / 0.304 |
| B | U-Net rainfall only | 5.35 / 15.11 / 0.283 | 6.63 / 16.88 / 0.304 |
| C | U-Net + DEM | 5.39 / 14.98 / 0.289 | 6.58 / 16.69 / 0.314 |
| Cw | U-Net + DEM (weighted loss) | 6.04 / 14.39 / 0.372 | 6.91 / 15.60 / 0.419 |
| D | U-Net + DEM + ERA5-Land | 5.41 / 14.57 / 0.331 | 6.45 / 16.04 / 0.376 |

**Selected (val MAE, 0.1-mm tie broken by val corr): D - U-Net + DEM + ERA5-Land.**

Improvement of the selected model over the bilinear baseline (test): 15.9% lower MAE.

Note: the MAE-trained rows (B/C/D) under-detect heavy rain (smoothed fields). The weighted-loss row Cw trades ~0.7 mm val MAE for clearly better correlation and heavy-rain F1 - use Cw when heavy-rain detection matters more than mean error.

## Heavy-rain event F1 (test)

| Row | F1>=10mm | F1>=25mm | F1>=50mm |
|---|---|---|---|
| A | 0.413 | 0.217 | 0.112 |
| B | 0.186 | 0.055 | 0.003 |
| C | 0.229 | 0.061 | 0.003 |
| Cw | 0.374 | 0.312 | 0.181 |
| D | 0.319 | 0.146 | 0.043 |
