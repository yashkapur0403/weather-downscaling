# Layer-1 ablation (same dates/split for every row)

Reference: CHIRPS 0.05-deg (a reference product, not ground truth).
Split: years {'train': [2018, 2019, 2020], 'val': [2021], 'test': [2022]}. Event F1 pooled over all valid fine pixels.

| Row | Model | val MAE | val RMSE | val corr | test MAE | test RMSE | test corr |
|---|---|---|---|---|---|---|---|
| A | Bilinear IMD baseline | 8.55 / 16.33 / 0.288 | 9.60 / 18.18 / 0.385 |
| B | U-Net rainfall only | 7.13 / 14.52 / 0.293 | 8.51 / 16.93 / 0.396 |
| C | U-Net + DEM | 7.04 / 14.16 / 0.318 | 8.36 / 16.50 / 0.419 |
| Cw | U-Net + DEM (weighted loss) | 11.40 / 18.80 / 0.226 | 12.27 / 20.11 / 0.287 |
| D | U-Net + DEM + ERA5-Land | 6.93 / 13.79 / 0.346 | 8.17 / 16.01 / 0.446 |
| E | U-Net + DEM + ERA5-Land (heavy-rain weighted) | 7.44 / 12.82 / 0.460 | 8.21 / 14.57 / 0.518 |

**Selected (val MAE, 0.1-mm tie broken by val corr): E - U-Net + DEM + ERA5-Land (heavy-rain weighted).**

Improvement of the selected model over the bilinear baseline (test): 14.5% lower MAE.

Note: the MAE-trained rows (B/C/D) under-detect heavy rain (smoothed fields). The weighted-loss row Cw trades ~0.7 mm val MAE for clearly better correlation and heavy-rain F1 - use Cw when heavy-rain detection matters more than mean error.

## Heavy-rain event F1 (test)

| Row | F1>=10mm | F1>=25mm | F1>=50mm |
|---|---|---|---|
| A | 0.477 | 0.343 | 0.234 |
| B | 0.281 | 0.226 | 0.115 |
| C | 0.332 | 0.231 | 0.104 |
| Cw | 0.440 | 0.305 | 0.188 |
| D | 0.393 | 0.281 | 0.015 |
| E | 0.602 | 0.425 | 0.129 |
