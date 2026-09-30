import numpy as np
import pandas as pd
import json
from pathlib import Path

# Load admin map
admin_map = np.load('data/aux_data/admin/grid_admin_map_deccan.npz', allow_pickle=True)
fine_i = admin_map['fine_i']
fine_j = admin_map['fine_j']
subdistrict_idx = admin_map['subdistrict_idx']
subdistrict_names = admin_map['subdistrict_names']
district_names = admin_map['district_names']
state_names = admin_map['state_names']

# Load test predictions
pred = np.load('outputs/prediction_test.npz')
rainfall = pred['rainfall_mm']  # shape (n_dates, H, W)
dates = pred['dates']

# Load mask M_test
M_test = np.load('data/processed/M_test.npy')
if M_test.ndim == 4:
    M_test = M_test[:, 0]

n_dates, H, W = rainfall.shape

# Build a mapping from cell to subdistrict
# fine_i and fine_j are arrays of cell coordinates mapped to subdistrict_idx
unique_sub_indices = np.unique(subdistrict_idx)
valid_sub_mask = unique_sub_indices >= 0
unique_sub_indices = unique_sub_indices[valid_sub_mask]

records = []

print("Aggregating block rainfall...")
for d_idx, date in enumerate(dates):
    day_rain = rainfall[d_idx]
    day_mask = M_test[d_idx]
    
    # We aggregate cells where matching mask == 1
    for s_idx in unique_sub_indices:
        # find cells belonging to this subdistrict
        cell_mask = (subdistrict_idx == s_idx)
        cell_is, cell_js = fine_i[cell_mask], fine_j[cell_mask]
        
        # Get values and masks for these cells
        cell_vals = day_rain[cell_is, cell_js]
        cell_m = day_mask[cell_is, cell_js]
        
        # Keep only cells where mask equals 1
        valid_vals = cell_vals[cell_m == 1]
        
        if len(valid_vals) > 0:
            mean_rain = np.nanmean(valid_vals)
        else:
            mean_rain = np.nan
            
        records.append({
            'date': date,
            'state': state_names[s_idx] if s_idx < len(state_names) else 'Unknown',
            'district': district_names[s_idx] if s_idx < len(district_names) else 'Unknown',
            'block': subdistrict_names[s_idx] if s_idx < len(subdistrict_names) else 'Unknown',
            'subdistrict_idx': s_idx,
            'rainfall_mm': mean_rain
        })

df = pd.DataFrame(records)
out_dir = Path('outputs/layer2')
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / 'block_rainfall.csv'
df.to_csv(out_csv, index=False)
print(f"Wrote block rainfall CSV to {out_csv}")
