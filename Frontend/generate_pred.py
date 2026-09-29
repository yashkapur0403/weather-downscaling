import json
import torch
import numpy as np
from pathlib import Path
import sys

# add scripts to path so we can import model
sys.path.insert(0, 'scripts')
from train import SmallUNet
import config

print("Loading meta...")
with open('data/processed/meta.json') as f:
    meta = json.load(f)

print("Loading model...")
ckpt = torch.load('models/best_model.pt', map_location='cpu', weights_only=False)
channels = ckpt.get("channels") or ckpt["args"]["channels"]
rain_scale = float(ckpt.get("rain_scale", 100.0))
width = int(ckpt.get("width", 16))
residual = bool(ckpt.get("residual", True))

model = SmallUNet(cin=len(channels), width=width, residual=residual)
model.load_state_dict(ckpt["model"])
model.eval()

print("Loading data...")
X_test = np.load('data/processed/X_test.npy')
M_test = np.load('data/processed/M_test.npy')
dates = meta['split']['test']['dates']
lat = meta['grid']['fine_lat']
lon = meta['grid']['fine_lon']

print(f"X_test shape: {X_test.shape}")
assert X_test.shape[0] == len(dates)

print("Predicting...")
base = None
if residual:
    i0 = channels.index("imd_rain")
    base = torch.from_numpy(X_test[:, i0:i0 + 1].copy())

with torch.no_grad():
    out = model(torch.from_numpy(X_test), baseline=base).numpy()

pred_mm = np.clip(out[:, 0] * rain_scale, 0.0, None)
# Ensure proper shape (n_days, H, W)
if M_test.ndim == 4:
    M_test = M_test[:, 0]
pred_mm = np.where(M_test == 1, pred_mm, np.nan)

out_file = 'outputs/prediction_test.npz'
Path('outputs').mkdir(exist_ok=True)
np.savez_compressed(out_file, dates=np.array(dates),
                    latitude=lat, longitude=lon,
                    rainfall_mm=pred_mm.astype("float32"))

print(f"Saved {out_file}")
