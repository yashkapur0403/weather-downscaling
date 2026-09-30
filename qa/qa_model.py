"""QA: independently validate the Layer-1 checkpoint (needs torch)."""
import json, sys, os
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # repo root (this script lives in qa/)
os.chdir(HERE)
sys.path.insert(0, str(HERE / "scripts"))

import torch
import numpy as np
from train import SmallUNet
import config

def jprint(label, obj):
    print(f"{label}: {json.dumps(obj, default=str)}")

ck = torch.load("models/best_model.pt", map_location="cpu", weights_only=False)
print("== checkpoint top-level keys ==")
print(sorted(ck.keys()))
channels = ck.get("channels") or ck["args"]["channels"]
rain_scale = float(ck.get("rain_scale", 100.0))
width = int(ck.get("width", 16))
residual = bool(ck.get("residual", True))
jprint("channels", channels)
jprint("rain_scale", rain_scale)
jprint("width", width)
jprint("residual", residual)

# args (training hyperparameters) if present
if "args" in ck:
    a = ck["args"]
    keep = {k: a[k] for k in ("loss", "epochs", "lr", "batch_size", "seed",
                              "weight_rain_mm", "weight_mult", "early_stop_patience",
                              "channels", "width", "residual") if k in a}
    jprint("args(subset)", keep)

sd = ck["model"]
print("\n== state_dict ==")
print("n tensors:", len(sd))
# head weight must NOT be all zeros (else the model is an untrained pure baseline)
hw = sd["head.weight"]
print("head.weight shape:", tuple(hw.shape), "all_zero:", bool((hw == 0).all()),
      "absmax:", float(hw.abs().max()))
print("head.bias:", float(sd["head.bias"].item()))

model = SmallUNet(cin=len(channels), width=width, residual=residual)
missing, unexpected = model.load_state_dict(sd, strict=True), None
n_params = sum(p.numel() for p in model.parameters())
print("\n== parameter count ==")
print("instantiated SmallUNet params:", n_params)
print("checkpoint state_dict tensors:", len(sd))

# Replicate backend routes_data.unet_param_count independently
def unet_param_count(cin, w):
    return (9*cin*w + w) + w + (9*w*(2*w) + 2*w) + (9*(2*w)*(4*w) + 4*w) \
        + (4*w*2*w*2*2 + 2*w) + (9*(4*w)*(2*w) + 2*w) \
        + (2*w*w*2*2 + w) + (9*(2*w)*w + w) + (w*1 + 1)
print("unet_param_count(cin=%d,width=%d) from routes_data formula:" % (len(channels), width),
      unet_param_count(len(channels), width))

print("\n== config cross-check ==")
for k in ("CHANNELS", "WIDTH", "RESIDUAL", "RAIN_SCALE_MM", "REGION_DEFAULT"):
    if hasattr(config, k):
        jprint("config." + k, getattr(config, k))

print("\n== history (best epoch) ==")
h = json.load(open("models/best_model_history.json"))
print("history keys:", list(h.keys()))
if "history" in h:
    hist = h["history"]
    best = min(hist, key=lambda r: r.get("val_mae", 1e9))
    jprint("best_val_row", best)
    jprint("n_epochs", len(hist))
print("\n== reported metrics vs checkpoint ==")
ab = json.load(open("outputs/metrics/ablation.json"))
jprint("selection", ab.get("selection"))
res = ab.get("results", {})
for split, d in res.items():
    jprint("results." + split, d)
mj = json.load(open("outputs/metrics/metrics.json"))
print("metrics.json keys:", list(mj.keys()))
