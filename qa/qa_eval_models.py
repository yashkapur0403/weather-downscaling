"""Evaluate candidate checkpoints on val+test with full event metrics."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent   # repo root (this script lives in qa/)
sys.path.insert(0, str(HERE / "scripts"))
import numpy as np, torch
from train import SmallUNet, load_data, predict_mm
from evaluate import evaluate_model

X, Y, M, meta = load_data()
rain_scale = float(meta["normalization"]["rain_scale_mm"])
channels = meta["channels"]
imd_idx = channels.index("imd_rain")

def batch_predict(model, Xk, rs, residual, base):
    out = []
    for i in range(0, Xk.shape[0], 16):
        out.append(predict_mm(model, Xk[i:i+16], rs, "cpu",
                              baseline=(base[i:i+16] if base is not None else None), residual=residual))
    return np.concatenate(out, 0)

def show(tag, split, r):
    f = " ".join(f"{r[f'>={t:g}mm']['F1']:.3f}" for t in (10, 25, 50))
    print(f"{tag:22s} {split:5s} MAE={r['MAE']:6.2f} RMSE={r['RMSE']:6.2f} corr={r['corr']:6.3f} F1(10/25/50)={f}")

print(f"{'model':22s} {'split':5s}  metrics")
for split in ("val", "test"):
    b1 = np.clip(X[split][:, imd_idx] * rain_scale, 0, None)
    show("A_bilinear", split, evaluate_model(b1, Y[split], M[split]))

for name in ("model_d", "model_e", "model_c_weighted"):
    p = HERE / "models" / f"{name}.pt"
    if not p.exists():
        print("missing", name); continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    chans = ck.get("channels") or ck["args"]["channels"]
    ci = [channels.index(ch) for ch in chans]
    residual = bool(ck.get("residual", True)); width = int(ck.get("width", 16))
    rs = float(ck.get("rain_scale", rain_scale))
    model = SmallUNet(len(ci), width, residual); model.load_state_dict(ck["model"]); model.eval()
    for split in ("val", "test"):
        Xk = np.ascontiguousarray(X[split][:, ci])
        base = np.ascontiguousarray(X[split][:, imd_idx:imd_idx+1]) if residual else None
        pred = batch_predict(model, Xk, rs, residual, base)
        show(name, split, evaluate_model(pred, Y[split], M[split]))
