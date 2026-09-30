"""QA: independently reproduce Layer-1 test metrics from data.zip + checkpoint.

Replicates scripts/evaluate.py::evaluate_model + event_metrics and
scripts/train.py::metrics exactly (same masking, thresholds, units).
"""
import json, os, sys, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # repo root (this script lives in qa/)
ROOT = HERE                              # repo root (data.zip is here)
os.chdir(HERE)
sys.path.insert(0, str(HERE / "scripts"))

import numpy as np
import torch
from train import SmallUNet, predict_mm

THRESHOLDS = (10.0, 25.0, 50.0)

def event_metrics(pred_mm, y_mm, m):
    out = {}
    m = m.astype(bool)
    for thr in THRESHOLDS:
        obs = (y_mm >= thr) & m
        prd = (pred_mm >= thr) & m
        tp = int((obs & prd).sum()); fp = int((~obs & prd).sum()); fn = int((obs & ~prd).sum())
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if (precision == precision and recall == recall and (precision + recall) > 0) else float("nan"))
        out[f">={thr:g}mm"] = {"precision": precision, "recall": recall, "F1": f1, "n_events": tp + fn}
    return out

def metrics(pred_mm, y_mm, m):
    m = m.astype(bool)
    if m.sum() == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "corr": float("nan")}
    e = pred_mm[m] - y_mm[m]
    out = {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2)))}
    a, b = pred_mm[m], y_mm[m]
    out["corr"] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    return out

def evaluate_model(pred_mm, Y, M):
    y = Y[:, 0]; m = M[:, 0]
    res = metrics(pred_mm, y, m); res.update(event_metrics(pred_mm, y, m))
    return res

# ---- extract test arrays from data.zip (repo root) ----
out_dir = HERE / "qa" / "qa_processed"
out_dir.mkdir(exist_ok=True)
zpath = ROOT / "data.zip"
need = ["data/processed/X_test.npy", "data/processed/Y_test.npy",
        "data/processed/M_test.npy", "data/processed/meta.json"]
with zipfile.ZipFile(zpath) as z:
    have = set(z.namelist())
    for n in need:
        target = out_dir / Path(n).name
        if not target.exists():
            print("extracting", n, flush=True)
            with z.open(n) as src, open(target, "wb") as dst:
                dst.write(src.read())
        else:
            print("cached", target.name)

X_test = np.load(out_dir / "X_test.npy")
Y_test = np.load(out_dir / "Y_test.npy")
M_test = np.load(out_dir / "M_test.npy")
meta_zip = json.loads((out_dir / "meta.json").read_text())
print("X_test", X_test.shape, X_test.dtype, "Y_test", Y_test.shape, "M_test", M_test.shape)
channels = meta_zip["channels"]; rain_scale = float(meta_zip["normalization"]["rain_scale_mm"])
imd_idx = channels.index("imd_rain")
print("channels", channels, "rain_scale", rain_scale)

# sanity: zip meta matches repo meta (same data)
repo_meta = json.loads((HERE / "data/processed/meta.json").read_text())
print("zip meta test dates == repo meta test dates:",
      meta_zip["split"]["test"]["dates"] == repo_meta["split"]["test"]["dates"])
print("M_test valid per day (unique):", np.unique(M_test[:, 0].astype(bool).sum(axis=(1, 2))))

results = {}

# ---- row A: bilinear IMD baseline ----
b1 = np.clip(X_test[:, imd_idx] * rain_scale, 0.0, None)
results["A"] = evaluate_model(b1, Y_test, M_test)

# ---- row D: best_model.pt (selected) ----
ck = torch.load("models/best_model.pt", map_location="cpu", weights_only=False)
chans = ck.get("channels") or ck["args"]["channels"]
ci = [channels.index(c) for c in chans]
residual = bool(ck.get("residual", True)); width = int(ck.get("width", 16))
rs = float(ck.get("rain_scale", rain_scale))
model = SmallUNet(cin=len(ci), width=width, residual=residual)
model.load_state_dict(ck["model"]); model.eval()
print("loaded D: channels", chans, "ci", ci, "residual", residual, "width", width, "rs", rs)

Xk = np.ascontiguousarray(X_test[:, ci])
base = np.ascontiguousarray(X_test[:, imd_idx:imd_idx + 1]) if residual else None
preds = []
CH = 16
for i in range(0, Xk.shape[0], CH):
    p = predict_mm(model, Xk[i:i+CH], rs, "cpu", baseline=(base[i:i+CH] if base is not None else None), residual=residual)
    preds.append(p)
pred_D = np.concatenate(preds, axis=0)
results["D"] = evaluate_model(pred_D, Y_test, M_test)

print(json.dumps(results, indent=1))

# ---- compare to committed ablation.json ----
ab = json.loads((HERE / "outputs/metrics/ablation.json").read_text())
print("\n== reproduced vs committed (test) ==")
for row in ("A", "D"):
    rep = results[row]; com = ab["results"]["test"][row]
    for k in ("MAE", "RMSE", "corr"):
        d = abs(rep[k] - com[k])
        print(f"{row} {k}: reproduced={rep[k]:.6f} committed={com[k]:.6f} absdiff={d:.2e}")
    for thr in ("10", "25", "50"):
        kk = f">={thr}mm"
        print(f"{row} F1@{thr}: reproduced={rep[kk]['F1']:.6f} committed={com[kk]['F1']:.6f} "
              f"n_events rep={rep[kk]['n_events']} com={com[kk]['n_events']}")
