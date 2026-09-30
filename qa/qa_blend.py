"""Test blends of model_e (MAE-optimal) and model_f (heavy-rain weighted)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # repo root (this script lives in qa/)
sys.path.insert(0, str(HERE / "scripts"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from train import SmallUNet, predict_mm, metrics  # noqa: E402
from evaluate import event_metrics  # noqa: E402

meta = json.load(open(HERE / "data" / "processed" / "meta.json"))
RS = float(meta["normalization"]["rain_scale_mm"])
allc = meta["channels"]
S = {k: {a: np.load(HERE / "data" / "processed" / f"{a}_{k}.npy")
         for a in ("X", "Y", "M")} for k in ("val", "test")}


def batch_predict(model, Xk, base, residual):
    out = []
    for i in range(0, Xk.shape[0], 16):
        out.append(predict_mm(model, Xk[i:i + 16], RS, "cpu",
                              baseline=(base[i:i + 16] if base is not None else None),
                              residual=residual))
    return np.concatenate(out, 0)


def load(name):
    ck = torch.load(HERE / "models" / f"{name}.pt", map_location="cpu", weights_only=False)
    ci = [allc.index(c) for c in ck["channels"]]
    residual = bool(ck.get("residual", True))
    m = SmallUNet(len(ci), int(ck.get("width", 16)), residual)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ci, residual


P = {}
for k in ("val", "test"):
    X = S[k]["X"]
    P[k] = {"A_B1": np.clip(X[:, 0] * RS, 0, None)}
    pe = pf = None
    for nm in ("model_e", "model_f"):
        m, ci, res = load(nm)
        p = batch_predict(m, np.ascontiguousarray(X[:, ci]), X[:, 0:1] if res else None, res)
        P[k][nm] = p
        if nm == "model_e":
            pe = p
        else:
            pf = p
    P[k]["mean_w50"] = 0.5 * pe + 0.5 * pf
    P[k]["mean_w65"] = 0.65 * pe + 0.35 * pf
    P[k]["max"] = np.maximum(pe, pf)
    P[k]["quantile_hi"] = np.maximum(pe, 0.5 * (pe + pf) + 0.5 * (pf - pe))
    P[k]["rank_blend"] = np.where(pf > pe, pe + 0.5 * (pf - pe), pe)


print(f"{'candidate':12s} {'split':5s} {'MAE':>6s} {'RMSE':>6s} {'corr':>6s} | "
      f"{'F1@10':>6s} {'F1@25':>6s} {'F1@50':>6s}")
out = {}
for name in P["val"]:
    out[name] = {}
    for k in ("val", "test"):
        Y, M = S[k]["Y"][:, 0], S[k]["M"][:, 0]
        r = metrics(P[k][name], Y, M); r.update(event_metrics(P[k][name], Y, M))
        out[name][k] = {kk: r[kk] if kk in ("MAE", "RMSE", "corr") else None
                        for kk in ("MAE", "RMSE", "corr")}
        out[name][k].update({f"f1_{t:g}": r[f">={t:g}mm"]["F1"] for t in (10, 25, 50)})
        print(f"{name:12s} {k:5s} {r['MAE']:6.2f} {r['RMSE']:6.2f} {r['corr']:6.3f} | "
              f"{r['>=10mm']['F1']:6.3f} {r['>=25mm']['F1']:6.3f} {r['>=50mm']['F1']:6.3f}")

json.dump(out, open(HERE / "qa" / "qa_blend.json", "w"), indent=1, default=float)
print("\n-> qa_blend.json")
