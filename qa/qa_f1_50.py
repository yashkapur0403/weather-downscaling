"""Diagnose F1@>=50mm and test cheap fixes (decision-threshold calibration).

No training. Loads model_e (and baselines) on val/test, reports
precision/recall at the nominal 50 mm threshold, then finds the decision
threshold that maximises VAL F1 and reports the resulting TEST F1.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # repo root (this script lives in qa/)
sys.path.insert(0, str(HERE / "scripts"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from train import SmallUNet, predict_mm  # noqa: E402

RS = 100.0
S = {k: {a: np.load(HERE / "data" / "processed" / f"{a}_{k}.npy")
         for a in ("X", "Y", "M")} for k in ("val", "test")}


def f1_at(pred, obs, mask, thr):
    prd = (pred >= thr) & mask
    obs = (obs >= thr) & mask
    tp = int((obs & prd).sum()); fp = int((~obs & prd).sum()); fn = int((obs & ~prd).sum())
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if (p == p and r == r and p + r > 0) else float("nan")
    return {"p": p, "r": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def batch_predict(model, Xk, base, residual, rs=RS):
    out = []
    for i in range(0, Xk.shape[0], 16):
        out.append(predict_mm(model, Xk[i:i + 16], rs, "cpu",
                              baseline=(base[i:i + 16] if base is not None else None),
                              residual=residual))
    return np.concatenate(out, 0)


ck = torch.load(HERE / "models" / "model_e.pt", map_location="cpu", weights_only=False)
chans = ck.get("channels")
allc = json.load(open(HERE / "data" / "processed" / "meta.json"))["channels"]
ci = [allc.index(c) for c in chans]
residual = bool(ck.get("residual", True)); width = int(ck.get("width", 16))
model = SmallUNet(len(ci), width, residual); model.load_state_dict(ck["model"]); model.eval()

preds = {}
for k in ("val", "test"):
    X, Y, M = S[k]["X"], S[k]["Y"], S[k]["M"]
    preds[k] = {
        "B1": np.clip(X[:, 0] * RS, 0, None),
        "E": batch_predict(model, np.ascontiguousarray(X[:, ci]),
                           X[:, 0:1] if residual else None, residual),
    }

print("=== nominal 50 mm decision threshold (as reported) ===")
for k in ("val", "test"):
    Y, M = S[k]["Y"][:, 0], S[k]["M"][:, 0].astype(bool)
    for name in ("B1", "E"):
        d = f1_at(preds[k][name], Y, M, 50.0)
        print(f"{k:5s} {name:3s} thr=50.0  P={d['p']:.3f} R={d['r']:.3f} "
              f"F1={d['f1']:.3f}  tp={d['tp']} fp={d['fp']} fn={d['fn']}")

print("\n=== where does E put truth>=50 pixels? (val+test pooled finite) ===")
for k in ("val", "test"):
    Y, M = S[k]["Y"][:, 0], S[k]["M"][:, 0].astype(bool)
    sel = (Y >= 50) & M
    pv = preds[k]["E"][sel]
    ov = np.clip(S[k]["X"][:, 0] * RS, 0, None)[sel]
    print(f"{k:5s} n_obs_ge50={sel.sum():6d}  E: mean={pv.mean():6.2f} "
          f"med={np.median(pv):6.2f} max={pv.max():6.2f}  frac(E>=50)={np.mean(pv >= 50):.3f}")
    print(f"{'':5s} {'':6s}  B1: mean={ov.mean():6.2f} med={np.median(ov):6.2f} "
          f"frac(B1>=50)={np.mean(ov >= 50):.3f}")

print("\n=== decision-threshold sweep on VAL -> applied to TEST ===")
Yv, Mv = S["val"]["Y"][:, 0], S["val"]["M"][:, 0].astype(bool)
Yt, Mt = S["test"]["Y"][:, 0], S["test"]["M"][:, 0].astype(bool)
best = {}
for name in ("B1", "E"):
    rows = []
    for thr in np.arange(20, 60.01, 1.0):
        dv = f1_at(preds["val"][name], Yv, Mv, thr)
        rows.append((thr, dv["f1"]))
    thr_star = max(rows, key=lambda t: (t[1] if t[1] == t[1] else -1))[0]
    dv = f1_at(preds["val"][name], Yv, Mv, thr_star)
    dt = f1_at(preds["test"][name], Yt, Mt, thr_star)
    best[name] = {"thr_star": float(thr_star), "val_f1": dv["f1"], "test_f1": dt["f1"],
                  "test_p": dt["p"], "test_r": dt["r"],
                  "test_at_50": f1_at(preds["test"][name], Yt, Mt, 50.0)["f1"]}
    print(f"{name}: thr*={thr_star:5.1f}  val_F1={dv['f1']:.3f}  "
          f"TEST F1={dt['f1']:.3f} (P={dt['p']:.3f} R={dt['r']:.3f})  "
          f"vs test F1@50={best[name]['test_at_50']:.3f}")

print("\n=== multiplicative calibration pred*k, k chosen on VAL ===")
for name in ("E",):
    rows = []
    for kk in np.arange(0.7, 1.81, 0.05):
        dv = f1_at(preds["val"][name] * kk, Yv, Mv, 50.0)
        rows.append((float(kk), dv["f1"]))
    kk_star = max(rows, key=lambda t: (t[1] if t[1] == t[1] else -1))[0]
    dv = f1_at(preds["val"][name] * kk_star, Yv, Mv, 50.0)
    dt = f1_at(preds["test"][name] * kk_star, Yt, Mt, 50.0)
    print(f"{name}: k*={kk_star:.2f}  val_F1={dv['f1']:.3f}  TEST F1={dt['f1']:.3f} "
          f"(P={dt['p']:.3f} R={dt['r']:.3f})")

json.dump(best, open(HERE / "qa" / "qa_f1_50.json", "w"), indent=1)
print("\n-> qa_f1_50.json")
