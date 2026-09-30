"""Compare candidate checkpoints + baselines.

Nominal event metrics use obs_thr == dec_thr (the reported convention).
'Calibrated' metrics fix the OBSERVATION threshold (>=50mm) and choose the
DECISION threshold on the VAL split, then apply it to TEST.
"""
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
OBS_THR = 50.0


def batch_predict(model, Xk, base, residual, rs=RS):
    out = []
    for i in range(0, Xk.shape[0], 16):
        out.append(predict_mm(model, Xk[i:i + 16], rs, "cpu",
                              baseline=(base[i:i + 16] if base is not None else None),
                              residual=residual))
    return np.concatenate(out, 0)


def load_model(name):
    ck = torch.load(HERE / "models" / f"{name}.pt", map_location="cpu", weights_only=False)
    chans = ck.get("channels") or ck["args"]["channels"]
    ci = [allc.index(c) for c in chans]
    residual = bool(ck.get("residual", True))
    m = SmallUNet(len(ci), int(ck.get("width", 16)), residual)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ci, residual


def prf(pred, obs, mask, dec_thr, obs_thr):
    """Precision/recall/F1 with obs hit defined by obs_thr and forecast by dec_thr."""
    o = (obs >= obs_thr) & mask
    p = (pred >= dec_thr) & mask
    tp = int((o & p).sum()); fp = int((~o & p).sum()); fn = int((o & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if (prec == prec and rec == rec and prec + rec > 0) else float("nan"))
    return {"p": prec, "r": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


preds = {}
for k in ("val", "test"):
    X = S[k]["X"]
    preds[k] = {"B1": np.clip(X[:, 0] * RS, 0, None)}
    for name in ("model_d", "model_e", "model_f", "model_c_weighted"):
        if not (HERE / "models" / f"{name}.pt").exists():
            continue
        m, ci, residual = load_model(name)
        preds[k][name] = batch_predict(m, np.ascontiguousarray(X[:, ci]),
                                       X[:, 0:1] if residual else None, residual)

out = {}
print(f"{'model':18s} {'split':5s} {'MAE':>6s} {'RMSE':>6s} {'corr':>6s} | "
      f"{'F1@10':>6s} {'F1@25':>6s} {'F1@50':>6s} | "
      f"{'P@50':>5s} {'R@50':>5s} | {'d*':>5s} {'F1_50cal':>8s}")
for name in preds["val"]:
    out[name] = {}
    # choose the decision threshold for the >=50 event on VAL
    Yv, Mv = S["val"]["Y"][:, 0], S["val"]["M"][:, 0].astype(bool)
    sweep = [(float(t), prf(preds["val"][name], Yv, Mv, t, OBS_THR)["f1"])
             for t in np.arange(5, 70.01, 1.0)]
    d_star = max(sweep, key=lambda x: (x[1] if x[1] == x[1] else -1))[0]
    for k in ("val", "test"):
        Y, M = S[k]["Y"][:, 0], S[k]["M"][:, 0]
        r = metrics(preds[k][name], Y, M)
        r.update(event_metrics(preds[k][name], Y, M))
        cal = prf(preds[k][name], Y, M.astype(bool), d_star, OBS_THR)
        nom = prf(preds[k][name], Y, M.astype(bool), OBS_THR, OBS_THR)
        out[name][k] = {"MAE": r["MAE"], "RMSE": r["RMSE"], "corr": r["corr"],
                        "f1_10": r[">=10mm"]["F1"], "f1_25": r[">=25mm"]["F1"],
                        "f1_50": r[">=50mm"]["F1"], "P_50": nom["p"], "R_50": nom["r"],
                        "dec_thr_star": d_star, "f1_50_calibrated": cal["f1"],
                        "P_50_cal": cal["p"], "R_50_cal": cal["r"]}
        print(f"{name:18s} {k:5s} {r['MAE']:6.2f} {r['RMSE']:6.2f} {r['corr']:6.3f} | "
              f"{r['>=10mm']['F1']:6.3f} {r['>=25mm']['F1']:6.3f} {r['>=50mm']['F1']:6.3f} | "
              f"{nom['p']:5.3f} {nom['r']:5.3f} | {d_star:5.1f} {cal['f1']:8.3f}")

json.dump(out, open(HERE / "qa" / "qa_f1_50_variants.json", "w"), indent=1, default=float)
print("\n-> qa_f1_50_variants.json")
print("\nNote: d* is the >=50mm DECISION threshold chosen on VAL to maximise VAL F1.")
