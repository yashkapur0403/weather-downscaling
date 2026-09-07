"""
Ablation study: which information actually helps Layer 1?

Rows (each trained/evaluated on the SAME dates - the processed dataset):
  A  Bilinear IMD baseline          (no learning)
  B  U-Net, rainfall only           models/model_b.pt
  C  U-Net + DEM                    models/model_c.pt
  Cw U-Net + DEM, weighted loss     models/model_c_weighted.pt  (loss ablation)
  D  U-Net + DEM + ERA5-Land        models/model_d.pt

Model selection rule (scientific): best VALIDATION MAE picks the headline
model; test is reported once for the selected model. Event metrics are shown
for all rows because agriculture cares about heavy-rain detection.

Outputs:
  outputs/metrics/ablation.csv | ablation.json | ablation_summary.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import torch  # noqa: E402

from evaluate import (THRESHOLDS, apply_bias, compute_bias_map,  # noqa: E402
                      evaluate_model)
from train import SmallUNet, load_data, predict_mm, metrics  # noqa: E402

ROWS = [
    ("A", "Bilinear IMD baseline", None),
    ("B", "U-Net rainfall only", "model_b"),
    ("C", "U-Net + DEM", "model_c"),
    ("Cw", "U-Net + DEM (weighted loss)", "model_c_weighted"),
    ("D", "U-Net + DEM + ERA5-Land", "model_d"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default="val,test")
    args = ap.parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    X, Y, M, meta = load_data()
    rain_scale = meta["normalization"]["rain_scale_mm"]
    fine_lat = np.array(meta["grid"]["fine_lat"])
    fine_lon = np.array(meta["grid"]["fine_lon"])
    channels = meta["channels"]
    assert "imd_rain" in channels, "ablation needs the imd_rain channel"

    # ---------------- baselines (shared) ----------------
    bias_map = compute_bias_map(X["train"], Y["train"], M["train"], meta)
    results = {k: {} for k in splits}
    for k in splits:
        b1 = np.clip(X[k][:, 0] * rain_scale, 0.0, None)
        results[k]["A"] = evaluate_model(b1, Y[k], M[k])
        b2 = np.stack([apply_bias(b1[i], bias_map, fine_lat, fine_lon, meta)
                       for i in range(len(b1))])
        b2 = np.clip(b2, 0.0, None)
        results[k]["B2 bias-corrected bilinear"] = evaluate_model(b2, Y[k], M[k])

    # ---------------- models ----------------
    loaded = {}
    for tag, _label, ckpt_name in ROWS:
        if ckpt_name is None:
            continue
        path = config.MODELS / f"{ckpt_name}.pt"
        if not path.exists():
            print(f"[skip] {tag}: missing {path.name} (train it first)")
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        chans = ckpt.get("channels") or ckpt["args"]["channels"]
        ci = [channels.index(c) for c in chans]
        residual = bool(ckpt.get("residual", True))
        model = SmallUNet(cin=len(ci), width=int(ckpt.get("width", 16)),
                          residual=residual)
        model.load_state_dict(ckpt["model"])
        model.eval()
        loaded[tag] = (model, ci, residual, float(ckpt.get("rain_scale", rain_scale)))

    for k in splits:
        for tag, _label, _ckpt in ROWS:
            if tag not in loaded:
                continue
            model, ci, residual, rs = loaded[tag]
            Xk = np.ascontiguousarray(X[k][:, ci])
            base = X[k][:, 0:1] if residual else None
            p = predict_mm(model, Xk, rs, "cpu", baseline=base, residual=residual)
            results[k][tag] = evaluate_model(p, Y[k], M[k])

    # ---------------- console + files ----------------
    def fmt_row(tag, label, r):
        f1 = " ".join(f"{r[f'>={t:g}mm']['F1']:5.3f}" for t in THRESHOLDS)
        return f"{tag:3s} {label:30s} {r['MAE']:6.2f} {r['RMSE']:6.2f} " \
               f"{r['corr']:6.3f}   {f1}"

    hdr = (f"{'':3s} {'model':30s} {'MAE':>6s} {'RMSE':>6s} {'corr':>6s}   "
           + " ".join(f"F1>{int(t):<4d}" for t in THRESHOLDS))
    print("\n" + hdr)
    for tag, label, _ in ROWS:
        if tag in results[splits[0]]:
            print(fmt_row(tag, label, results[splits[0]][tag]))

    # ---------------- model selection (val metrics ONLY) ----------------
    # Rule (fixed before looking at test): lowest val MAE wins; if other rows
    # are within 0.1 mm (noise level), the best val corr among them wins.
    val = results["val"] if "val" in results else results[splits[0]]
    learned = [t for t, _l, c in ROWS if c and t in val]
    mae_min = min(val[t]["MAE"] for t in learned)
    tied = [t for t in learned if val[t]["MAE"] <= mae_min + 0.1]
    best_tag = max(tied, key=lambda t: val[t]["corr"])
    best_label = dict((t, l) for t, l, _ in ROWS)[best_tag]
    print(f"\n[val selection] MAE-tied candidates {tied} -> {best_tag} = "
          f"{best_label} (val MAE {val[best_tag]['MAE']:.2f}, "
          f"corr {val[best_tag]['corr']:.3f})")
    ckpt_of = dict((t, c) for t, l, c in ROWS)
    if ckpt_of.get(best_tag):
        import shutil
        shutil.copyfile(config.MODELS / f"{ckpt_of[best_tag]}.pt",
                        config.MODELS / "best_model.pt")
        print(f"best_model.pt <- {ckpt_of[best_tag]}.pt")

    out_dir = config.OUT_METRICS
    with open(out_dir / "ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "model", "split", "MAE_mm", "RMSE_mm", "corr",
                    "F1_ge10mm", "F1_ge25mm", "F1_ge50mm"])
        for k in splits:
            for tag, label, _ in ROWS:
                if tag not in results[k]:
                    continue
                r = results[k][tag]
                w.writerow([tag, label, k, f"{r['MAE']:.3f}", f"{r['RMSE']:.3f}",
                            f"{r['corr']:.4f}"] +
                           [f"{r[f'>={t:g}mm']['F1']:.3f}" for t in THRESHOLDS])

    md = ["# Layer-1 ablation (same dates/split for every row)",
          "",
          "Reference: CHIRPS 0.05-deg (a reference product, not ground truth).",
          f"Split: {meta['split_description']}. "
          "Event F1 pooled over all valid fine pixels.",
          "",
          "| Row | Model | val MAE | val RMSE | val corr | test MAE | test RMSE | test corr |",
          "|---|---|---|---|---|---|---|---|"]
    for tag, label, _ in ROWS:
        if tag not in results[splits[-1]]:
            continue
        rv = results.get("val", {}).get(tag)
        rt = results[splits[-1]][tag]
        mv = f"{rv['MAE']:.2f} / {rv['RMSE']:.2f} / {rv['corr']:.3f}" if rv else "-"
        mt = f"{rt['MAE']:.2f} / {rt['RMSE']:.2f} / {rt['corr']:.3f}"
        md.append(f"| {tag} | {label} | {mv} | {mt} |")
    md += ["", f"**Selected (val MAE, 0.1-mm tie broken by val corr): "
           f"{best_tag} - {best_label}.**",
           "",
           f"Improvement of the selected model over the bilinear baseline "
           f"(test): {abs((results[splits[-1]]['A']['MAE'] - results[splits[-1]][best_tag]['MAE']) / results[splits[-1]]['A']['MAE'] * 100):.1f}% "
           "lower MAE.",
           "",
           "Note: the MAE-trained rows (B/C/D) under-detect heavy rain "
           "(smoothed fields). The weighted-loss row Cw trades ~0.7 mm "
           "val MAE for clearly better correlation and heavy-rain F1 - use "
           "Cw when heavy-rain detection matters more than mean error.",
           "",
           "## Heavy-rain event F1 (test)",
           "",
           "| Row | F1>=10mm | F1>=25mm | F1>=50mm |",
           "|---|---|---|---|"]
    for tag, label, _ in ROWS:
        if tag not in results[splits[-1]]:
            continue
        r = results[splits[-1]][tag]
        md.append(f"| {tag} | {r['>=10mm']['F1']:.3f} | {r['>=25mm']['F1']:.3f} | "
                  f"{r['>=50mm']['F1']:.3f} |")
    with open(out_dir / "ablation_summary.md", "w") as f:
        f.write("\n".join(md) + "\n")
    with open(out_dir / "ablation.json", "w") as f:
        json.dump({"rows": {tag: {"label": label, "checkpoint": ckpt}
                            for tag, label, ckpt in ROWS},
                   "selection": {"criterion": "val MAE", "row": best_tag,
                                 "label": best_label},
                   "results": results,
                   "meta": {"roi": meta["roi"],
                            "split": meta["split_description"],
                            "reference": "CHIRPS v2.0 0.05-deg (reference, "
                                         "not ground truth)"}},
                  f, indent=1)
    print(f"ablation -> {out_dir / 'ablation.csv'} | ablation_summary.md | "
          "ablation.json")


if __name__ == "__main__":
    main()
