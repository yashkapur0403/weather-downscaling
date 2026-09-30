"""
Ablation study: which information actually helps Layer 1?

Rows (each trained/evaluated on the SAME dates - the processed dataset):
  A  Bilinear IMD baseline          (no learning)
  B  U-Net, rainfall only           models/model_b.pt
  C  U-Net + DEM                    models/model_c.pt
  Cw U-Net + DEM, weighted loss     models/model_c_weighted.pt
  D  U-Net + DEM + ERA5-Land        models/model_d.pt

Model selection rule:
  Best VALIDATION MAE selects the headline model.
  If models are within 0.1 mm MAE, validation correlation breaks the tie.
  Test is reported once for the selected model.

Outputs:
  outputs/metrics/ablation.csv
  outputs/metrics/ablation.json
  outputs/metrics/ablation_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import torch  # noqa: E402

from evaluate import (  # noqa: E402
    THRESHOLDS,
    apply_bias,
    compute_bias_map,
    evaluate_model,
)
from train import SmallUNet, load_data, predict_mm  # noqa: E402


ROWS = [
    ("A", "Bilinear IMD baseline", None),
    ("B", "U-Net rainfall only", "model_b"),
    ("C", "U-Net + DEM", "model_c"),
    ("Cw", "U-Net + DEM (weighted loss)", "model_c_weighted"),
    ("D", "U-Net + DEM + ERA5-Land", "model_d"),
]


def load_checkpoint(path: Path, channels: list[str], rain_scale: float):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    chans = ckpt.get("channels")
    if chans is None:
        chans = ckpt.get("args", {}).get("channels")

    if not chans:
        raise ValueError(
            f"Checkpoint {path.name} does not contain channel information."
        )

    missing = [c for c in chans if c not in channels]
    if missing:
        raise ValueError(
            f"Checkpoint {path.name} requires channels not present "
            f"in processed dataset: {missing}"
        )

    ci = [channels.index(c) for c in chans]
    residual = bool(ckpt.get("residual", True))

    width = int(
        ckpt.get(
            "width",
            ckpt.get("args", {}).get("width", 16),
        )
    )

    model = SmallUNet(
        cin=len(ci),
        width=width,
        residual=residual,
    )

    model.load_state_dict(ckpt["model"])
    model.eval()

    checkpoint_rain_scale = float(
        ckpt.get("rain_scale", rain_scale)
    )

    return model, ci, residual, checkpoint_rain_scale


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default="val,test")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    valid_splits = {"train", "val", "test"}
    invalid = [s for s in splits if s not in valid_splits]
    if invalid:
        raise ValueError(
            f"Invalid split(s): {invalid}. "
            f"Choose from {sorted(valid_splits)}."
        )

    X, Y, M, meta = load_data()

    rain_scale = float(meta["normalization"]["rain_scale_mm"])
    fine_lat = np.array(meta["grid"]["fine_lat"])
    fine_lon = np.array(meta["grid"]["fine_lon"])
    channels = meta["channels"]

    if "imd_rain" not in channels:
        raise ValueError("Ablation requires the imd_rain channel.")

    imd_idx = channels.index("imd_rain")

    print(f"processed channels = {channels}")
    print(f"imd_rain index     = {imd_idx}")
    print(f"rain scale         = {rain_scale}")

    bias_map = compute_bias_map(
        X["train"], Y["train"], M["train"], meta
    )

    results = {k: {} for k in splits}

    for k in splits:
        b1 = np.clip(
            X[k][:, imd_idx] * rain_scale,
            0.0,
            None,
        )

        results[k]["A"] = evaluate_model(b1, Y[k], M[k])

        b2 = np.stack([
            apply_bias(
                b1[i],
                bias_map,
                fine_lat,
                fine_lon,
                meta,
            )
            for i in range(len(b1))
        ])

        b2 = np.clip(b2, 0.0, None)

        results[k]["B2 bias-corrected bilinear"] = evaluate_model(
            b2, Y[k], M[k]
        )

    loaded = {}

    print("\nLoading checkpoints...")

    for tag, label, ckpt_name in ROWS:
        if ckpt_name is None:
            continue

        path = config.MODELS / f"{ckpt_name}.pt"

        if not path.exists():
            print(f"[skip] {tag}: missing {path.name}")
            continue

        try:
            model, ci, residual, rs = load_checkpoint(
                path,
                channels,
                rain_scale,
            )

            loaded[tag] = (model, ci, residual, rs)

            print(
                f"[loaded] {tag}: {label} "
                f"channels={[channels[i] for i in ci]} "
                f"residual={residual}"
            )

        except Exception as e:
            print(f"[skip] {tag}: failed to load {path.name}: {e}")

    for k in splits:
        for tag, _label, _ckpt in ROWS:
            if tag not in loaded:
                continue

            model, ci, residual, rs = loaded[tag]

            Xk = np.ascontiguousarray(X[k][:, ci])

            if residual:
                base = np.ascontiguousarray(
                    X[k][:, imd_idx:imd_idx + 1]
                )
            else:
                base = None

            p = predict_mm(
                model,
                Xk,
                rs,
                "cpu",
                baseline=base,
                residual=residual,
            )

            results[k][tag] = evaluate_model(
                p,
                Y[k],
                M[k],
            )

    def fmt_row(tag, label, r):
        f1 = " ".join(
            f"{r[f'>={t:g}mm']['F1']:5.3f}"
            for t in THRESHOLDS
        )

        return (
            f"{tag:3s} "
            f"{label:32s} "
            f"{r['MAE']:6.2f} "
            f"{r['RMSE']:6.2f} "
            f"{r['corr']:6.3f}   "
            f"{f1}"
        )

    hdr = (
        f"{'':3s} {'model':32s} {'MAE':>6s} "
        f"{'RMSE':>6s} {'corr':>6s}   "
        + " ".join(f"F1>{int(t):<4d}" for t in THRESHOLDS)
    )

    print("\n" + hdr)
    print("-" * len(hdr))

    display_split = "val" if "val" in results else splits[0]

    for tag, label, _ in ROWS:
        if tag in results[display_split]:
            print(fmt_row(tag, label, results[display_split][tag]))

    if "B2 bias-corrected bilinear" in results[display_split]:
        print(
            fmt_row(
                "B2",
                "Bias-corrected bilinear",
                results[display_split]["B2 bias-corrected bilinear"],
            )
        )

    if "val" not in results:
        raise ValueError(
            "Validation results are required for model selection. "
            "Run with --splits val,test."
        )

    val = results["val"]

    learned = [
        tag
        for tag, _label, ckpt in ROWS
        if ckpt is not None and tag in val
    ]

    if not learned:
        raise RuntimeError(
            "No learned model checkpoints were found. "
            "Train model_b/model_c/model_c_weighted/model_d first."
        )

    mae_min = min(val[tag]["MAE"] for tag in learned)

    tied = [
        tag
        for tag in learned
        if val[tag]["MAE"] <= mae_min + 0.1
    ]

    best_tag = max(tied, key=lambda tag: val[tag]["corr"])

    labels = {tag: label for tag, label, _ in ROWS}
    best_label = labels[best_tag]

    print(
        "\n[val selection] "
        f"MAE-tied candidates {tied} -> {best_tag} = {best_label} "
        f"(val MAE {val[best_tag]['MAE']:.2f}, "
        f"corr {val[best_tag]['corr']:.3f})"
    )

    ckpt_of = {tag: ckpt for tag, _label, ckpt in ROWS}
    selected_ckpt = ckpt_of.get(best_tag)

    if selected_ckpt:
        src = config.MODELS / f"{selected_ckpt}.pt"
        dst = config.MODELS / "best_model.pt"
        shutil.copyfile(src, dst)
        print(f"best_model.pt <- {selected_ckpt}.pt")

    out_dir = config.OUT_METRICS
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "ablation.csv"

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)

        w.writerow([
            "row",
            "model",
            "split",
            "MAE_mm",
            "RMSE_mm",
            "corr",
            "F1_ge10mm",
            "F1_ge25mm",
            "F1_ge50mm",
        ])

        for k in splits:
            for tag, label, _ in ROWS:
                if tag not in results[k]:
                    continue

                r = results[k][tag]

                w.writerow(
                    [
                        tag,
                        label,
                        k,
                        f"{r['MAE']:.3f}",
                        f"{r['RMSE']:.3f}",
                        f"{r['corr']:.4f}",
                    ]
                    + [
                        f"{r[f'>={t:g}mm']['F1']:.3f}"
                        for t in THRESHOLDS
                    ]
                )

            if "B2 bias-corrected bilinear" in results[k]:
                r = results[k]["B2 bias-corrected bilinear"]

                w.writerow(
                    [
                        "B2",
                        "Bias-corrected bilinear",
                        k,
                        f"{r['MAE']:.3f}",
                        f"{r['RMSE']:.3f}",
                        f"{r['corr']:.4f}",
                    ]
                    + [
                        f"{r[f'>={t:g}mm']['F1']:.3f}"
                        for t in THRESHOLDS
                    ]
                )

    test_split = "test" if "test" in results else splits[-1]

    md = [
        "# Layer-1 ablation (same dates/split for every row)",
        "",
        "Reference: CHIRPS 0.05-deg "
        "(a reference product, not ground truth).",
        f"Split: {meta['split_description']}.",
        "Event F1 pooled over all valid fine pixels.",
        "",
        "| Row | Model | Val MAE | Val RMSE | Val Corr | "
        "Test MAE | Test RMSE | Test Corr |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for tag, label, _ in ROWS:
        if tag not in results[test_split]:
            continue

        rv = results.get("val", {}).get(tag)
        rt = results[test_split][tag]

        if rv:
            val_text = f"{rv['MAE']:.2f}"
            val_rmse = f"{rv['RMSE']:.2f}"
            val_corr = f"{rv['corr']:.3f}"
        else:
            val_text = val_rmse = val_corr = "-"

        md.append(
            f"| {tag} | {label} | {val_text} | {val_rmse} | "
            f"{val_corr} | {rt['MAE']:.2f} | "
            f"{rt['RMSE']:.2f} | {rt['corr']:.3f} |"
        )

    if "B2 bias-corrected bilinear" in results[test_split]:
        rt = results[test_split]["B2 bias-corrected bilinear"]
        rv = results.get("val", {}).get("B2 bias-corrected bilinear")

        if rv:
            bval = f"{rv['MAE']:.2f}"
            bval_rmse = f"{rv['RMSE']:.2f}"
            bval_corr = f"{rv['corr']:.3f}"
        else:
            bval = bval_rmse = bval_corr = "-"

        md.append(
            f"| B2 | Bias-corrected bilinear | {bval} | "
            f"{bval_rmse} | {bval_corr} | {rt['MAE']:.2f} | "
            f"{rt['RMSE']:.2f} | {rt['corr']:.3f} |"
        )

    md += [
        "",
        f"**Selected: {best_tag} - {best_label}.**",
        "",
    ]

    if "A" in results[test_split] and best_tag in results[test_split]:
        baseline_mae = results[test_split]["A"]["MAE"]
        selected_mae = results[test_split][best_tag]["MAE"]

        if baseline_mae > 0:
            improvement = (
                (baseline_mae - selected_mae)
                / baseline_mae
                * 100.0
            )

            md.append(
                f"Selected model vs bilinear baseline "
                f"(test MAE): {improvement:.1f}% change in MAE."
            )

    md += [
        "",
        "## Heavy-rain event F1 (test)",
        "",
        "| Row | F1>=10mm | F1>=25mm | F1>=50mm |",
        "|---|---:|---:|---:|",
    ]

    for tag, _label, _ in ROWS:
        if tag not in results[test_split]:
            continue

        r = results[test_split][tag]

        md.append(
            f"| {tag} | {r['>=10mm']['F1']:.3f} | "
            f"{r['>=25mm']['F1']:.3f} | {r['>=50mm']['F1']:.3f} |"
        )

    if "B2 bias-corrected bilinear" in results[test_split]:
        r = results[test_split]["B2 bias-corrected bilinear"]

        md.append(
            f"| B2 | {r['>=10mm']['F1']:.3f} | "
            f"{r['>=25mm']['F1']:.3f} | {r['>=50mm']['F1']:.3f} |"
        )

    md += [
        "",
        "Model selection uses validation MAE only, with a fixed "
        "0.1-mm tie window broken by validation correlation. "
        "Test data is not used for model selection.",
    ]

    summary_path = out_dir / "ablation_summary.md"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    json_path = out_dir / "ablation.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "rows": {
                    tag: {
                        "label": label,
                        "checkpoint": ckpt,
                    }
                    for tag, label, ckpt in ROWS
                },
                "selection": {
                    "criterion": "validation MAE",
                    "tie_window_mm": 0.1,
                    "tie_breaker": "validation correlation",
                    "row": best_tag,
                    "label": best_label,
                },
                "results": results,
                "meta": {
                    "roi": meta["roi"],
                    "split": meta["split_description"],
                    "reference": (
                        "CHIRPS v2.0 0.05-deg "
                        "(reference, not ground truth)"
                    ),
                },
            },
            f,
            indent=1,
        )

    print(
        f"\nablation -> {csv_path} | "
        f"{summary_path} | {json_path}"
    )


if __name__ == "__main__":
    main()
