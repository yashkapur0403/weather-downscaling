"""
Evaluate baselines vs models: MAE / RMSE / corr, heavy-rain event metrics,
% improvement over baseline, difference maps, and machine-readable prediction
files for Layer 2.

Baselines
  B1  bilinear IMD            : IMD 0.25 bilinearly upsampled (channel imd_rain)
  B2  bias-corrected bilinear : B1 + per-coarse-cell mean bias
        bias[c] = mean over TRAIN days of (CHIRPS_coarse - IMD_coarse) at cell c,
        added on the fine grid (bias map bilinearly upsampled).
        Computed from training data only -> no leakage.

Event metrics: pooled over all valid fine pixels of a split, thresholds
10 / 25 / 50 mm/day -> precision, recall, F1 (agriculture cares about hits).

Wording rule: improvements are reported as "X% lower MAE than ...", never
"accuracy = X%".

Outputs
  outputs/metrics/metrics.csv|json       full metric tables
  outputs/figures/comparison_<date>.png  5-panel: IMD, B1, model, CHIRPS, error
  outputs/maps/<kind>_<date>.png         individual maps (same colour scale)
  prediction/<model>_<split>.npz         rainfall on the fine grid for Layer 2
  prediction/<model>_<split>_schema.json field documentation + one sample
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
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from grids import aggregate_to_coarse, bilinear_upsample  # noqa: E402
from train import SmallUNet, load_data, predict_mm, metrics  # noqa: E402

THRESHOLDS = (10.0, 25.0, 50.0)


def event_metrics(pred_mm, y_mm, m):
    out = {}
    m = m.astype(bool)
    for thr in THRESHOLDS:
        obs = (y_mm >= thr) & m
        prd = (pred_mm >= thr) & m
        tp = int((obs & prd).sum())
        fp = int((~obs & prd).sum())
        fn = int((obs & ~prd).sum())
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if precision == precision and recall == recall and
              (precision + recall) > 0 else float("nan"))
        out[f">={thr:g}mm"] = {"precision": precision, "recall": recall, "F1": f1,
                               "n_events": tp + fn}
    return out


def compute_bias_map(X_train, Y_train, M_train, meta):
    """Per-coarse-cell mean bias (CHIRPS - IMD), from the TRAIN split only."""
    sub = meta["grid"]["fine_sub"]
    y = Y_train[:, 0]
    m = M_train[:, 0].astype(bool)
    imd = X_train[:, 0]
    # aggregate both to the coarse grid, then average the bias over valid days
    n_days = len(y)
    Hf, Wf = y.shape[1:]
    H, W = Hf // sub, Wf // sub
    bias = np.zeros((H, W), dtype="float64")
    cnt = np.zeros((H, W), dtype="float64")
    for t in range(n_days):
        yc = aggregate_to_coarse(np.where(m[t], y[t], np.nan), sub)
        ic = aggregate_to_coarse(imd[t], sub)
        d = yc - ic
        d = np.nan_to_num(d, nan=0.0)
        w = (np.isfinite(yc) & np.isfinite(ic)).astype("float64")
        bias += d * w
        cnt += w
    bias = np.where(cnt > 0, bias / np.maximum(cnt, 1), 0.0)
    return bias.astype("float32")  # (H, W) coarse cells


def apply_bias(b1_fine_day, bias_map, fine_lat, fine_lon, meta):
    """Add the coarse bias map (bilinearly upsampled) to a fine baseline day."""
    sub = meta["grid"]["fine_sub"]
    # fine grid -> coarse grid is exact block structure; use nearest upsample of
    # the bias map: repeat each coarse cell value over its sub x sub block
    up = np.repeat(np.repeat(bias_map, sub, axis=0), sub, axis=1)
    h = min(up.shape[0], b1_fine_day.shape[0])
    w = min(up.shape[1], b1_fine_day.shape[1])
    out = b1_fine_day.copy()
    out[:h, :w] += up[:h, :w]
    return out


def panel(ax, field, fine_lat, fine_lon, title, vmax=None, diff=False):
    if diff:
        im = ax.imshow(field, origin="lower",
                       extent=[fine_lon[0], fine_lon[-1], fine_lat[0], fine_lat[-1]],
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
    else:
        im = ax.imshow(field, origin="lower",
                       extent=[fine_lon[0], fine_lon[-1], fine_lat[0], fine_lat[-1]],
                       cmap="YlGnBu", vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("lon [E]")
    ax.set_ylabel("lat [N]")
    return im


def evaluate_model(pred_mm, Y, M):
    y, m = Y[:, 0], M[:, 0]
    res = metrics(pred_mm, y, m)
    res.update(event_metrics(pred_mm, y, m))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="best_model",
                    help="comma list of checkpoint names under models/ "
                         "(without .pt)")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--showcase-date", default=None,
                    help="date for the comparison figure (default: wettest "
                         "test day)")
    args = ap.parse_args()
    model_names = [s.strip() for s in args.models.split(",") if s.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    X, Y, M, meta = load_data()
    rain_scale = meta["normalization"]["rain_scale_mm"]
    fine_lat = np.array(meta["grid"]["fine_lat"])
    fine_lon = np.array(meta["grid"]["fine_lon"])
    dates = meta["dates"]
    channels = meta["channels"]
    has_imd = "imd_rain" in channels
    assert has_imd, "bilinear baselines require the imd_rain channel"

    # ---------- baselines ----------
    bias_map = compute_bias_map(X["train"], Y["train"], M["train"], meta)
    print(f"[baseline2] per-cell bias range {bias_map.min():+.2f}.."
          f"{bias_map.max():+.2f} mm (train-derived)")

    def baselines_for(k):
        b1 = np.clip(X[k][:, 0] * rain_scale, 0.0, None)
        b2 = np.stack([apply_bias(b1[i], bias_map, fine_lat, fine_lon, meta)
                       for i in range(len(b1))])
        b2 = np.clip(b2, 0.0, None)
        return {"B1 bilinear IMD": b1, "B2 bias-corrected bilinear": b2}

    # ---------- models ----------
    models = {}
    for name in model_names:
        ckpt_path = config.MODELS / f"{name}.pt"
        if not ckpt_path.exists():
            print(f"[skip] missing checkpoint {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        chans = ckpt.get("channels") or ckpt["args"]["channels"]
        ci = [channels.index(c) for c in chans]
        residual = bool(ckpt.get("residual", True))
        model = SmallUNet(cin=len(ci), width=int(ckpt.get("width", 16)),
                          residual=residual)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models[name] = {"model": model, "ci": ci, "residual": residual,
                        "rain_scale": float(ckpt.get("rain_scale", rain_scale))}
        print(f"[model] {name}: channels={chans} residual={residual}")

    results = {}
    preds = {}
    for k in splits:
        results[k] = {}
        bl = baselines_for(k)
        for bname, bpred in bl.items():
            results[k][bname] = evaluate_model(bpred, Y[k], M[k])
            preds.setdefault(bname, {})[k] = bpred
        for name, md in models.items():
            Xk = np.ascontiguousarray(X[k][:, md["ci"]])
            base = X[k][:, 0:1] if md["residual"] else None
            p = predict_mm(md["model"], Xk, md["rain_scale"], "cpu",
                           baseline=base, residual=md["residual"])
            results[k][f"U-Net [{name}]"] = evaluate_model(p, Y[k], M[k])
            preds.setdefault(f"U-Net [{name}]", {})[k] = p

        # ---------- console table ----------
        i0, i1 = meta["split"][k]["indices"]
        n_days_k = i1 - i0 + 1
        print(f"\n=== {k.upper()} ({n_days_k} days) ===")
        hdr = (f"{'model':30s} {'MAE':>6s} {'RMSE':>6s} {'corr':>6s} "
               + "".join(f"{'F1>' + str(int(t)) + 'mm':>9s}" for t in THRESHOLDS))
        print(hdr)
        base_mae = results[k]["B1 bilinear IMD"]["MAE"]
        for name, r in results[k].items():
            f1s = "".join(f"{r[f'>={t:g}mm']['F1']:9.3f}" for t in THRESHOLDS)
            print(f"{name:30s} {r['MAE']:6.2f} {r['RMSE']:6.2f} "
                  f"{r['corr']:6.3f} {f1s}")

    # ---------- improvement summary (scientific wording) ----------
    print("\n=== improvement over B1 (bilinear IMD), TEST split ===")
    tk = "test" if "test" in results else splits[-1]
    b1r = results[tk]["B1 bilinear IMD"]
    b2r = results[tk]["B2 bias-corrected bilinear"]
    print(f"  B2 bias-corrected: MAE {abs((b1r['MAE'] - b2r['MAE']) / b1r['MAE'] * 100):.1f}% "
          f"{'lower' if b2r['MAE'] < b1r['MAE'] else 'HIGHER'} than B1")
    for name in results[tk]:
        if name.startswith("U-Net"):
            r = results[tk][name]
            print(f"  {name}: MAE {abs((b1r['MAE'] - r['MAE']) / b1r['MAE'] * 100):.1f}% "
                  f"{'lower' if r['MAE'] < b1r['MAE'] else 'HIGHER'} than B1; "
                  f"{abs((b2r['MAE'] - r['MAE']) / b2r['MAE'] * 100):.1f}% "
                  f"{'lower' if r['MAE'] < b2r['MAE'] else 'HIGHER'} than B2")

    # ---------- metrics files ----------
    with open(config.OUT_METRICS / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "model", "MAE_mm", "RMSE_mm", "corr"])
        for k in splits:
            for name, r in results[k].items():
                w.writerow([k, name, f"{r['MAE']:.3f}", f"{r['RMSE']:.3f}",
                            f"{r['corr']:.4f}"])
        for k in splits:
            for name, r in results[k].items():
                for thr, ev in r.items():
                    if thr.startswith(">="):
                        w.writerow([k, f"{name} events", thr,
                                    f"prec={ev['precision']:.3f}",
                                    f"rec={ev['recall']:.3f} F1={ev['F1']:.3f}"])
    with open(config.OUT_METRICS / "metrics.json", "w") as f:
        json.dump({"meta": {"roi": meta["roi"], "dates": meta["dates"],
                            "rain_scale_mm": rain_scale,
                            "channels": channels,
                            "split": meta["split"],
                            "reference": "CHIRPS v2.0 0.05-deg (a REFERENCE "
                                         "product, not ground truth)",
                            "event_thresholds_mm": list(THRESHOLDS)},
                   "results": results}, f, indent=1)
    print(f"\nmetrics -> {config.OUT_METRICS / 'metrics.csv'}")

    # ---------- prediction files for Layer 2 ----------
    pred_dir = config.ROOT / "prediction"
    pred_dir.mkdir(exist_ok=True)
    for name, per_split in preds.items():
        for k, p in per_split.items():
            slug = name.replace(" ", "_").replace("[", "").replace("]", "")
            npz_path = pred_dir / f"{slug}_{k}.npz"
            np.savez_compressed(
                npz_path,
                dates=np.array([dates[i] for i in
                                range(meta["split"][k]["indices"][0],
                                      meta["split"][k]["indices"][1] + 1)]),
                latitude=fine_lat, longitude=fine_lon,
                rainfall_mm=p.astype("float32"))
            schema = {
                "file": npz_path.name,
                "description": "Layer-1 downscaled daily rainfall on the fine "
                               "grid; ready for Panchayat-polygon intersection "
                               "(Layer 2). Zarr/GeoTIFF export can be added "
                               "without changing consumers.",
                "fields": {
                    "dates": "YYYY-MM-DD per sample",
                    "latitude": "fine-grid lat centers (deg N, ascending)",
                    "longitude": "fine-grid lon centers (deg E, ascending)",
                    "rainfall_mm": "(n_days, lat, lon) daily rainfall mm/day",
                },
                "grid": {"n_lat": len(fine_lat), "n_lon": len(fine_lon),
                         "resolution_deg": 0.05,
                         "roi": meta["roi"]},
                "sample_first_date": {
                    "date": dates[meta["split"][k]["indices"][0]],
                    "rainfall_mm_shape": list(p[0].shape),
                    "domain_mean_mm": float(np.nanmean(p[0])),
                },
            }
            with open(pred_dir / f"{slug}_{k}_schema.json", "w") as f:
                json.dump(schema, f, indent=1)
    print(f"predictions -> {pred_dir}/ (npz + schema per model/split)")

    # ---------- showcase figure + maps ----------
    tk = "test" if "test" in splits else splits[0]
    if args.showcase_date:
        show = dates.index(args.showcase_date) - meta["split"][tk]["indices"][0]
    else:
        yv = [np.nanmean(Y[tk][i, 0][M[tk][i, 0].astype(bool)])
              if M[tk][i, 0].astype(bool).any() else 0.0
              for i in range(len(Y[tk]))]
        show = int(np.argmax(yv))
    d = dates[meta["split"][tk]["indices"][0] + show]
    y_show = np.where(M[tk][show, 0].astype(bool), Y[tk][show, 0], np.nan)
    vmax = float(np.nanpercentile(y_show[np.isfinite(y_show)], 99)) \
        if np.isfinite(y_show).any() else 10.0
    panels = [(X[tk][show, 0] * rain_scale, "IMD 0.25 bilinear (input)"),
              (preds["B1 bilinear IMD"][tk][show], "B1: bilinear IMD"),
              (preds["B2 bias-corrected bilinear"][tk][show],
               "B2: bias-corrected bilinear")]
    for name in models:
        panels.append((preds[f"U-Net [{name}]"][tk][show],
                       f"U-Net [{name}] (IMD+DEM"
                       f"{'+ERA5' if 'era5_t2m' in models[name]['ci'] and False else ''})"))
    panels.append((y_show, "CHIRPS 0.05 (reference)"))
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4.6 * (len(panels) + 1), 4.6))
    for ax, (fld, ttl) in zip(axes, panels):
        panel(ax, fld, fine_lat, fine_lon, ttl, vmax=vmax)
    err = preds[f"U-Net [{model_names[-1]}]"][tk][show] - y_show
    panel(axes[-1], err, fine_lat, fine_lon,
          f"error: U-Net [{model_names[-1]}] - CHIRPS", vmax=max(vmax, 1.0),
          diff=True)
    fig.suptitle(f"Rainfall downscaling - {d} (mm/day)", fontsize=12)
    fig.colorbar(axes[0].images[0], ax=axes, shrink=0.85, label="mm/day")
    out = config.OUT_FIGS / f"comparison_{d}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out}")

    # individual maps, same colour scale
    for fld, tag in [(X[tk][show, 0] * rain_scale, "imd_coarse"),
                     (preds["B1 bilinear IMD"][tk][show], "baseline_b1"),
                     (preds["B2 bias-corrected bilinear"][tk][show], "baseline_b2"),
                     (y_show, "chirps_ref"),
                     (err, "error_unet_vs_chirps")] + \
            [(preds[f"U-Net [{n}]"][tk][show], f"unet_{n}") for n in models]:
        fig, ax = plt.subplots(figsize=(6, 5))
        if tag.startswith("error"):
            im = panel(ax, fld, fine_lat, fine_lon, f"{tag} {d}", vmax=max(vmax, 1.0),
                       diff=True)
            fig.colorbar(im, ax=ax, label="mm/day (model - reference)")
        else:
            im = panel(ax, fld, fine_lat, fine_lon, f"{tag} {d}", vmax=vmax)
            fig.colorbar(im, ax=ax, label="mm/day")
        fig.savefig(config.OUT_MAPS / f"{tag}_{d}.png", dpi=140,
                    bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
