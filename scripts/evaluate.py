from pathlib import Path

content = r'''"""
Evaluate baselines vs models: MAE / RMSE / corr, heavy-rain event metrics,
% improvement over baseline, difference maps, and machine-readable prediction
files for Layer 2.

Baselines
  B1  bilinear IMD            : IMD 0.25 bilinearly upsampled
  B2  bias-corrected bilinear : B1 + per-coarse-cell mean bias

Event metrics: pooled over all valid fine pixels of a split,
thresholds 10 / 25 / 50 mm/day.

Outputs
  outputs/metrics/metrics.csv|json
  outputs/figures/comparison_<date>.png
  outputs/maps/<kind>_<date>.png
  prediction/<model>_<split>.npz
  prediction/<model>_<split>_schema.json
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

        precision = (
            tp / (tp + fp)
            if (tp + fp)
            else float("nan")
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn)
            else float("nan")
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (
                precision == precision
                and recall == recall
                and (precision + recall) > 0
            )
            else float("nan")
        )

        out[f">={thr:g}mm"] = {
            "precision": precision,
            "recall": recall,
            "F1": f1,
            "n_events": tp + fn,
        }

    return out


def compute_bias_map(X_train, Y_train, M_train, meta):
    """Per-coarse-cell mean bias (CHIRPS - IMD), from TRAIN split only."""
    sub = meta["grid"]["fine_sub"]

    y = Y_train[:, 0]
    m = M_train[:, 0].astype(bool)

    # Explicitly locate IMD rather than assuming it is always channel 0.
    channels = meta["channels"]
    imd_idx = channels.index("imd_rain")
    imd = X_train[:, imd_idx]

    n_days = len(y)

    Hf, Wf = y.shape[1:]
    H, W = Hf // sub, Wf // sub

    bias = np.zeros((H, W), dtype="float64")
    cnt = np.zeros((H, W), dtype="float64")

    for t in range(n_days):
        yc = aggregate_to_coarse(
            np.where(m[t], y[t], np.nan),
            sub,
        )

        ic = aggregate_to_coarse(
            imd[t],
            sub,
        )

        d = yc - ic

        valid = np.isfinite(yc) & np.isfinite(ic)

        bias += np.where(valid, d, 0.0)
        cnt += valid.astype("float64")

    bias = np.where(
        cnt > 0,
        bias / np.maximum(cnt, 1),
        0.0,
    )

    return bias.astype("float32")


def apply_bias(b1_fine_day, bias_map, fine_lat, fine_lon, meta):
    """Add the coarse bias map to a fine baseline day."""
    sub = meta["grid"]["fine_sub"]

    # The coarse grid maps exactly onto sub x sub fine-grid blocks.
    up = np.repeat(
        np.repeat(bias_map, sub, axis=0),
        sub,
        axis=1,
    )

    h = min(up.shape[0], b1_fine_day.shape[0])
    w = min(up.shape[1], b1_fine_day.shape[1])

    out = b1_fine_day.copy()
    out[:h, :w] += up[:h, :w]

    return out


def panel(
    ax,
    field,
    fine_lat,
    fine_lon,
    title,
    vmax=None,
    diff=False,
):
    if diff:
        im = ax.imshow(
            field,
            origin="lower",
            extent=[
                fine_lon[0],
                fine_lon[-1],
                fine_lat[0],
                fine_lat[-1],
            ],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
    else:
        im = ax.imshow(
            field,
            origin="lower",
            extent=[
                fine_lon[0],
                fine_lon[-1],
                fine_lat[0],
                fine_lat[-1],
            ],
            cmap="YlGnBu",
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("lon [E]")
    ax.set_ylabel("lat [N]")

    return im


def evaluate_model(pred_mm, Y, M):
    y = Y[:, 0]
    m = M[:, 0]

    res = metrics(pred_mm, y, m)
    res.update(event_metrics(pred_mm, y, m))

    return res


def load_model_checkpoint(
    ckpt_path,
    channels,
    default_rain_scale,
):
    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    chans = ckpt.get("channels")

    if chans is None:
        chans = ckpt.get("args", {}).get("channels")

    if not chans:
        raise ValueError(
            f"Checkpoint {ckpt_path.name} does not contain channels."
        )

    missing = [
        c for c in chans
        if c not in channels
    ]

    if missing:
        raise ValueError(
            f"{ckpt_path.name} requires missing channels: {missing}"
        )

    ci = [
        channels.index(c)
        for c in chans
    ]

    residual = bool(
        ckpt.get("residual", True)
    )

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

    model.load_state_dict(
        ckpt["model"]
    )

    model.eval()

    rs = float(
        ckpt.get(
            "rain_scale",
            default_rain_scale,
        )
    )

    return model, ci, residual, rs, chans


def main():
    ap = argparse.ArgumentParser(
        description=__doc__
    )

    ap.add_argument(
        "--models",
        default="best_model",
        help=(
            "comma list of checkpoint names under models/ "
            "(without .pt)"
        ),
    )

    ap.add_argument(
        "--splits",
        default="val,test",
    )

    ap.add_argument(
        "--showcase-date",
        default=None,
        help=(
            "date for the comparison figure "
            "(default: wettest test day)"
        ),
    )

    args = ap.parse_args()

    model_names = [
        s.strip()
        for s in args.models.split(",")
        if s.strip()
    ]

    splits = [
        s.strip()
        for s in args.splits.split(",")
        if s.strip()
    ]

    valid_splits = {"train", "val", "test"}

    invalid_splits = [
        s for s in splits
        if s not in valid_splits
    ]

    if invalid_splits:
        raise ValueError(
            f"Invalid split(s): {invalid_splits}"
        )

    X, Y, M, meta = load_data()

    rain_scale = float(
        meta["normalization"]["rain_scale_mm"]
    )

    fine_lat = np.array(
        meta["grid"]["fine_lat"]
    )

    fine_lon = np.array(
        meta["grid"]["fine_lon"]
    )

    dates = meta["dates"]
    channels = meta["channels"]

    if "imd_rain" not in channels:
        raise ValueError(
            "Bilinear baselines require the imd_rain channel."
        )

    imd_idx = channels.index("imd_rain")

    print(
        f"channels={channels}"
    )

    print(
        f"imd_rain index={imd_idx}"
    )

    print(
        f"rain_scale={rain_scale}"
    )

    # ---------------------------------------------------------
    # Baselines
    # ---------------------------------------------------------

    bias_map = compute_bias_map(
        X["train"],
        Y["train"],
        M["train"],
        meta,
    )

    print(
        f"[baseline2] per-cell bias range "
        f"{bias_map.min():+.2f}.."
        f"{bias_map.max():+.2f} mm "
        f"(train-derived)"
    )

    def baselines_for(k):
        b1 = np.clip(
            X[k][:, imd_idx] * rain_scale,
            0.0,
            None,
        )

        b2 = np.stack(
            [
                apply_bias(
                    b1[i],
                    bias_map,
                    fine_lat,
                    fine_lon,
                    meta,
                )
                for i in range(len(b1))
            ]
        )

        b2 = np.clip(
            b2,
            0.0,
            None,
        )

        return {
            "B1 bilinear IMD": b1,
            "B2 bias-corrected bilinear": b2,
        }

    # ---------------------------------------------------------
    # Models
    # ---------------------------------------------------------

    models = {}

    for name in model_names:
        ckpt_path = (
            config.MODELS / f"{name}.pt"
        )

        if not ckpt_path.exists():
            print(
                f"[skip] missing checkpoint "
                f"{ckpt_path}"
            )
            continue

        try:
            (
                model,
                ci,
                residual,
                rs,
                chans,
            ) = load_model_checkpoint(
                ckpt_path,
                channels,
                rain_scale,
            )
        except Exception as e:
            print(
                f"[skip] failed loading "
                f"{ckpt_path.name}: {e}"
            )
            continue

        models[name] = {
            "model": model,
            "ci": ci,
            "residual": residual,
            "rain_scale": rs,
            "channels": chans,
        }

        print(
            f"[model] {name}: "
            f"channels={chans} "
            f"residual={residual}"
        )

    if not models:
        print(
            "[warning] No model checkpoints were loaded. "
            "Only baselines will be evaluated."
        )

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    results = {}
    preds = {}

    for k in splits:
        results[k] = {}

        bl = baselines_for(k)

        for bname, bpred in bl.items():
            results[k][bname] = evaluate_model(
                bpred,
                Y[k],
                M[k],
            )

            preds.setdefault(
                bname,
                {},
            )[k] = bpred

        for name, md in models.items():
            Xk = np.ascontiguousarray(
                X[k][:, md["ci"]]
            )

            if md["residual"]:
                base = np.ascontiguousarray(
                    X[k][:, imd_idx:imd_idx + 1]
                )
            else:
                base = None

            p = predict_mm(
                md["model"],
                Xk,
                md["rain_scale"],
                "cpu",
                baseline=base,
                residual=md["residual"],
            )

            results[k][
                f"U-Net [{name}]"
            ] = evaluate_model(
                p,
                Y[k],
                M[k],
            )

            preds.setdefault(
                f"U-Net [{name}]",
                {},
            )[k] = p

        # -----------------------------------------------------
        # Console table
        # -----------------------------------------------------

        i0, i1 = meta["split"][k]["indices"]
        n_days_k = i1 - i0 + 1

        print(
            f"\n=== {k.upper()} "
            f"({n_days_k} days) ==="
        )

        hdr = (
            f"{'model':32s} "
            f"{'MAE':>7s} "
            f"{'RMSE':>7s} "
            f"{'corr':>7s} "
            + "".join(
                f"{'F1>' + str(int(t)) + 'mm':>10s}"
                for t in THRESHOLDS
            )
        )

        print(hdr)
        print("-" * len(hdr))

        for name, r in results[k].items():
            f1s = "".join(
                f"{r[f'>={t:g}mm']['F1']:10.3f}"
                for t in THRESHOLDS
            )

            print(
                f"{name:32s} "
                f"{r['MAE']:7.2f} "
                f"{r['RMSE']:7.2f} "
                f"{r['corr']:7.3f} "
                f"{f1s}"
            )

    # ---------------------------------------------------------
    # Improvement summary
    # ---------------------------------------------------------

    print(
        "\n=== improvement over B1 "
        "(bilinear IMD), TEST split ==="
    )

    tk = (
        "test"
        if "test" in results
        else splits[-1]
    )

    b1r = results[tk][
        "B1 bilinear IMD"
    ]

    b2r = results[tk][
        "B2 bias-corrected bilinear"
    ]

    def pct_change(reference, value):
        if reference == 0:
            return float("nan")

        return (
            (reference - value)
            / reference
            * 100.0
        )

    b2_change = pct_change(
        b1r["MAE"],
        b2r["MAE"],
    )

    if b2_change >= 0:
        print(
            f"  B2 bias-corrected: "
            f"MAE {b2_change:.1f}% lower than B1"
        )
    else:
        print(
            f"  B2 bias-corrected: "
            f"MAE {abs(b2_change):.1f}% higher than B1"
        )

    for name, r in results[tk].items():
        if not name.startswith("U-Net"):
            continue

        change_b1 = pct_change(
            b1r["MAE"],
            r["MAE"],
        )

        change_b2 = pct_change(
            b2r["MAE"],
            r["MAE"],
        )

        relation_b1 = (
            "lower"
            if change_b1 >= 0
            else "higher"
        )

        relation_b2 = (
            "lower"
            if change_b2 >= 0
            else "higher"
        )

        print(
            f"  {name}: "
            f"MAE {abs(change_b1):.1f}% "
            f"{relation_b1} than B1; "
            f"{abs(change_b2):.1f}% "
            f"{relation_b2} than B2"
        )

    # ---------------------------------------------------------
    # Output directories
    # ---------------------------------------------------------

    config.OUT_METRICS.mkdir(
        parents=True,
        exist_ok=True,
    )

    config.OUT_FIGS.mkdir(
        parents=True,
        exist_ok=True,
    )

    config.OUT_MAPS.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Metrics CSV
    # ---------------------------------------------------------

    metrics_csv = (
        config.OUT_METRICS / "metrics.csv"
    )

    with open(
        metrics_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)

        w.writerow([
            "split",
            "model",
            "metric_type",
            "metric",
            "value",
        ])

        for k in splits:
            for name, r in results[k].items():
                w.writerow([
                    k,
                    name,
                    "continuous",
                    "MAE_mm",
                    f"{r['MAE']:.3f}",
                ])

                w.writerow([
                    k,
                    name,
                    "continuous",
                    "RMSE_mm",
                    f"{r['RMSE']:.3f}",
                ])

                w.writerow([
                    k,
                    name,
                    "continuous",
                    "corr",
                    f"{r['corr']:.4f}",
                ])

                for thr in THRESHOLDS:
                    ev = r[f">={thr:g}mm"]

                    w.writerow([
                        k,
                        name,
                        "event",
                        f">={thr:g}mm_precision",
                        f"{ev['precision']:.4f}",
                    ])

                    w.writerow([
                        k,
                        name,
                        "event",
                        f">={thr:g}mm_recall",
                        f"{ev['recall']:.4f}",
                    ])

                    w.writerow([
                        k,
                        name,
                        "event",
                        f">={thr:g}mm_F1",
                        f"{ev['F1']:.4f}",
                    ])

                    w.writerow([
                        k,
                        name,
                        "event",
                        f">={thr:g}mm_n_events",
                        ev["n_events"],
                    ])

    # ---------------------------------------------------------
    # Metrics JSON
    # ---------------------------------------------------------

    metrics_json = (
        config.OUT_METRICS / "metrics.json"
    )

    with open(
        metrics_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "meta": {
                    "roi": meta["roi"],
                    "dates": meta["dates"],
                    "rain_scale_mm": rain_scale,
                    "channels": channels,
                    "split": meta["split"],
                    "reference": (
                        "CHIRPS v2.0 0.05-deg "
                        "(a REFERENCE product, "
                        "not ground truth)"
                    ),
                    "event_thresholds_mm": list(
                        THRESHOLDS
                    ),
                },
                "results": results,
            },
            f,
            indent=1,
        )

    print(
        f"\nmetrics -> {metrics_csv}"
    )

    # ---------------------------------------------------------
    # Prediction files for Layer 2
    # ---------------------------------------------------------

    pred_dir = (
        config.ROOT / "prediction"
    )

    pred_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, per_split in preds.items():
        for k, p in per_split.items():

            slug = (
                name
                .replace(" ", "_")
                .replace("[", "")
                .replace("]", "")
            )

            npz_path = (
                pred_dir
                / f"{slug}_{k}.npz"
            )

            i0, i1 = meta["split"][k]["indices"]

            split_dates = np.array(
                dates[i0:i1 + 1]
            )

            np.savez_compressed(
                npz_path,
                dates=split_dates,
                latitude=fine_lat,
                longitude=fine_lon,
                rainfall_mm=p.astype("float32"),
            )

            schema = {
                "file": npz_path.name,
                "description": (
                    "Layer-1 downscaled daily rainfall "
                    "on the fine grid; ready for "
                    "Panchayat-polygon intersection "
                    "(Layer 2)."
                ),
                "fields": {
                    "dates": (
                        "YYYY-MM-DD per sample"
                    ),
                    "latitude": (
                        "fine-grid lat centers "
                        "(deg N, ascending)"
                    ),
                    "longitude": (
                        "fine-grid lon centers "
                        "(deg E, ascending)"
                    ),
                    "rainfall_mm": (
                        "(n_days, lat, lon) "
                        "daily rainfall mm/day"
                    ),
                },
                "grid": {
                    "n_lat": len(fine_lat),
                    "n_lon": len(fine_lon),
                    "resolution_deg": 0.05,
                    "roi": meta["roi"],
                },
                "sample_first_date": {
                    "date": str(
                        split_dates[0]
                    ),
                    "rainfall_mm_shape": list(
                        p[0].shape
                    ),
                    "domain_mean_mm": float(
                        np.nanmean(p[0])
                    ),
                },
            }

            with open(
                pred_dir
                / f"{slug}_{k}_schema.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    schema,
                    f,
                    indent=1,
                )

    print(
        f"predictions -> {pred_dir}/ "
        "(npz + schema per model/split)"
    )

    # ---------------------------------------------------------
    # Showcase figure
    # ---------------------------------------------------------

    tk = (
        "test"
        if "test" in splits
        else splits[0]
    )

    if args.showcase_date:
        if args.showcase_date not in dates:
            raise ValueError(
                f"Showcase date {args.showcase_date} "
                "not found in metadata dates."
            )

        global_idx = dates.index(
            args.showcase_date
        )

        i0 = meta["split"][tk]["indices"][0]

        show = global_idx - i0

        if show < 0 or show >= len(Y[tk]):
            raise ValueError(
                f"Showcase date {args.showcase_date} "
                f"is not inside the {tk} split."
            )

    else:
        yv = []

        for i in range(len(Y[tk])):
            valid = (
                M[tk][i, 0]
                .astype(bool)
            )

            if valid.any():
                yv.append(
                    np.nanmean(
                        Y[tk][i, 0][valid]
                    )
                )
            else:
                yv.append(0.0)

        show = int(
            np.argmax(yv)
        )

    split_start = (
        meta["split"][tk]["indices"][0]
    )

    d = dates[
        split_start + show
    ]

    y_show = np.where(
        M[tk][show, 0].astype(bool),
        Y[tk][show, 0],
        np.nan,
    )

    finite_y = y_show[
        np.isfinite(y_show)
    ]

    vmax = (
        float(
            np.nanpercentile(
                finite_y,
                99,
            )
        )
        if finite_y.size
        else 10.0
    )

    vmax = max(vmax, 1.0)

    panels = [
        (
            X[tk][show, imd_idx]
            * rain_scale,
            "IMD 0.25 bilinear (input)",
        ),
        (
            preds["B1 bilinear IMD"][tk][show],
            "B1: bilinear IMD",
        ),
        (
            preds[
                "B2 bias-corrected bilinear"
            ][tk][show],
            "B2: bias-corrected bilinear",
        ),
    ]

    for name in models:
        panel_label = (
            f"U-Net [{name}] "
            f"({', '.join(models[name]['channels'])})"
        )

        panels.append(
            (
                preds[
                    f"U-Net [{name}]"
                ][tk][show],
                panel_label,
            )
        )

    panels.append(
        (
            y_show,
            "CHIRPS 0.05 (reference)",
        )
    )

    # Use the last loaded model for the error panel,
    # matching the last requested model.
    if models:
        last_model_name = next(
            reversed(models)
        )

        error_pred = preds[
            f"U-Net [{last_model_name}]"
        ][tk][show]

        error_title = (
            f"error: U-Net [{last_model_name}] "
            "- CHIRPS"
        )
    else:
        error_pred = (
            preds["B1 bilinear IMD"]
            [tk][show]
        )

        error_title = (
            "error: B1 bilinear IMD - CHIRPS"
        )

    err = error_pred - y_show

    n_panels = len(panels) + 1

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(
            4.6 * n_panels,
            4.6,
        ),
    )

    if n_panels == 1:
        axes = [axes]

    for ax, (fld, ttl) in zip(
        axes,
        panels,
    ):
        panel(
            ax,
            fld,
            fine_lat,
            fine_lon,
            ttl,
            vmax=vmax,
        )

    panel(
        axes[-1],
        err,
        fine_lat,
        fine_lon,
        error_title,
        vmax=max(
            float(
                np.nanpercentile(
                    np.abs(
                        err[
                            np.isfinite(err)
                        ]
                    ),
                    99,
                )
            )
            if np.isfinite(err).any()
            else 1.0,
            1.0,
        ),
        diff=True,
    )

    fig.suptitle(
        f"Rainfall downscaling - {d} "
        "(mm/day)",
        fontsize=12,
    )

    fig.colorbar(
        axes[0].images[0],
        ax=axes,
        shrink=0.85,
        label="mm/day",
    )

    out = (
        config.OUT_FIGS
        / f"comparison_{d}.png"
    )

    fig.savefig(
        out,
        dpi=140,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"figure -> {out}"
    )

    # ---------------------------------------------------------
    # Individual maps
    # ---------------------------------------------------------

    error_vmax = max(
        float(
            np.nanpercentile(
                np.abs(
                    err[
                        np.isfinite(err)
                    ]
                ),
                99,
            )
        )
        if np.isfinite(err).any()
        else 1.0,
        1.0,
    )

    map_items = [
        (
            X[tk][show, imd_idx]
            * rain_scale,
            "imd_coarse",
        ),
        (
            preds["B1 bilinear IMD"][tk][show],
            "baseline_b1",
        ),
        (
            preds[
                "B2 bias-corrected bilinear"
            ][tk][show],
            "baseline_b2",
        ),
        (
            y_show,
            "chirps_ref",
        ),
        (
            err,
            "error_unet_vs_chirps",
        ),
    ]

    for n in models:
        map_items.append(
            (
                preds[
                    f"U-Net [{n}]"
                ][tk][show],
                f"unet_{n}",
            )
        )

    for fld, tag in map_items:

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        if tag.startswith("error"):
            im = panel(
                ax,
                fld,
                fine_lat,
                fine_lon,
                f"{tag} {d}",
                vmax=error_vmax,
                diff=True,
            )

            fig.colorbar(
                im,
                ax=ax,
                label="mm/day "
                      "(model - reference)",
            )

        else:
            im = panel(
                ax,
                fld,
                fine_lat,
                fine_lon,
                f"{tag} {d}",
                vmax=vmax,
            )

            fig.colorbar(
                im,
                ax=ax,
                label="mm/day",
            )

        fig.savefig(
            config.OUT_MAPS
            / f"{tag}_{d}.png",
            dpi=140,
            bbox_inches="tight",
        )

        plt.close(fig)


if __name__ == "__main__":
    main()
'''

path = Path("/mnt/data/evaluate.py")
path.write_text(content, encoding="utf-8")
print(path)
