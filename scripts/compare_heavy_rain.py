"""
Compare heavy-rain experiments (E-J) against the Model D baseline.

Evaluates all available checkpoints on the TEST split and prints:
  - MAE, RMSE, corr
  - F1 at 10 / 25 / 50 mm

Writes results to:
  outputs/metrics/heavy_rain_comparison.md
  outputs/metrics/heavy_rain_comparison.json

Usage (from repo root):
    python scripts/compare_heavy_rain.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import config  # noqa: E402
import torch   # noqa: E402

from evaluate import THRESHOLDS, evaluate_model  # noqa: E402
from train import SmallUNet, load_data, predict_mm  # noqa: E402

MODELS_TO_EVAL: list[tuple[str, str]] = [
    ("model_d",  "D  U-Net+DEM+ERA5 (baseline, mae loss)"),
    ("model_e",  "E  U-Net+DEM+ERA5 (log1p loss)"),
    ("model_f",  "F  U-Net+DEM+ERA5 (combined gentle ramp)"),
    ("model_g",  "G  U-Net+DEM+ERA5 (extreme-rain spike)"),
    ("model_h",  "H  U-Net+DEM+ERA5 (log1p+combined ramp)"),
    ("model_i",  "I  U-Net+DEM+ERA5 (log1p+combined+extreme)"),
    ("model_j",  "J  U-Net+DEM+ERA5 (weighted MAE)"),
]


def load_checkpoint(path: Path, channels: list[str], rain_scale: float):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    chans = ckpt.get("channels") or ckpt.get("args", {}).get("channels")
    if not chans:
        raise ValueError(f"{path.name}: no channel info in checkpoint")
    missing = [c for c in chans if c not in channels]
    if missing:
        raise ValueError(f"{path.name}: needs channels {missing}")
    ci = [channels.index(c) for c in chans]
    residual = bool(ckpt.get("residual", True))
    width = int(ckpt.get("width", ckpt.get("args", {}).get("width", 16)))
    model = SmallUNet(cin=len(ci), width=width, residual=residual)
    model.load_state_dict(ckpt["model"])
    model.eval()
    rs = float(ckpt.get("rain_scale", rain_scale))
    return model, ci, residual, rs


def main() -> None:
    X, Y, M, meta = load_data()
    rain_scale = float(meta["normalization"]["rain_scale_mm"])
    channels = meta["channels"]
    imd_idx = channels.index("imd_rain")

    results: dict[str, dict] = {}

    for tag, label in MODELS_TO_EVAL:
        path = config.MODELS / f"{tag}.pt"
        if not path.exists():
            print(f"[skip] {tag}: checkpoint not found")
            continue
        try:
            model, ci, residual, rs = load_checkpoint(path, channels, rain_scale)
        except Exception as e:
            print(f"[skip] {tag}: {e}")
            continue

        for split in ("val", "test"):
            Xk = np.ascontiguousarray(X[split][:, ci])
            base = (np.ascontiguousarray(X[split][:, imd_idx:imd_idx + 1])
                    if residual else None)
            p = predict_mm(model, Xk, rs, "cpu", baseline=base, residual=residual)
            r = evaluate_model(p, Y[split], M[split])
            results.setdefault(tag, {})[split] = {
                "label": label,
                "MAE": r["MAE"], "RMSE": r["RMSE"], "corr": r["corr"],
                **{f"F1>={t:g}mm": r[f">={t:g}mm"]["F1"] for t in THRESHOLDS},
                **{f"prec>={t:g}mm": r[f">={t:g}mm"]["precision"] for t in THRESHOLDS},
                **{f"rec>={t:g}mm": r[f">={t:g}mm"]["recall"] for t in THRESHOLDS},
            }
        print(f"[ok] {tag}: {label}")

    if not results:
        print("No checkpoints found — run run_heavy_rain_experiments.py first.")
        return

    # ------------------------------------------------------------------ table
    header = (f"{'Model':50s}  {'ValMAE':>7s}  "
              f"{'TestMAE':>7s}  {'RMSE':>7s}  {'Corr':>6s}  "
              f"{'F1≥10':>6s}  {'F1≥25':>6s}  {'F1≥50':>6s}  "
              f"{'ΔMAE':>7s}  {'ΔF1₅₀':>7s}")
    sep = "-" * len(header)
    print("\n" + header)
    print(sep)

    for tag, label in MODELS_TO_EVAL:
        if tag not in results:
            continue
        vr = results[tag].get("val", {})
        tr = results[tag]["test"]
        val_mae = f"{vr['MAE']:.2f}" if vr else "  -  "
        baseline = results.get("model_d", {}).get("test", {})
        delta_mae = tr["MAE"] - baseline.get("MAE", tr["MAE"])
        delta_f1_50 = tr["F1>=50mm"] - baseline.get("F1>=50mm", tr["F1>=50mm"])
        row = (f"{tr['label']:50s}  {val_mae:>7s}  "
               f"{tr['MAE']:7.2f}  {tr['RMSE']:7.2f}  {tr['corr']:6.3f}  "
               f"{tr['F1>=10mm']:6.3f}  {tr['F1>=25mm']:6.3f}  {tr['F1>=50mm']:6.3f}  "
               f"{delta_mae:+7.2f}  {delta_f1_50:+7.3f}")
        print(row)

    # ----------------------------------------------------------- markdown file
    out_dir = config.OUT_METRICS
    out_dir.mkdir(parents=True, exist_ok=True)

    md_lines = [
        "# Heavy-Rain Preservation Experiments",
        "",
        "Baseline: **Model D** (U-Net + DEM + ERA5, plain MAE loss).",
        "All models share the same architecture (SmallUNet, width=16, residual=True)",
        "and channel set (imd_rain, dem, era5_t2m, era5_t2m_max, era5_dewp).",
        "Test split: 2022 Jun–Sep (122 days). Metrics over all valid land pixels.",
        "",
        "| Row | Model | Val MAE | Test MAE | Test RMSE | Test Corr |"
        " F1≥10mm | F1≥25mm | F1≥50mm | ΔMAE vs D | ΔF1≥25 vs D | ΔF1≥50 vs D |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, label in MODELS_TO_EVAL:
        if tag not in results:
            continue
        vr = results[tag].get("val", {})
        tr = results[tag]["test"]
        dtest = results.get("model_d", {}).get("test", tr)
        val_mae = f"{vr['MAE']:.2f}" if vr else "-"
        md_lines.append(
            f"| {tag.upper()} | {label.strip()} | {val_mae}"
            f" | {tr['MAE']:.2f} | {tr['RMSE']:.2f} | {tr['corr']:.3f}"
            f" | {tr['F1>=10mm']:.3f} | {tr['F1>=25mm']:.3f}"
            f" | {tr['F1>=50mm']:.3f} | {tr['MAE'] - dtest['MAE']:+.2f}"
            f" | {tr['F1>=25mm'] - dtest['F1>=25mm']:+.3f}"
            f" | {tr['F1>=50mm'] - dtest['F1>=50mm']:+.3f} |"
        )

    md_lines += [
        "",
        "## Notes on loss design",
        "",
        "| ID | Loss | Design goal |",
        "|---|---|---|",
        "| D | `mae` | Baseline: minimize masked MAE in normalized space |",
        "| E | `log1p` | Train in log1p(mm) space; penalises relative under-prediction of heavy events |",
        "| F | `combined` | α·plain_MAE + (1-α)·ramped_weighted_MAE; gentle linear ramp 10→50 mm |",
        "| G | `extreme` | plain_MAE + spike term (×8) for pixels ≥ 50 mm |",
        "| H | `log1p_combined` | log1p space with the same ramp as F; combines E+F |",
        "| I | `log1p_combined_extreme` | H plus the explicit extreme-event term in log space |",
        "| J | `weighted` | hard 3x weighting at >=25 mm, using all D channels |",
        "",
        "Loss constants (config.py): COMBINED_ALPHA=0.6, COMBINED_MULT_LO=1.5,",
        "COMBINED_MULT_HI=4.0, EXTREME_THR_MM=50, EXTREME_MULT=8.",
        "",
        "## Reproducibility",
        "",
        "Run `python scripts/run_heavy_rain_experiments.py --skip-existing` from the Layer-1 project root.",
        "All candidates use the D channels, residual SmallUNet width 16, seed 42, batch 16,",
        "patch 48, learning rate 0.001, up to 300 epochs, and patience 40.",
        "Per-checkpoint recorded arguments:",
        "",
        "| Model | Loss | Seed | Epochs | Patience | Best validation loss |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for tag, _ in MODELS_TO_EVAL:
        path = config.MODELS / f"{tag}.pt"
        if tag not in results or not path.exists():
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        args = ckpt.get("args", {})
        md_lines.append(
            f"| {tag.upper()} | {ckpt.get('loss', args.get('loss', '-'))}"
            f" | {args.get('seed', '-')} | {args.get('epochs', '-')}"
            f" | {args.get('patience', '-')} | {ckpt.get('best_val_loss', '-'):.6g} |"
        )

    md_path = out_dir / "heavy_rain_comparison.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\nReport -> {md_path}")

    json_path = out_dir / "heavy_rain_comparison.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"JSON   -> {json_path}")


if __name__ == "__main__":
    main()
