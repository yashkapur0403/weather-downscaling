"""
Heavy-rain preservation experiments: E, F, G, H, I, J.

All use the full Model-D channel set (imd_rain, dem, era5_t2m, era5_t2m_max,
era5_dewp) and the same residual U-Net architecture.  The only variable is the
training loss:

  E   log1p          -- MAE in log1p(mm) space; preserves relative error signal
  F   combined       -- alpha*plain_MAE + (1-alpha)*ramped_heavy_MAE (gentle)
  G   extreme        -- plain MAE + large spike weight for pixels >= 50 mm
  H   log1p_combined -- log1p space + combined ramp (E + F combined)
    I   log1p_combined_extreme -- H plus an explicit >=50 mm loss term
    J   weighted       -- hard >=25 mm weighting on the full D channel set

Usage (from repo root):
    python scripts/run_heavy_rain_experiments.py [--skip-existing]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
TRAIN = ROOT / "scripts" / "train.py"
COMPARE = ROOT / "scripts" / "compare_heavy_rain.py"
MODELS = ROOT / "models"

D_CHANNELS = "imd_rain,dem,era5_t2m,era5_t2m_max,era5_dewp"

EXPERIMENTS: list[tuple[str, str, str]] = [
    # (model_tag, loss_name, description)
    ("model_e", "log1p",         "Model E: log1p loss"),
    ("model_f", "combined",      "Model F: combined gentle ramp"),
    ("model_g", "extreme",       "Model G: extreme-rain spike"),
    ("model_h", "log1p_combined","Model H: log1p + combined ramp"),
    ("model_i", "log1p_combined_extreme",
     "Model I: log1p + combined ramp + extreme weighting"),
    ("model_j", "weighted",      "Model J: hard-threshold weighted MAE"),
]


def run(cmd: list[str], label: str) -> int:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    print(f"\n[{label}] finished in {dt:.0f}s, exit={result.returncode}")
    return result.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip training if checkpoint already exists")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    args = ap.parse_args()

    failed: list[str] = []

    for tag, loss, desc in EXPERIMENTS:
        ckpt = MODELS / f"{tag}.pt"
        if args.skip_existing and ckpt.exists():
            print(f"\n[skip] {tag} already exists: {ckpt}")
            continue

        cmd = [
            PYTHON, str(TRAIN),
            "--channels", D_CHANNELS,
            "--loss", loss,
            "--out-name", tag,
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
        ]
        rc = run(cmd, desc)
        if rc != 0:
            print(f"[ERROR] {tag} training exited with code {rc}")
            failed.append(tag)

    print("\n" + "=" * 70)
    if failed:
        print(f"[WARN] The following experiments failed: {failed}")
    else:
        print("[OK] All experiments finished.  Running comparison...")
        subprocess.run([PYTHON, str(COMPARE)], cwd=ROOT)


if __name__ == "__main__":
    main()
