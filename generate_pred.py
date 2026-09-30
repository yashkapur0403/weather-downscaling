"""Write outputs/prediction_test.npz - the fine-grid rainfall field that
Layer 2 / the backend consume.

The deployed predictor is whatever scripts/ensemble.py resolves:
    models/ensemble.json  (weighted mean of the listed checkpoints) if present,
    otherwise the single models/best_model.pt.
Both paths are identical in what they produce except for the number of
checkpoints averaged, so the NPZ always matches the deployed model.

Run from the repo root:   python generate_pred.py
Optional override:        python generate_pred.py --checkpoint models/model_e.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from ensemble import load_members, load_single, predict_mm_with  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=None,
                    help="explicit single checkpoint (default: the deployed ensemble)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    meta = json.loads((ROOT / "data" / "processed" / "meta.json").read_text())
    all_channels = meta["channels"]

    if a.checkpoint:
        nets, spec, manifest = load_single(a.checkpoint, ROOT / "scripts")
    else:
        nets, spec, manifest = load_members(ROOT / "models", ROOT / "scripts")
    deployed = (" + ".join(f"{n} (w={w:.2f})"
                           for n, w in zip(spec["members"], spec["weights"]))
                if len(nets) > 1 else spec["members"][0])
    print(f"deployed predictor: {deployed}")
    print(f"  manifest={'models/ensemble.json' if manifest else 'none (single checkpoint)'}"
          f" | channels={spec['channels']} | residual={spec['residual']} "
          f"| rain_scale={spec['rain_scale']}")

    X = np.load(ROOT / "data" / "processed" / f"X_{a.split}.npy")
    M = np.load(ROOT / "data" / "processed" / f"M_{a.split}.npy")
    dates = meta["split"][a.split]["dates"]
    lat, lon = meta["grid"]["fine_lat"], meta["grid"]["fine_lon"]
    assert X.shape[0] == len(dates), (X.shape, len(dates))
    print(f"X_{a.split} shape: {X.shape}")

    ci = [all_channels.index(c) for c in spec["channels"]]
    Xk = np.ascontiguousarray(X[:, ci])
    i0 = spec["channels"].index("imd_rain")
    base = Xk[:, i0:i0 + 1] if spec["residual"] else None
    pred_mm = predict_mm_with(nets, spec, Xk, base)

    m = M[:, 0] if M.ndim == 4 else M
    pred_mm = np.where(m == 1, pred_mm, np.nan)

    out_file = Path(a.out) if a.out else ROOT / "outputs" / f"prediction_{a.split}.npz"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_file, dates=np.array(dates),
                        latitude=lat, longitude=lon,
                        rainfall_mm=pred_mm.astype("float32"))
    finite = pred_mm[np.isfinite(pred_mm)]
    print(f"Saved {out_file}")
    print(f"  finite cells/day={int(np.isfinite(pred_mm[0]).sum())} "
          f"| max={finite.max():.2f} mm | mean={finite.mean():.2f} mm")

    # ---- provenance: bind the served grid to the exact models that made it --
    # Consumers (backend, reports) can compare this against the artefacts they
    # read, so a retrain can never silently leave a stale grid in place.
    def sha(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()

    ens_manifest = manifest          # keep the ensemble manifest for the record
    prov = {
        "generated_by": "generate_pred.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "geometry": {"split": a.split, "n_dates": len(dates),
                     "first_date": str(dates[0]), "last_date": str(dates[-1]),
                     "shape": list(int(x) for x in pred_mm.shape),
                     "dtype": "float32"},
        "values": {"finite_cells_per_day": int(np.isfinite(pred_mm[0]).sum()),
                   "max_mm": float(finite.max()),
                   "mean_mm": float(finite.mean()),
                   "any_negative": bool(np.nanmin(pred_mm) < 0),
                   "nan_masked": True},
        "predictor": {
            "members": spec["members"], "weights": spec["weights"],
            "rain_scale": spec["rain_scale"], "channels": spec["channels"],
            "n_parameters": spec["n_parameters"],
            "manifest": ("models/ensemble.json" if ens_manifest else None),
            "checkpoint_sha256": {n: sha(ROOT / "models" / n)
                                  for n in spec["members"]},
            "clipping": "per-member clip(net_k(X)*rain_scale, 0) then weighted mean",
        },
        "artefact": {"path": str(out_file.relative_to(ROOT)).replace("\\", "/"),
                     "sha256": sha(out_file)},
        "inputs": {"X": f"data/processed/X_{a.split}.npy",
                   "M": f"data/processed/M_{a.split}.npy",
                   "meta": "data/processed/meta.json"},
    }
    mf = ROOT / "outputs" / "metrics" / "layer1_manifest.json"
    mf.parent.mkdir(parents=True, exist_ok=True)
    mf.write_text(json.dumps(prov, indent=1))
    print(f"  provenance -> {mf.relative_to(ROOT)}")
    print(f"  sha256={prov['artefact']['sha256'][:16]}... "
          f"predictor={' + '.join(spec['members'])} weights={spec['weights']}")


if __name__ == "__main__":
    main()
