"""Export Layer 1 / Layer 2 artifacts as pickle files for later loading.

Writes (whichever sources exist):
  models/layer1_model.pkl              checkpoint dict (state_dict + channels)
  prediction/infer_2022-07-10.pkl      Layer-1 rainfall field (dates/lat/lon/rain)
  outputs/layer2/panchayat_weather.pkl Layer-2 Panchayat table

  python scripts/export_pickle.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import config  # noqa: E402


def _dump(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.2f} MB)")
    return path


def export_layer1_model() -> Path | None:
    ckpt = config.MODELS / "best_model.pt"
    if not ckpt.exists():
        alts = sorted(config.MODELS.glob("model_*.pt"))
        ckpt = alts[-1] if alts else None
    if ckpt is None or not ckpt.exists():
        print("[skip] no models/*.pt — train.py has not saved a checkpoint here")
        return None
    import torch
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    bundle = {
        "format": "layer1_unet_pickle_v1",
        "source_pt": str(ckpt),
        "channels": state.get("channels") or (state.get("args") or {}).get("channels"),
        "rain_scale": float(state.get("rain_scale", 100.0)),
        "width": int(state.get("width", 16)),
        "residual": bool(state.get("residual", True)),
        "state_dict": {k: v.cpu() for k, v in state["model"].items()},
        "roi": dict(config.ROI_DEFAULT),
        "fine_sub": config.FINE_SUB,
    }
    return _dump(bundle, config.MODELS / "layer1_model.pkl")


def export_layer1_field() -> Path | None:
    npz = config.PRED_DIR / "infer_2022-07-10.npz"
    if not npz.exists():
        hits = sorted(config.PRED_DIR.glob("infer_*.npz"))
        npz = hits[-1] if hits else None
    if npz is None:
        print("[skip] no prediction/infer_*.npz")
        return None
    z = np.load(npz, allow_pickle=True)
    bundle = {
        "format": "layer1_field_pickle_v1",
        "source_npz": str(npz),
        "dates": [str(d) for d in z["dates"]],
        "latitude": np.asarray(z["latitude"]),
        "longitude": np.asarray(z["longitude"]),
        "rainfall_mm": np.asarray(z["rainfall_mm"]),
    }
    return _dump(bundle, config.PRED_DIR / f"{npz.stem}.pkl")


def export_layer2() -> Path | None:
    csv = config.OUT_LAYER2 / "panchayat_weather_updated.csv"
    if not csv.exists():
        csv = config.OUT_LAYER2 / "panchayat_weather.csv"
    if not csv.exists():
        print("[skip] no Layer 2 CSV — run layer2_panchayat_mapping.py")
        return None
    df = pd.read_csv(csv)
    bundle = {
        "format": "layer2_panchayat_pickle_v1",
        "source_csv": str(csv),
        "dataframe": df,
        "columns": list(df.columns),
        "n_panchayats": int(df["panchayat_id"].nunique()) if "panchayat_id" in df.columns else len(df),
    }
    return _dump(bundle, config.OUT_LAYER2 / "panchayat_weather.pkl")


def main() -> None:
    export_layer1_model()
    export_layer1_field()
    export_layer2()
    print("load later with:  pickle.load(open(path, 'rb'))")


if __name__ == "__main__":
    main()
