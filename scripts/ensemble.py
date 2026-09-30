"""
Shared loader for the DEPLOYED Layer-1 predictor.

Why an ensemble
---------------
A single MAE-trained residual U-Net is deliberately calibrated to the
conditional mean of the fine-grid rainfall field.  Conditional means of
extreme days sit far below the extreme threshold, so a >=50 mm/day event is
systematically predicted at ~26-28 mm and the nominal F1 at that threshold
collapses even though the underlying ranking is good (measured on the test
split: precision 0.44, recall 0.08 for the MAE-optimal checkpoint).

Averaging two checkpoints that share architecture/channels/scaling but were
trained with DIFFERENT loss weightings (plain MAE vs heavy-rain weighted)
reduces that shrinkage: the heavy-weighted member supplies the extreme values
and the MAE member keeps the mean error low.  The member pair and the mixing
weight are chosen on the VALIDATION split only (scripts/ablation.py), never on
test.

Artefact
--------
    models/ensemble.json        written by scripts/ablation.py
        {"members": [{"checkpoint": "model_e.pt", "weight": 0.5}, ...], ...}

If the manifest is absent the loader falls back to the single
models/best_model.pt, so every consumer keeps working on an old checkout.

Consumers (all use this module, so there is exactly ONE definition of "what
the deployed model is"):
    generate_pred.py         -> outputs/prediction_test.npz
    backend/live_infer.py    -> on-demand inference
    scripts/ablation.py      -> reported metrics
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_MANIFEST = "ensemble.json"
FALLBACK_CHECKPOINT = "best_model.pt"
# Members must agree on these, otherwise averaging would mix incompatible fields.
_AGREE_ON = ("channels", "residual", "width", "rain_scale")


def read_manifest(models_dir: str | Path) -> tuple[list[dict], dict | None]:
    """Return (members, manifest). Falls back to the single best_model.pt."""
    p = Path(models_dir) / DEFAULT_MANIFEST
    if not p.exists():
        return [{"checkpoint": FALLBACK_CHECKPOINT, "weight": 1.0}], None
    man = json.loads(p.read_text())
    members = man.get("members") or [{"checkpoint": FALLBACK_CHECKPOINT, "weight": 1.0}]
    if not members:
        raise ValueError(f"{p} has no members")
    return members, man


def _load_one(ckpt_path: Path, scripts_dir: str | Path | None = None):
    """Load one checkpoint -> (net, info). Validates the architecture fields."""
    import sys

    import torch

    if scripts_dir is None:
        scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from train import SmallUNet  # noqa: E402

    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"checkpoint {ckpt_path} missing")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    chans = ckpt.get("channels") or ckpt["args"]["channels"]
    info = {"channels": list(chans),
            "residual": bool(ckpt.get("residual", True)),
            "width": int(ckpt.get("width", 16)),
            "rain_scale": float(ckpt.get("rain_scale", 100.0))}
    net = SmallUNet(cin=len(chans), width=info["width"], residual=info["residual"])
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net, info


def _combine(loaded, weights, manifest):
    """Validate agreement across members and pack the spec."""
    spec = None
    for info in loaded:
        if spec is None:
            spec = info
            continue
        for k in _AGREE_ON:
            if info[k] != spec[k]:
                raise ValueError(f"ensemble members disagree on {k}: "
                                 f"{spec[k]} vs {info[k]}")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("ensemble weights sum to zero")
    spec = dict(spec)
    spec["weights"] = [w / total for w in weights]
    spec["n_parameters"] = None
    return spec, manifest


def load_members(models_dir: str | Path, scripts_dir: str | Path | None = None):
    """Load the DEPLOYED ensemble.

    Returns (nets, spec, manifest) where spec carries the shared geometry
    (channels, residual, width, rain_scale) plus the normalised weights.
    """
    import torch  # noqa: F401  (imported here so a missing torch fails early)

    models_dir = Path(models_dir)
    members, manifest = read_manifest(models_dir)
    nets, infos, weights, names = [], [], [], []
    for m in members:
        net, info = _load_one(models_dir / m["checkpoint"], scripts_dir)
        nets.append(net)
        infos.append(info)
        weights.append(float(m.get("weight", 1.0)))
        names.append(m["checkpoint"])
    spec, manifest = _combine(infos, weights, manifest)
    spec["members"] = names
    spec["n_parameters"] = int(sum(p.numel() for p in nets[0].parameters()))
    return nets, spec, manifest


def load_single(ckpt_path: str | Path, scripts_dir: str | Path | None = None):
    """Explicit single-checkpoint override (CLI use). Returns (nets, spec, None)."""
    net, info = _load_one(Path(ckpt_path), scripts_dir)
    spec, _ = _combine([info], [1.0], None)
    spec["members"] = [Path(ckpt_path).name]
    spec["n_parameters"] = int(sum(p.numel() for p in net.parameters()))
    return [net], spec, None


def ensemble_predict_mm(nets, spec, X, baseline=None):
    """Deployed rainfall field in mm/day, as a torch tensor (n, H, W).

    The field is the WEIGHTED MEAN OF THE PER-MEMBER MILLIMETRE FIELDS:
        pred = sum_k w_k * clip(net_k(X, base) * rain_scale, 0)
    Clipping each member before averaging is deliberate and load-bearing: it is
    exactly what scripts/ablation.py and the reported metrics compute
    (train.predict_mm clips per member), so the served field and the published
    numbers cannot drift apart. Averaging in normalized space and clipping once
    would give a different field wherever a member predicts a negative residual.
    """
    import torch

    acc = None
    with torch.no_grad():
        for net, w in zip(nets, spec["weights"]):
            p = net(X, baseline=baseline)
            p = torch.clamp(p[:, 0] * spec["rain_scale"], min=0.0)
            acc = p * w if acc is None else acc + p * w
    return acc


def predict_mm(models_dir, X, baseline=None, scripts_dir=None):
    """Deployed daily rainfall in mm/day for a batch of model inputs.

    X: (n, cin, H, W) float32, already channel-subset and normalized to match
    spec["channels"].  baseline: (n, 1, H, W) normalized imd_rain when the
    members are in residual mode.
    """
    nets, spec, _ = load_members(models_dir, scripts_dir)
    return predict_mm_with(nets, spec, X, baseline)


def predict_mm_with(nets, spec, X, baseline=None):
    """numpy wrapper of ensemble_predict_mm -> (n, H, W) mm/day.
    Never NaN: applying the land/valid mask is the caller's job."""
    import numpy as np
    import torch

    xb = torch.from_numpy(np.ascontiguousarray(X, dtype="float32"))
    bb = None if baseline is None else torch.from_numpy(
        np.ascontiguousarray(baseline, dtype="float32"))
    return ensemble_predict_mm(nets, spec, xb, bb).numpy()
