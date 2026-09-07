"""
Train the downscaling model: [channels] -> fine CHIRPS rainfall.

Key scientific properties:
  * Residual learning: the network predicts a CORRECTION to the bilinear
    baseline (channel 'imd_rain'), i.e. pred = baseline + residual. It can
    therefore never do worse than the baseline at initialization.
  * Model selection strictly on VALIDATION MAE; test is never touched.
  * Loss options: mae | weighted (heavy-rain emphasis) | log1p.
  * Strict time-based splits produced by preprocess.py (no shuffling across time).

Run:
  python scripts/train.py --channels imd_rain,dem,era5_t2m,era5_t2m_max,era5_dewp,era5_wind \
      --loss weighted --out-name model_ablation_d
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


# --------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    """~0.12M params at width=16: 2 downsample levels, 3x3 convs."""

    def __init__(self, cin=2, width=16, residual=True):
        super().__init__()
        self.residual = residual
        w = width
        self.enc1 = ConvBlock(cin, w)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(w, 2 * w)
        self.pool2 = nn.MaxPool2d(2)
        self.bott = ConvBlock(2 * w, 4 * w)
        self.up2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
        self.dec2 = ConvBlock(4 * w, 2 * w)
        self.up1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
        self.dec1 = ConvBlock(2 * w, w)
        self.head = nn.Conv2d(w, 1, 1)
        nn.init.zeros_(self.head.weight)   # start as pure baseline (residual mode)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, baseline=None):
        h, w = x.shape[-2:]
        ph = (4 - h % 4) % 4
        pw = (4 - w % 4) % 4
        if ph or pw:
            x = nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bott(self.pool2(e2))
        u2 = nn.functional.interpolate(self.up2(b), size=e2.shape[-2:],
                                       mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = nn.functional.interpolate(self.up1(d2), size=e1.shape[-2:],
                                       mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        out = self.head(d1)[..., :h, :w]
        if self.residual and baseline is not None:
            return baseline + out
        return out


# --------------------------------------------------------------------------
def make_loss(name: str, rain_scale: float):
    """Return loss(pred_norm, y_norm, mask) with heavy-rain emphasis options."""
    if name == "mae":
        def loss(p, y, m):
            return ((p - y).abs() * m).sum() / m.sum().clamp(min=1.0)
        return loss
    if name == "weighted":
        # emphasize heavy-rain pixels (>= config.WEIGHT_RAIN_MM, train-data scale)
        thr = config.WEIGHT_RAIN_MM / rain_scale

        def loss(p, y, m):
            w = m * (1.0 + (config.WEIGHT_MULT - 1.0) * (y >= thr).float())
            return ((p - y).abs() * w).sum() / w.sum().clamp(min=1.0)
        return loss
    if name == "log1p":
        # MAE in log1p space on mm -> emphasizes relative errors on light rain
        def loss(p, y, m):
            pl = torch.log1p(p.clamp(min=0) * rain_scale)
            yl = torch.log1p(y.clamp(min=0) * rain_scale)
            return ((pl - yl).abs() * m).sum() / m.sum().clamp(min=1.0)
        return loss
    raise ValueError(f"unknown loss {name}")


def predict_mm(model, X, rain_scale, device, baseline=None, residual=True):
    model.eval()
    with torch.no_grad():
        b = None
        if residual and baseline is not None:
            b = (torch.from_numpy(np.ascontiguousarray(baseline)).to(device)
                 if isinstance(baseline, np.ndarray) else baseline.to(device))
        out = model(torch.from_numpy(X).to(device), baseline=b).cpu().numpy()
    return np.clip(out[:, 0] * rain_scale, 0.0, None)


def metrics(pred_mm, y_mm, m):
    m = m.astype(bool)
    if m.sum() == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "corr": float("nan")}
    e = pred_mm[m] - y_mm[m]
    out = {"MAE": float(np.mean(np.abs(e))),
           "RMSE": float(np.sqrt(np.mean(e ** 2)))}
    a, b = pred_mm[m], y_mm[m]
    out["corr"] = (float(np.corrcoef(a, b)[0, 1])
                   if a.std() > 0 and b.std() > 0 else float("nan"))
    return out


def load_data():
    X = {k: np.load(config.PROCESSED / f"X_{k}.npy") for k in ("train", "val", "test")}
    Y = {k: np.load(config.PROCESSED / f"Y_{k}.npy") for k in ("train", "val", "test")}
    M = {k: np.load(config.PROCESSED / f"M_{k}.npy") for k in ("train", "val", "test")}
    with open(config.PROCESSED / "meta.json") as f:
        meta = json.load(f)
    return X, Y, M, meta


def channel_indices(meta, channels: list[str]) -> list[int]:
    available = meta["channels"]
    missing = [c for c in channels if c not in available]
    if missing:
        raise ValueError(f"channels {missing} not in processed data "
                         f"(available: {available})")
    return [available.index(c) for c in channels]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--channels", default=",".join(config.CHANNELS_DEFAULT),
                    help="comma list, subset of meta.channels")
    ap.add_argument("--loss", default="mae", choices=["mae", "weighted", "log1p"])
    ap.add_argument("--residual", type=int, default=1,
                    help="1 = learn correction to bilinear baseline (default), "
                         "0 = predict the full field")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patch", type=int, default=48,
                    help="random NxN training crops (0 = full frames)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-subset", type=int, default=0,
                    help="evaluate epoch selection on an evenly-spaced subset "
                         "of N val days for speed (0 = full val). Final "
                         "reported metrics always use the FULL val set.")
    ap.add_argument("--out-name", default="best_model",
                    help="checkpoint name under models/ (no extension)")
    args = ap.parse_args()
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, Y, M, meta = load_data()
    rain_scale = meta["normalization"]["rain_scale_mm"]
    ci = channel_indices(meta, channels)
    print(f"device={device} channels={[meta['channels'][i] for i in ci]} "
          f"loss={args.loss} residual={bool(args.residual)}")
    print(f"train={X['train'].shape}")

    def prep(k):
        Xk = torch.from_numpy(X[k][:, ci].copy())
        Yk = torch.from_numpy(Y[k] / rain_scale)
        Mk = torch.from_numpy(M[k])
        base = Xk[:, 0:1].clone()          # bilinear baseline (normalized)
        return Xk, Yk, Mk, base

    T = {k: prep(k) for k in ("train", "val", "test")}
    loss_fn = make_loss(args.loss, rain_scale)

    model = SmallUNet(cin=len(ci), width=args.width,
                      residual=bool(args.residual)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: SmallUNet(width={args.width}) params={n_params/1e3:.1f}k")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(args.patience // 3, 5))

    ckpt_path = config.MODELS / f"{args.out_name}.pt"
    rng = np.random.default_rng(args.seed)
    best_val, best_epoch, history, t0 = float("inf"), -1, [], time.time()

    Xtr, Ytr, Mtr, Btr = T["train"]
    n_days, _, H, W = Xtr.shape
    steps = max(n_days // args.batch, 1)
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        for _ in range(steps):
            if args.patch > 0:
                p = min(args.patch, H, W)
                dy = rng.integers(0, n_days, args.batch)
                oy = rng.integers(0, H - p + 1, args.batch)
                ox = rng.integers(0, W - p + 1, args.batch)
                sl = lambda A: torch.stack([A[dy[j], :, oy[j]:oy[j]+p, ox[j]:ox[j]+p]
                                            for j in range(args.batch)]).to(device)
                xb, yb, mb, bb = sl(Xtr), sl(Ytr), sl(Mtr), sl(Btr)
            else:
                ii = rng.integers(0, n_days, args.batch)
                xb, yb, mb, bb = (A[ii].to(device) for A in (Xtr, Ytr, Mtr, Btr))
            if rng.random() < 0.5:
                xb, yb, mb, bb = (torch.flip(A, [-1]) for A in (xb, yb, mb, bb))
            if rng.random() < 0.5:
                xb, yb, mb, bb = (torch.flip(A, [-2]) for A in (xb, yb, mb, bb))
            opt.zero_grad()
            loss = loss_fn(model(xb, baseline=bb), yb, mb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            if args.val_subset > 0 and len(T["val"][0]) > args.val_subset:
                vs = np.linspace(0, len(T["val"][0]) - 1,
                                 args.val_subset).astype(int)
                vs = np.unique(vs)
            else:
                vs = np.arange(len(T["val"][0]))
            Xv, Yv, Mv, Bv = (T["val"][i][torch.as_tensor(vs)].to(device)
                              for i in range(4))
            val_loss = loss_fn(model(Xv, baseline=Bv), Yv, Mv).item()
        sched.step(val_loss)
        history.append({"epoch": epoch,
                        "train_loss": float(np.mean(tr_losses)),
                        "val_loss": val_loss,
                        "lr": opt.param_groups[0]["lr"]})
        if val_loss < best_val - 1e-5:
            best_val, best_epoch = val_loss, epoch
            torch.save({"model": model.state_dict(),
                        "args": {**vars(args), "channels": channels,
                                 "channel_indices": ci},
                        "rain_scale": rain_scale, "width": args.width,
                        "residual": bool(args.residual),
                        "residual_baseline": "imd_rain (normalized bilinear IMD)",
                        "best_val_loss": best_val,
                        "channels": channels,
                        "loss": args.loss,
                        "history": history}, ckpt_path)
        if epoch % 25 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  train {np.mean(tr_losses):.4f}  "
                  f"val {val_loss:.4f}  best {best_val:.4f} @ {best_epoch}")
        if epoch - best_epoch >= args.patience:
            print(f"early stopping at epoch {epoch}")
            break

    dt = time.time() - t0
    print(f"training done in {dt:.0f}s; best val loss {best_val:.4f} @ {best_epoch}")
    print(f"checkpoint -> {ckpt_path}")

    # val report in mm (model selection metric, reported for transparency)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    for k in ("val",):
        pred = predict_mm(model, X[k][:, ci].copy(), rain_scale, device,
                          baseline=X[k][:, 0:1].copy() if args.residual else None,
                          residual=bool(args.residual))
        m = metrics(pred, Y[k][:, 0], M[k][:, 0])
        print(f"{k}: MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} corr={m['corr']:.3f} (mm)")

    with open(config.MODELS / f"{args.out_name}_history.json", "w") as f:
        json.dump({"args": {**vars(args), "channels": channels},
                   "history": history, "best_val_loss": best_val,
                   "best_epoch": best_epoch, "train_seconds": dt}, f, indent=1)


if __name__ == "__main__":
    main()
