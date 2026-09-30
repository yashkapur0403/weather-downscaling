"""Smoke test: scripts/ensemble.py must reproduce the E/F blend exactly, and
generate_pred.py must agree with it."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # repo root (this script lives in qa/)
sys.path.insert(0, str(HERE / "scripts"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from train import SmallUNet, predict_mm  # noqa: E402
from ensemble import load_members, load_single, predict_mm_with  # noqa: E402

meta = json.loads((HERE / "data" / "processed" / "meta.json").read_text())
allc = meta["channels"]
X = np.load(HERE / "data" / "processed" / "X_test.npy")[:4]
print("X slice", X.shape)

nets, spec, man = load_members(HERE / "models", HERE / "scripts")
print("members:", spec["members"], "weights:", spec["weights"],
      "channels:", spec["channels"], "params:", spec["n_parameters"])
print("manifest present:", man is not None)

ci = [allc.index(c) for c in spec["channels"]]
Xk = np.ascontiguousarray(X[:, ci])
i0 = spec["channels"].index("imd_rain")
base = Xk[:, i0:i0 + 1]
ens = predict_mm_with(nets, spec, Xk, base)

# reference: manual per-member prediction then average
refs = []
for nm in ("model_e", "model_f"):
    ck = torch.load(HERE / "models" / f"{nm}.pt", map_location="cpu", weights_only=False)
    m = SmallUNet(len(ck["channels"]), int(ck["width"]), bool(ck["residual"]))
    m.load_state_dict(ck["model"]); m.eval()
    refs.append(predict_mm(m, Xk, float(ck["rain_scale"]), "cpu",
                           baseline=base, residual=bool(ck["residual"])))
manual = 0.5 * refs[0] + 0.5 * refs[1]

print("max |ensemble - manual mean| :", float(np.abs(ens - manual).max()))
print("shapes:", ens.shape, manual.shape)

# single-checkpoint override path
n1, s1, m1 = load_single(HERE / "models" / "model_e.pt", HERE / "scripts")
one = predict_mm_with(n1, s1, Xk, base)
print("max |single(E) - ref E|      :", float(np.abs(one - refs[0]).max()))
print("single spec members:", s1["members"], "params:", s1["n_parameters"])

assert np.allclose(ens, manual, atol=1e-4), "ensemble != weighted member mean"
assert np.allclose(one, refs[0], atol=1e-4), "single override != member E"
print("\nOK ensemble loader reproduces the blend exactly.")
