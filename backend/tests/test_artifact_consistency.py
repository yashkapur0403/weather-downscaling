"""Layer-1 artefact consistency.

A retrain that forgets to regenerate outputs/prediction_test.npz must not go
unnoticed: generate_pred.py records the sha256 of the grid, of every
contributing checkpoint and the mixing weights in
outputs/metrics/layer1_manifest.json, and /api/metrics republishes them.

These tests are torch-free on purpose - they validate the artefacts that the
torch environment produced, so they run in the backend's light venv.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "outputs" / "metrics" / "layer1_manifest.json"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def prov():
    if not MANIFEST.exists():
        pytest.skip("outputs/metrics/layer1_manifest.json absent (run generate_pred.py)")
    return json.loads(MANIFEST.read_text())


def test_grid_hash_matches_recorded(prov):
    """The served grid is byte-identical to what the manifest describes."""
    grid = REPO / prov["artefact"]["path"]
    assert grid.exists(), f"{grid} missing but recorded in the manifest"
    assert _sha256(grid) == prov["artefact"]["sha256"]


def test_grid_values_match_recorded(prov):
    z = np.load(REPO / prov["artefact"]["path"])
    g, v = prov["geometry"], prov["values"]
    rain = z["rainfall_mm"]
    assert rain.shape[0] == g["n_dates"]
    assert list(rain.shape) == g["shape"]
    assert str(z["dates"][0]) == g["first_date"]
    assert str(z["dates"][-1]) == g["last_date"]
    finite = rain[np.isfinite(rain)]
    assert int(np.isfinite(rain[0]).sum()) == v["finite_cells_per_day"]
    assert float(finite.max()) == pytest.approx(v["max_mm"], rel=1e-6)
    assert float(finite.mean()) == pytest.approx(v["mean_mm"], rel=1e-6)
    assert not (finite < 0).any(), "negative rainfall in the served grid"


def test_checkpoint_hashes_match_recorded(prov):
    """Every checkpoint that contributed to the grid is still on disk unchanged."""
    for name, digest in prov["predictor"]["checkpoint_sha256"].items():
        ckpt = REPO / "models" / name
        assert ckpt.exists(), f"{name} recorded as contributing but missing"
        assert _sha256(ckpt) == digest, f"{name} changed since the grid was generated"


def test_manifest_single_source_of_truth(prov):
    """Models/ensemble.json and the grid's provenance must agree."""
    ens = REPO / "models" / "ensemble.json"
    if not ens.exists():
        pytest.skip("no ensemble manifest (single-checkpoint deployment)")
    man = json.loads(ens.read_text())
    declared = [m["checkpoint"] for m in man["members"]]
    assert declared == prov["predictor"]["members"]
    w = prov["predictor"]["weights"]
    assert pytest.approx(sum(w), rel=1e-9) == 1.0
    assert all(x > 0 for x in w)
    for m in man["members"]:
        assert (REPO / "models" / m["checkpoint"]).exists()
    # the declared weight vector must be the one that was actually averaged
    assert w == pytest.approx([m["weight"] for m in man["members"]], rel=1e-9)


def test_api_reports_the_grid_it_serves():
    """/api/metrics must publish the provenance of the grid, not a different one."""
    if not MANIFEST.exists():
        pytest.skip("layer1_manifest.json absent")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metrics_loader import load_metrics

    m = load_metrics(REPO / "outputs" / "metrics")
    prov = json.loads(MANIFEST.read_text())
    api = m["layer1_provenance"]
    assert api is not None, "/api/metrics dropped layer1_provenance"
    assert api["grid_sha256"] == prov["artefact"]["sha256"]
    assert api["members"] == prov["predictor"]["members"]
    assert api["max_mm"] == pytest.approx(prov["values"]["max_mm"], rel=1e-6)


def test_selection_declares_the_deployed_model():
    """ablation.json must state which row is deployed and how it was chosen."""
    p = REPO / "outputs" / "metrics" / "ablation.json"
    if not p.exists():
        pytest.skip("ablation.json absent")
    ab = json.loads(p.read_text())
    sel = ab["selection"]
    assert sel["criterion"] == "val MAE"
    assert sel["tie_window_mm"] > 0
    assert sel["row"] in ab["results"]["val"], "selected row has no val metrics"
    assert sel["row"] in ab["results"]["test"], "selected row has no test metrics"
    if sel["row"] == "EF":
        assert ab.get("ensemble"), "EF selected but no ensemble block recorded"
        assert sel["deployed"].startswith("ensemble")
    rows = ab["rows"][sel["row"]]
    assert rows["label"] == sel["label"]
