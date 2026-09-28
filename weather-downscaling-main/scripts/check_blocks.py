"""Diagnose Layer 2 coverage: which Panchayats sit on the Layer 1 grid edge."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import config  # noqa: E402
from layer2_panchayat_mapping import (  # noqa: E402
    discover_layer1_path, load_layer1, grid_to_points, load_panchayats,
)
from shapely.geometry import box


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=None, help="Layer 1 .npz or .npy")
    ap.add_argument("--block", default="MANGALURU")
    ap.add_argument("--panchayats", default=str(
        config.RAW_PANCHAYAT / "LGD_Panchayats.parquet"))
    args = ap.parse_args()

    field = load_layer1(discover_layer1_path(Path(args.input) if args.input else None))
    points = grid_to_points(field)
    dlat = float(abs(field.latitude[1] - field.latitude[0])) if len(field.latitude) > 1 else 0.05
    dlon = float(abs(field.longitude[1] - field.longitude[0])) if len(field.longitude) > 1 else 0.05
    roi_box = box(field.longitude.min() - dlon / 2, field.latitude.min() - dlat / 2,
                  field.longitude.max() + dlon / 2, field.latitude.max() + dlat / 2)
    gps = load_panchayats(Path(args.panchayats), args.block, None, roi_box=None)

    print(f"Layer 1 lat: {field.latitude.min():.4f} to {field.latitude.max():.4f}")
    print(f"Layer 1 lon: {field.longitude.min():.4f} to {field.longitude.max():.4f}")
    print(f"ROI box intersects block: {gps.intersects(roi_box).sum()} / {len(gps)}")
    print(f"Block bounds: {gps.total_bounds}")
    print()
    for _, row in gps.iterrows():
        b = row.geometry.bounds
        near = points.cx[b[0] - 0.1:b[2] + 0.1, b[1] - 0.1:b[3] + 0.1]
        in_gp = near[near.geometry.within(row.geometry)]
        print(
            f"GP {row['gpcode']!s:10s} {str(row['gpname'])[:20]:20s} "
            f"lat {b[1]:.3f}-{b[3]:.3f} | pts_within={len(in_gp)}"
        )


if __name__ == "__main__":
    main()
