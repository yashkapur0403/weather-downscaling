"""
One-off: write outputs/layer2/panchayat_index.csv (panchayat_id, lat, lon) from the LGD polygons so the
backend can place every panchayat exactly. Needs geopandas and data/.../LGD_Panchayats.parquet
(python scripts/fetch_lgd_panchayats.py). UNTESTED here (the parquet is not in git) - check the printed sample.

    python build_panchayat_index.py --repo ..
"""
import argparse, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="..")
a = ap.parse_args()
root = Path(a.repo).resolve()
sys.path.insert(0, str(root / "scripts"))
import geopandas as gpd
import layer2_config as cfg
import pandas as pd

gdf = gpd.read_parquet(cfg.DEFAULT_PARQUET)
summ = pd.read_csv(root / "outputs" / "layer2" / "panchayat_summary.csv", usecols=["panchayat_id"])
id_col = next((c for c in ("panchayat_id", "gpcode", "lgd_code") if c in gdf.columns), None)
if id_col is None:
    sys.exit(f"no id column found in {list(gdf.columns)}")
proj = gdf.to_crs(cfg.METRIC_CRS)
cent = proj.geometry.representative_point().to_crs(4326)   # inside the polygon, unlike a plain centroid
out = pd.DataFrame({"panchayat_id": gdf[id_col].astype(int), "lat": cent.y, "lon": cent.x})
out = out[out.panchayat_id.isin(summ.panchayat_id)].drop_duplicates("panchayat_id")
dst = root / "outputs" / "layer2" / "panchayat_index.csv"
out.to_csv(dst, index=False)
print(f"wrote {len(out)} rows -> {dst}")
print(out.head())
