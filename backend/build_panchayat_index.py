"""
Build outputs/layer2/panchayat_index.csv (panchayat_id, lat, lon) from the LGD
Panchayat polygons, so the backend samples the U-Net grid at a point INSIDE each
Panchayat's own boundary (not a shared block/district centroid).

Needs geopandas + shapely and
    data/raw/administrative/panchayat/LGD_Panchayats.parquet
(fetch with `python scripts/fetch_lgd_panchayats.py`). The parquet is NOT in git.

    python backend/build_panchayat_index.py --repo .

Notes
- The id column is `gpcode`; blank/non-numeric codes are dropped.
- We use `representative_point()` (guaranteed inside the polygon) in a metric CRS,
  not a plain centroid (which can fall outside a concave polygon or in the sea).
- Rows are restricted to the ids present in panchayat_summary.csv.
"""
import argparse
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="..")
a = ap.parse_args()
root = Path(a.repo).resolve()
sys.path.insert(0, str(root / "scripts"))
import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402
import layer2_config as cfg  # noqa: E402

gdf = gpd.read_parquet(cfg.DEFAULT_PARQUET)

id_col = next((c for c in ("gpcode", "panchayat_id", "lgd_code") if c in gdf.columns), None)
if id_col is None:
    sys.exit(f"no id column found in {list(gdf.columns)}")

gdf = gdf.copy()
gdf["_pid"] = pd.to_numeric(gdf[id_col], errors="coerce")
before = len(gdf)
gdf = gdf[gdf["_pid"].notna()].copy()
gdf["_pid"] = gdf["_pid"].astype("int64")
print(f"parquet rows {before} -> {len(gdf)} with a numeric id")

# fix invalid geometries, then take a point guaranteed inside each polygon
geom = gdf.geometry
invalid = ~geom.is_valid
if invalid.any():
    print(f"repairing {int(invalid.sum())} invalid geometries with buffer(0)")
    gdf.loc[invalid, gdf.geometry.name] = gdf.loc[invalid, gdf.geometry.name].buffer(0)

proj = gdf.to_crs(cfg.METRIC_CRS)
cent = proj.geometry.representative_point().to_crs(4326)

out = pd.DataFrame({"panchayat_id": gdf["_pid"].to_numpy(), "lat": cent.y.to_numpy(), "lon": cent.x.to_numpy()})

summ = pd.read_csv(root / "outputs" / "layer2" / "panchayat_summary.csv", usecols=["panchayat_id", "mapping_method"])
mapped = set(summ.loc[summ["mapping_method"] != "unmapped", "panchayat_id"].astype(int))
out = out[out["panchayat_id"].isin(summ["panchayat_id"])]
out = out.drop_duplicates("panchayat_id").reset_index(drop=True)

dst = root / "outputs" / "layer2" / "panchayat_index.csv"
out.to_csv(dst, index=False)
covered = len(set(out["panchayat_id"]) & mapped)
print(f"wrote {len(out)} rows -> {dst}")
print(f"coverage of mapped Panchayats: {covered}/{len(mapped)}")
print(out.head())
