"""Fetch + prepare the LGD Gram-Panchayat boundaries that Layer 2 needs.

Downloads the official-LGD-derived Gram-Panchayat GeoParquet (CC0), verifies its
SHA-256, and arrives at the exact file `layer2_panchayat_mapping.py` reads:

    data/raw/administrative/panchayat/LGD_Panchayats.parquet

By default it clips to the 15 states covered by the Deccan grid (+ the Deccan
ROI bbox) so Layer 2 loads fast and stays light. Use --full to keep all-India.

Usage (Windows Git Bash):
    python scripts/fetch_lgd_panchayats.py           # clip to the deccan ROI (default)
    python scripts/fetch_lgd_panchayats.py --full    # keep all-India, no clipping
    python scripts/fetch_lgd_panchayats.py --verify  # just validate the local file

Source
------
Release tag `admin/panchayats` of https://github.com/yashveeeeeeer/india-geodata ,
aggregating https://github.com/ramSeraph/indian_admin_boundaries , which sources
the official Local Government Directory (LGD, Ministry of Panchayati Raj) plus
ISRO Bhuvan. Upstream authority for panchayat boundaries is LGD
(https://lgdirectory.gov.in). License: CC0-1.0 (public domain).

The file is NOT committed to git (data/raw/ is ignored) — run this script again
on a fresh clone. It is a ~368 MB download.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import config  # noqa: E402
from shapely.geometry import box  # noqa: E402

# --- exact asset (name on the release is lowercase 'panchayats') -------------
URL = ("https://github.com/yashveeeeeeer/india-geodata/releases/download/"
       "admin/panchayats/LGD_panchayats.parquet")
SHA256 = "d1585c1686b204de9b87fa0c5e7996f31907ba9b8eed1fc8a34ef04848326400"
EXPECTED_BYTES = 368_147_580

OUT_DIR = ROOT / "data" / "raw" / "administrative" / "panchayat"
OUT = OUT_DIR / "LGD_Panchayats.parquet"

# Columns layer2_panchayat_mapping.load_panchayats() requires.
REQUIRED = ("gpcode", "gpname", "stname", "dtname", "blklgdcode", "blkname", "geometry")

# LGD `stname` spellings for the 15 states the Deccan grid covers
# (admin-map names in brackets; the two Daman/DNH UTs are one LGD entry).
DECCAN_STATES = {
    "ANDHRA PRADESH",            # AndhraPradesh
    "CHHATTISGARH",              # Chhattisgarh
    "GOA",                       # Goa
    "GUJARAT",                   # Gujarat
    "KARNATAKA",                 # Karnataka
    "KERALA",                    # Kerala
    "MADHYA PRADESH",            # MadhyaPradesh
    "MAHARASHTRA",               # Maharashtra
    "PUDUCHERRY",                # Puducherry
    "RAJASTHAN",                 # Rajasthan
    "TAMIL NADU",                # TamilNadu
    "TELANGANA",                 # Telangana
    "UTTAR PRADESH",             # UttarPradesh
    "DADRA,NAGAR HAVELI,DAMAN & DIU",  # DadraandNagarHaveli + DamanandDiu
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: Path) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"[fetch] downloading {URL}")
    with urllib.request.urlopen(URL) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r[fetch] {done/1e6:7.1f} / {total/1e6:.1f} MB "
                      f"({100*done/total:5.1f}%)", end="", flush=True)
    print()
    got = _sha256(part)
    if got != SHA256:
        part.unlink(missing_ok=True)
        sys.exit(f"[ERROR] sha256 mismatch\n  expected {SHA256}\n  got      {got}")
    print("[fetch] sha256 OK")
    part.replace(dest)


def verify(path: Path) -> dict:
    """Load and check the parquet the way Layer 2 will."""
    import geopandas as gpd

    g = gpd.read_parquet(path)
    missing = [c for c in REQUIRED if c not in g.columns]
    if missing:
        sys.exit(f"[ERROR] {path.name} is missing required columns: {missing}")
    epsg = g.crs.to_epsg() if g.crs is not None else None
    if epsg != 4326:
        sys.exit(f"[ERROR] {path.name} CRS is not EPSG:4326 (got {g.crs})")
    return {"features": len(g), "gps": int(g["gpcode"].nunique()),
            "states": int(g["stname"].nunique()), "size": path.stat().st_size}


def clip(src: Path, dst: Path) -> None:
    import geopandas as gpd

    roi = config.region_roi("deccan")
    bbox = box(roi["lon_min"], roi["lat_min"], roi["lon_max"], roi["lat_max"])
    keep = list(REQUIRED)

    g = gpd.read_parquet(src)
    sn = g["stname"].astype(str).str.strip().str.upper()
    found = set(sn.unique())
    absent = DECCAN_STATES - found
    if absent:
        print(f"[clip] note: states not present in source: {sorted(absent)}")
    sub = g[sn.isin(DECCAN_STATES)]
    sub = sub[sub.intersects(bbox)].to_crs("EPSG:4326")[keep]
    sub.to_parquet(dst, compression="zstd", compression_level=9)
    print(f"[clip] {len(g)} -> {len(sub)} features "
          f"({sub['gpcode'].nunique()} GPs) for the deccan ROI")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true",
                    help="keep the all-India file as-is (no state/ROI clipping)")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the output already exists")
    ap.add_argument("--verify", action="store_true",
                    help="only validate the existing output file")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify:
        if not OUT.exists():
            sys.exit(f"[ERROR] not found: {OUT}\nRun without --verify to fetch it.")
        info = verify(OUT)
        print(f"[ok] {OUT}\n     {info['features']:,} features | {info['gps']:,} GPs "
              f"| {info['states']} states | {info['size']/1e6:.1f} MB | EPSG:4326")
        return

    if OUT.exists() and not args.force:
        print(f"[skip] already present: {OUT}  (use --force to re-download)")
        verify(OUT)
        return

    full = OUT_DIR / "_LGD_panchayats_full.parquet"
    if not full.exists():
        download(full)
    else:
        print(f"[skip] full download already cached: {full}")

    if args.full:
        full.replace(OUT)
    else:
        clip(full, OUT)
        full.unlink(missing_ok=True)

    info = verify(OUT)
    print(f"[done] {OUT}\n       {info['features']:,} features | {info['gps']:,} GPs "
          f"| {info['states']} states | {info['size']/1e6:.1f} MB | EPSG:4326")
    print("       Layer 2 is now unblocked: run "
          "scripts/layer2_panchayat_mapping.py")


if __name__ == "__main__":
    main()
