"""
Retry failed (NaN-row) SoilGrids batch points at low concurrency.

The batch downloader caches whole batches; rows that failed with HTTP 429 are
stored as NaN. This script re-queries ONLY the NaN rows, merges the results
back into the cached batches, and reports remaining NaNs. Restartable: only
broken rows are touched, so it can be re-run until clean.

Run:
  python scripts/retry_soil_nan.py --region deccan
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default=config.REGION_DEFAULT)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-rounds", type=int, default=3)
    args = ap.parse_args()
    region = args.region
    props = list(config.SOILGRID_PROPS)
    cache_dir = config.RAW_SOIL / "batches"
    files = sorted(cache_dir.glob(f"{region}_batch_*.npz"))
    if not files:
        sys.exit(f"no batch caches in {cache_dir}")

    import requests

    def query(lat: float, lon: float):
        q = "&".join(f"property={p}" for p in props)
        url = (f"{config.SOILGRID_API}?lat={lat:.4f}&lon={lon:.4f}&{q}"
               f"&depth={config.SOILGRID_DEPTH}&value=mean")
        for attempt in range(4):
            try:
                r = requests.get(url, timeout=(10, 60))
                print(f"  [{lat:.2f},{lon:.2f}] HTTP {r.status_code}", flush=True)
                if r.status_code == 200:
                    layers = r.json()["properties"]["layers"]
                    by_name = {L["name"]: L for L in layers}
                    row = []
                    for p in props:
                        L = by_name.get(p)
                        if L is None:
                            row.append(np.nan)
                            continue
                        m = L["depths"][0]["values"]["mean"]
                        row.append(float("nan") if m is None else float(m))
                    return row
                print(f"  [{lat:.2f},{lon:.2f}] HTTP {r.status_code}",
                      flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  [{lat:.2f},{lon:.2f}] EXC {type(e).__name__}: {e}"[:150],
                      flush=True)
            time.sleep(int(getattr(config, "SOIL_RETRY_WAIT", 15))
                       * (attempt + 1))
        return None

    # batch caches store values in the deterministic land-cell ordering used
    # by build_aux.build_soil: np.where(land) row-major on the IMD grid ->
    # (lat_index, lon_index) pairs; reconstruct the coordinates from there.
    from build_aux import region_context  # noqa: E402
    ctx = region_context(region)
    ii, jj = np.where(ctx["land"])
    pts_lat, pts_lon = ctx["imd_lat"][ii], ctx["imd_lon"][jj]
    nb = (len(ii) + config.SOIL_BATCH - 1) // config.SOIL_BATCH
    files = [cache_dir / f"{region}_batch_{b:04d}.npz" for b in range(nb)]

    from concurrent.futures import ThreadPoolExecutor
    for rnd in range(1, args.max_rounds + 1):
        todo_total = 0
        for b, fp in enumerate(files):
            if not fp.exists():
                continue
            z = dict(np.load(fp))
            vals = np.column_stack([z[p] for p in props])
            bad = np.where(np.isnan(vals).any(axis=1))[0]
            if bad.size == 0:
                continue
            todo_total += bad.size
            sl = slice(b * config.SOIL_BATCH,
                       min((b + 1) * config.SOIL_BATCH, len(ii)))
            lats = pts_lat[sl][bad]
            lons = pts_lon[sl][bad]
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                rows = list(ex.map(query, lats, lons))
            n_fixed = 0
            for k, row in zip(bad, rows):
                if row is not None and not any(np.isnan(v) for v in row):
                    vals[k] = row
                    n_fixed += 1
            for p_i, p in enumerate(props):
                z[p] = vals[:, p_i].astype("float32")
            np.savez_compressed(fp, **z)
            print(f"[round {rnd}] {fp.name}: {n_fixed}/{bad.size} rows fixed")
        print(f"[round {rnd}] {todo_total} broken rows processed")
        if todo_total == 0:
            break
        time.sleep(30)
    # final status
    tot = bad_rows = 0
    for fp in files:
        z = np.load(fp)
        vals = np.column_stack([z[p] for p in props])
        tot += vals.shape[0]
        bad_rows += int(np.isnan(vals).any(axis=1).sum())
    print(f"[status] {bad_rows} NaN rows remain of {tot} points "
          f"({100 * bad_rows / max(tot, 1):.2f}%)")


if __name__ == "__main__":
    main()
