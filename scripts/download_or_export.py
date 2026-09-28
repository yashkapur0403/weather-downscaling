"""
Download all raw data needed for Layer 1. Failsafe: every step is independent;
a failure prints the exact reason and the missing file, and the rest of the
pipeline can run once the file is placed manually.

  - IMD 0.25 deg daily rainfall   -> data/raw/imd/ind<YEAR>_rfp25.nc
    (fetched via IMD's own form POST, same as the official web page)
  - CHIRPS 0.05 deg daily (p05)   -> data/raw/chirps/chirps-v2.0.<YYYY.MM>.days_p05.nc
    (monthly global files from UC Santa Barbara CHC)
  - DEM (SRTM 30 m, Terrarium PNG tiles hosted on AWS Open Data) ->
    data/raw/dem/dem_roi.npz   (elev_m, transform) over the ROI at ~30 m/px

If Earth Engine authentication is unavailable (it is not needed for any of the
above), see scripts/gee_export.js for the optional GEE alternative path.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

CHIRPS_URL = ("https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/"
              "netcdf/p05/by_month/chirps-v2.0.{ym}.days_p05.nc")
IMD_URL = "https://www.imdpune.gov.in/cmpg/Griddata/RF25.php"

TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

# ---- ERA5-Land auxiliary context (Open-Meteo archive API; no auth needed) --
# Daily aggregates are computed server-side over Asia/Kolkata days (physically
# sensible), sampled nearest-neighbour at IMD coarse cell centers.
# Note: ERA5-Land has no 10m wind -> wind comes from ERA5 (0.25 deg).


def era5_sample_points(roi: dict, step_deg: float | None = None):
    """Sampling grid for ERA5 context on a degree lattice (+1-cell margin).

    The step comes from config.ERA5_STEP_DEG[region] (1.0 deg for deccan,
    0.5 deg for the pilot) because the free Open-Meteo archive API rations
    request weight by (locations x days); temperature, dewpoint and wind
    daily aggregates are smooth synoptic fields, so a coarser lattice keeps
    their large-scale signal while staying inside the quota. Coordinates are
    snapped to exact step multiples so the returned lattice is clean (no
    float drift) and reproducible across runs.
    """
    import numpy as np
    if step_deg is None:
        step_deg = float(getattr(config, "ERA5_STEP_DEG", {}).get(
            getattr(config, "REGION_DEFAULT", ""), 0.5))
    la0 = np.floor((roi["lat_min"] - step_deg) / step_deg) * step_deg
    lo0 = np.floor((roi["lon_min"] - step_deg) / step_deg) * step_deg
    lats = np.arange(la0, roi["lat_max"] + step_deg, step_deg)
    lons = np.arange(lo0, roi["lon_max"] + step_deg, step_deg)
    return [(round(float(la), 4), round(float(lo), 4))
            for la in lats for lo in lons]


def era5_year_windows(start: str, end: str, max_days: int = 123,
                      monsoon_clip: bool | None = None):
    """Per-year fetch windows of at most max_days days inside [start, end].

    The free Open-Meteo archive API rations request weight by
    (locations x days), so each request is bounded. With monsoon_clip (the
    default: True when start/end equal the study defaults) each year is
    clipped to config.MONSOON_START..MONSOON_END (Jun 1 - Sep 30 = 122 days,
    the study design) so quota is not spent on days the pipeline never uses;
    custom ranges are fetched in full (split into consecutive chunks).
    """
    from datetime import date as _date, timedelta as _td
    if monsoon_clip is None:
        monsoon_clip = (start == config.START_DATE_DEFAULT and
                        end == config.END_DATE_DEFAULT)
    ms = (int(config.MONSOON_START[1:3]), int(config.MONSOON_START[4:6]))
    me = (int(config.MONSOON_END[1:3]), int(config.MONSOON_END[4:6]))
    d0 = _date.fromisoformat(start)
    d1 = _date.fromisoformat(end)
    out = {}
    y = d0.year
    while y <= d1.year:
        a = max(d0, _date(y, 1, 1))
        b = min(d1, _date(y, 12, 31))
        if monsoon_clip:
            a = max(a, _date(y, ms[0], ms[1]))
            b = min(b, _date(y, me[0], me[1]))
        chunks = []
        if a <= b:
            s = a
            while s <= b:
                e = min(b, s + _td(days=max_days - 1))
                chunks.append((s.isoformat(), e.isoformat()))
                s = e + _td(days=1)
        out[y] = chunks
        y += 1
    return out


def fetch_era5(roi: dict, start: str, end: str,
               region: str | None = None) -> Path | None:
    """Fetch daily ERA5-Land/ERA5 context, one request per model per window.

    The free Open-Meteo archive API enforces an hourly request/weight budget
    (locations x days), so: windows are capped at ~123 days, the lattice step
    comes from config.ERA5_STEP_DEG (1.0 deg for deccan), and results are
    cached PER YEAR AND REGION (era5_year_<Y>_<region>.npz). Each re-run
    skips cached years and retries only the missing ones; holdout years
    (test, then val, then train) are fetched first so a partial fetch is
    still scientifically useful. Sleeps config.ERA5_REQUEST_SLEEP seconds
    between calls.

    temperature_2m_mean/temperature_2m_max/dew_point_2m_mean come from the
    ERA5_LAND model; wind_speed_10m_mean from ERA5 (Land has no 10m wind).
    """
    import time as _time
    import numpy as np
    tag = region or config.REGION_DEFAULT
    step = float(getattr(config, "ERA5_STEP_DEG", {}).get(tag, 0.5))
    sleep_s = int(getattr(config, "ERA5_REQUEST_SLEEP", 20))
    out_dir = config.RAW_ERA5
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"era5_daily_{tag}.npz"
    pts = era5_sample_points(roi, step)
    lats = sorted({p[0] for p in pts})
    lons = sorted({p[1] for p in pts})
    windows = era5_year_windows(start, end)
    # fetch order: test years, then val years, then train years
    priority = (config.SPLIT_YEARS["test"] + config.SPLIT_YEARS["val"] +
                config.SPLIT_YEARS["train"])
    years = [y for y in priority if y in windows and windows[y]]
    print(f"[era5:{tag}] {len(pts)} pts on a {step}-deg lattice x years "
          f"{years} (holdout first; windows {windows[years[0]] if years else '-'}); "
          f"caching in {out_dir}")

    def request(model: str, daily_vars: str, y: int,
                ws: str, we: str) -> dict | None:
        lat_q = ",".join(str(p[0]) for p in pts)
        lon_q = ",".join(str(p[1]) for p in pts)
        url = (f"{config.ERA5_API}?latitude={lat_q}&longitude={lon_q}"
               f"&start_date={ws}&end_date={we}"
               f"&daily={daily_vars}&timezone={config.ERA5_TIMEZONE}"
               f"&models={model}")
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=(15, 180))
                if r.status_code == 429:
                    print(f"[era5:{model}:{y}] 429 quota - will retry on the "
                          "next run (results are cached per year)")
                    return None
                r.raise_for_status()
                return r.json()
            except Exception as e:
                print(f"[era5:{model}:{y}] attempt {attempt + 1} failed: "
                      f"{type(e).__name__}: {e}")
                _time.sleep(10)
        return None

    def parse(model: str, daily_vars: str, data):
        if data is None:
            return {}
        if isinstance(data, dict):
            data = [data]
        res = {}
        for item in data:
            key = (round(float(item["latitude"]), 4),
                   round(float(item["longitude"]), 4))
            dates = item["daily"]["time"]
            res[key] = {var: dict(zip(dates, item["daily"][var]))
                        for var in daily_vars.split(",")}
        return res

    got_years = []
    for y in years:
        cache = out_dir / f"era5_year_{y}_{tag}.npz"
        if cache.exists():
            print(f"[era5] {cache.name} already cached")
            got_years.append(y)
            continue
        chunks = windows[y]
        land_all, wind_all = {}, {}
        ok = True
        for (ws, we) in chunks:
            land = parse("era5_land", config.ERA5_DAILY_VARS["era5_land"],
                         request("era5_land", config.ERA5_DAILY_VARS["era5_land"],
                                 y, ws, we))
            _time.sleep(sleep_s)
            wind = parse("era5", config.ERA5_DAILY_VARS["era5"],
                         request("era5", config.ERA5_DAILY_VARS["era5"],
                                 y, ws, we))
            _time.sleep(sleep_s)
            if not land:
                ok = False
                break
            land_all.update(land)
            wind_all.update(wind)
        if not ok:
            continue
        land, wind = land_all, wind_all
        # snap requested grid to returned coordinates and assemble dense arrays
        land_keys = sorted(land)
        wind_keys = sorted(wind) if wind else []

        def snap(t, pool):
            return min(pool, key=lambda p: (p[0] - t[0]) ** 2 +
                       (p[1] - t[1]) ** 2)

        dates = sorted(next(iter(land.values()))["temperature_2m_mean"])
        arrs = {v: np.full((len(dates), len(lats), len(lons)), np.nan,
                           dtype="float32")
                for v in ("t2m_mean", "t2m_max", "dewp", "wind")}
        di = {d: i for i, d in enumerate(dates)}
        for la_i, la in enumerate(lats):
            for lo_i, lo in enumerate(lons):
                lk = snap((la, lo), land_keys)
                for src, dst in ((land[lk]["temperature_2m_mean"], "t2m_mean"),
                                 (land[lk]["temperature_2m_max"], "t2m_max"),
                                 (land[lk]["dew_point_2m_mean"], "dewp")):
                    for d, v in src.items():
                        if v is not None:
                            arrs[dst][di[d], la_i, lo_i] = v
                if wind:
                    wk = snap((la, lo), wind_keys)
                    for d, v in wind[wk]["wind_speed_10m_mean"].items():
                        if v is not None:
                            arrs["wind"][di[d], la_i, lo_i] = v
        np.savez_compressed(cache, dates=np.array(dates), lat=np.array(lats),
                            lon=np.array(lons),
                            t2m_mean=arrs["t2m_mean"], t2m_max=arrs["t2m_max"],
                            dewp=arrs["dewp"], wind=arrs["wind"])
        nan_frac = float(np.isnan(arrs["t2m_mean"]).mean())
        print(f"[era5] {y} OK -> {cache.name} ({len(dates)} days, "
              f"NaN {nan_frac:.4f})")
        got_years.append(y)

    # assemble the combined file from ALL cached years (any run, any order)
    cached = sorted(y for y in years
                    if (out_dir / f"era5_year_{y}_{tag}.npz").exists())
    if not cached:
        print("[era5] no years fetched (quota exhausted?) - re-run later; "
              "pipeline continues without ERA5 channels")
        return None
    all_dates, parts = [], {v: [] for v in ("t2m_mean", "t2m_max", "dewp", "wind")}
    for y in cached:
        z = np.load(out_dir / f"era5_year_{y}_{tag}.npz")
        all_dates += [str(d) for d in z["dates"]]
        for v in parts:
            parts[v].append(z[v])
    order = np.argsort(all_dates)
    dates_sorted = [all_dates[i] for i in order]
    np.savez_compressed(out, dates=np.array(dates_sorted),
                        lat=np.array(lats), lon=np.array(lons),
                        **{v: np.concatenate(parts[v])[order] for v in parts})
    print(f"[era5] combined -> {out} ({len(dates_sorted)} days, "
          f"years {cached})")
    if set(cached) != set(years):
        print(f"[era5] NOTE: years {[y for y in years if y not in cached]} "
              "still missing - re-run download_or_export.py later")
    return out


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n)
    return min(x, n - 1), min(y, n - 1)


def fetch_imd(year: int) -> Path | None:
    out = config.RAW_IMD / f"ind{year}_rfp25.nc"
    if out.exists() and out.stat().st_size > 1_000_000:
        print(f"[imd] {out.name} already present ({out.stat().st_size/1e6:.1f} MB)")
        return out
    if out.exists() and out.stat().st_size == 0:
        out.unlink()  # remove previous empty attempt
    import time as _time
    for attempt in range(3):
        print(f"[imd] requesting {year} via IMD form POST "
              f"(attempt {attempt + 1}/3) ...")
        try:
            r = requests.post(IMD_URL, data={"RF25": str(year)},
                              timeout=(15, 300), stream=True,
                              headers={  # IMD rejects non-browser UAs (8-byte reply)
                                  "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                                "Chrome/120.0 Safari/537.36",
                                  "Referer": "https://www.imdpune.gov.in/cmpg/Griddata/"
                                             "Rainfall_25_NetCDF.html",
                              })
            r.raise_for_status()
            ctype = r.headers.get("content-type", "")
            first = next(r.iter_content(8), b"")
            if not first.startswith(b"CDF") and "netcdf" not in ctype and "octet" not in ctype:
                print(f"[imd] FAILED: server returned {ctype!r} (likely an HTML "
                      f"error page). Save the file manually as {out}")
                return None
            with open(out, "wb") as f:
                f.write(first)
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            got = out.stat().st_size
            if got >= 1_000_000:
                print(f"[imd] OK -> {out} ({got/1e6:.1f} MB)")
                return out
            out.unlink()
            print(f"[imd] got only {got} bytes for {year} - IMD throttles "
                  "rapid repeated downloads")
        except Exception as e:
            if out.exists() and out.stat().st_size < 1_000_000:
                out.unlink()
            print(f"[imd] attempt failed: {type(e).__name__}: {e}")
        # fallback: the POST response leaks the server-side path via
        # Content-disposition (RF25/ind<Y>_rfp25.nc); it is directly fetchable
        try:
            direct = requests.get(f"{IMD_URL.rsplit('/', 1)[0]}/RF25/"
                                  f"ind{year}_rfp25.nc", timeout=(15, 300),
                                  stream=True,
                                  headers={"User-Agent": "Mozilla/5.0"})
            if direct.status_code == 200 and \
                    int(direct.headers.get("content-length", "0")) > 1_000_000:
                with open(out, "wb") as f:
                    for chunk in direct.iter_content(1 << 20):
                        f.write(chunk)
                print(f"[imd] OK via direct path -> {out} "
                      f"({out.stat().st_size/1e6:.1f} MB)")
                return out
        except Exception:
            pass
        if attempt < 2:
            wait = 75
            print(f"[imd] cooling down {wait}s before retrying year {year}")
            _time.sleep(wait)
    print(f"[imd] GIVING UP on {year} for now. Re-run download_or_export.py "
          f"later, or download year {year} manually from "
          f"https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html "
          f"and save as {out}")
    return None


def fetch_chirps_months(year: int, months: list[int]) -> list[Path]:
    got = []
    for m in months:
        ym = f"{year}.{m:02d}"
        out = config.RAW_CHIRPS / f"chirps-v2.0.{ym}.days_p05.nc"
        if out.exists() and out.stat().st_size > 1_000_000:
            print(f"[chirps] {out.name} already present")
            got.append(out)
            continue
        url = CHIRPS_URL.format(ym=ym)
        print(f"[chirps] downloading {url}")
        try:
            with requests.get(url, timeout=(15, 600), stream=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                done = 0
                with open(out, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if total and done % (20 << 20) < (1 << 20):
                            print(f"         {done/1e6:7.1f}/{total/1e6:.1f} MB")
            print(f"[chirps] OK -> {out} ({out.stat().st_size/1e6:.1f} MB)")
            got.append(out)
        except Exception as e:
            if out.exists():
                out.unlink()  # remove partial file
            print(f"[chirps] FAILED: {type(e).__name__}: {e}\n"
                  f"      Missing file: {out}\n"
                  f"      Get it manually from {url} (or the GEE export in "
                  f"scripts/gee_export.js) and place it there.")
    return got


def fetch_dem(roi: dict, region: str | None = None) -> Path | None:
    tag = region or config.REGION_DEFAULT
    out = config.RAW_DEM / f"dem_roi_{tag}.npz"
    if out.exists():
        print(f"[dem] {out} already present")
        return out
    legacy = config.RAW_DEM / "dem_roi.npz"   # pilot file (western_ghats box)
    if tag == "western_ghats" and legacy.exists():
        print(f"[dem] {legacy} already present (pilot ROI)")
        return legacy
    import numpy as np
    from PIL import Image
    import io as _io

    # zoom 10 (~76 m/px here) covers the whole deccan box with a manageable
    # tile count; the pilot (western_ghats) ROI still uses zoom 11 (~38 m/px).
    z = 11 if tag == "western_ghats" else 10
    # pad by 0.25 deg: the fine grid extends 0.125 deg beyond the ROI on every
    # side (fine cells tile the full coarse-cell area), and tile quantization
    # can shave a bit more
    pad = 0.25
    x0, y0 = lonlat_to_tile(roi["lon_min"] - pad, roi["lat_max"] + pad, z)
    x1, y1 = lonlat_to_tile(roi["lon_max"] + pad, roi["lat_min"] - pad, z)
    n = 2 ** z
    x1, y1 = min(x1, n - 1), min(y1, n - 1)
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    print(f"[dem] fetching {nx}x{ny} terrarium tiles at zoom {z} "
          f"({nx * ny} tiles) ...")
    W, H = 256 * nx, 256 * ny
    canvas = np.zeros((H, W), dtype="float32")

    def grab(txy):
        tx, ty = txy
        url = TERRARIUM.format(z=z, x=tx, y=ty)
        last_err = None
        for attempt in range(3):
            try:
                r = sess.get(url, timeout=30)
                r.raise_for_status()
                img = Image.open(_io.BytesIO(r.content)).convert("RGB")
                return tx, ty, np.asarray(img, dtype="float32"), None
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
        return tx, ty, None, last_err

    from concurrent.futures import ThreadPoolExecutor
    sess = requests.Session()
    tiles = [(tx, ty) for ty in range(y0, y1 + 1) for tx in range(x0, x1 + 1)]
    failures = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        for i, (tx, ty, arr, err) in enumerate(ex.map(grab, tiles)):
            if err is not None:
                failures.append((tx, ty, err))
                continue
            yy = (ty - y0) * 256
            xx = (tx - x0) * 256
            canvas[yy:yy + 256, xx:xx + 256] = (
                arr[:, :, 0] * 256.0 + arr[:, :, 1] + arr[:, :, 2] / 256.0 - 32768.0)
            if (i + 1) % 100 == 0:
                print(f"[dem]   {i + 1}/{len(tiles)} tiles")
    if failures:
        print(f"[dem] FAILED on {len(failures)} tile(s), e.g. "
              f"{failures[0][0]},{failures[0][1]}: {failures[0][2]}")
        print(f"      The DEM will be missing; preprocess.py will fail "
              f"until {out} exists (or place a GeoTIFF named "
              f"dem_roi.tif in {config.RAW_DEM}).")
        return None

    # georeference: Web Mercator -> lat/lon, crop to ROI
    def tile2lon(tx, z):
        return tx / n * 360.0 - 180.0

    def tile2lat(ty, z):
        lat_r = math.atan(math.sinh(math.pi * (1 - 2 * ty / n)))
        return math.degrees(lat_r)

    lon_l = np.linspace(tile2lon(x0, z), tile2lon(x1 + 1, z), W, dtype="float64")
    lat_t = np.linspace(tile2lat(y0, z), tile2lat(y1 + 1, z), H, dtype="float64")

    lat_max, lat_min = roi["lat_max"], roi["lat_min"]
    lon_min, lon_max = roi["lon_min"], roi["lon_max"]
    ci = np.clip(np.searchsorted(lon_l, [lon_min, lon_max]), 0, W)
    ri = np.clip(np.searchsorted(-lat_t, [-lat_max, -lat_min]), 0, H)
    elev = canvas[ri[0]:ri[1], ci[0]:ci[1]]
    lon_c = lon_l[ci[0]:ci[1]]
    lat_c = lat_t[ri[0]:ri[1]]

    np.savez_compressed(out, elev=elev, lat=lat_c, lon=lon_c, zoom=z)
    print(f"[dem] OK -> {out} ({elev.shape[1]}x{elev.shape[0]} px, "
          f"elev {np.nanmin(elev):.0f}..{np.nanmax(elev):.0f} m)")
    return out


def month_pairs(start: str, end: str):
    """All (year, month) pairs between two YYYY-MM-DD dates, inclusive."""
    y0, m0 = int(start[:4]), int(start[5:7])
    y1, m1 = int(end[:4]), int(end[5:7])
    pairs = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        pairs.append((y, m))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lat-min", type=float, default=None)
    ap.add_argument("--lat-max", type=float, default=None)
    ap.add_argument("--lon-min", type=float, default=None)
    ap.add_argument("--lon-max", type=float, default=None)
    ap.add_argument("--start", default=config.START_DATE_DEFAULT,
                    help="YYYY-MM-DD (overlapping IMD/CHIRPS/ERA5 period)")
    ap.add_argument("--end", default=config.END_DATE_DEFAULT)
    ap.add_argument("--skip-imd", action="store_true")
    ap.add_argument("--skip-chirps", action="store_true")
    ap.add_argument("--skip-dem", action="store_true")
    ap.add_argument("--skip-era5", action="store_true")
    ap.add_argument("--months", default=None,
                    help="comma list of months to download CHIRPS for "
                         "(default: config.CHIRPS_MONTHS = monsoon months)")
    ap.add_argument("--region", default=None,
                    help="study region name from config.REGIONS "
                         "(default: config.REGION_DEFAULT)")
    args = ap.parse_args()

    roi = config.clip_roi(vars(args))
    region = args.region or config.REGION_DEFAULT
    start, end = args.start, args.end
    ym_pairs = month_pairs(start, end)
    years = sorted({y for (y, _) in ym_pairs})
    months_cfg = (config.CHIRPS_MONTHS if args.months is None
                  else [int(m) for m in args.months.split(",")])
    ym_pairs_ch = [(y, m) for (y, m) in ym_pairs if m in months_cfg]

    print(f"Region: {region}\nROI: {roi}\nPeriod: {start} .. {end}\n")
    ok = {"imd": True, "chirps": True, "dem": True, "era5": True}
    if not args.skip_imd:
        for y in years:
            if fetch_imd(y) is None:
                ok["imd"] = False
    if not args.skip_chirps:
        # group months by year so multi-year windows fetch every needed file
        by_year = {}
        for y, m in ym_pairs_ch:
            by_year.setdefault(y, []).append(m)
        got_any = True
        for y, ms in sorted(by_year.items()):
            if not fetch_chirps_months(y, ms):
                got_any = False
        ok["chirps"] = got_any
    if not args.skip_dem:
        if fetch_dem(roi, region) is None:
            ok["dem"] = False
    if not args.skip_era5:
        try:
            if fetch_era5(roi, start, end, region) is None:
                ok["era5"] = False
        except Exception as e:
            print(f"[era5] FAILED: {e}\n      Layer-1 can still run without "
                  f"ERA5 channels (2-channel fallback preserved).")
            ok["era5"] = False

    print("\n==== SUMMARY ====")
    for k, v in ok.items():
        print(f"  {k:7s}: {'OK' if v else 'MISSING FILES - see messages above'}")
    if all(ok.values()):
        print("All raw data present. Next: python scripts/inspect_data.py")


if __name__ == "__main__":
    main()
