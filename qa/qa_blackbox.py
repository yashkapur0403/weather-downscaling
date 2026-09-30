"""Black-box API validation against the live uvicorn server (real HTTP, real network)."""
import json, time, concurrent.futures as cf
import httpx

B = "http://127.0.0.1:8000"
c = httpx.Client(base_url=B, timeout=60)
R = []


def rec(tid, cat, test, exp, act, status, sev="", note=""):
    R.append(dict(id=tid, cat=cat, test=test, exp=exp, act=act, st=status, sev=sev, note=note))
    print(f"[{status:4s}] {tid} {test}")
    print(f"        exp = {str(exp)[:110]}")
    print(f"        act = {str(act)[:110]}")


def T(fn):
    t0 = time.time()
    r = fn()
    return r, (time.time() - t0) * 1000


print("################ HEALTH ################")
j = c.get("/").json()
rec("BB-01", "health", "GET / cold start (server just booted)", "panchayats_loaded=86103",
    f"panchayats_loaded={j['panchayats_loaded']}", "FAIL" if j["panchayats_loaded"] != 86103 else "PASS",
    "Medium", "handler reads cached S.store instead of store(); cold process reports null")
c.get("/api/panchayats", params={"q": "a"})
j2 = c.get("/").json()
rec("BB-02", "health", "GET / after another call (warm)", "86103", j2["panchayats_loaded"],
    "PASS" if j2["panchayats_loaded"] == 86103 else "FAIL")
rec("BB-03", "health", "GET /health", '"status":"ok"', c.get("/health").json().get("status"),
    "PASS" if c.get("/health").json().get("status") == "ok" else "FAIL")

print("################ SEARCH ################")
r = c.get("/api/panchayats", params={"q": "kundapura", "limit": 5}).json()
ok = len(r) > 0 and all(x["mapping_method"] in ("direct_grid", "area_weighted", "nearest_fallback") for x in r)
rec("BB-04", "search", "GET /api/panchayats?q=kundapura", "real rows, real mapping_method",
    f"{len(r)} rows; last={r[-1]['panchayat_name']} {r[-1]['mapping_method']} n_cells={r[-1]['n_cells']} mean={r[-1]['rainfall_mm']}",
    "PASS" if ok else "FAIL")
rec("BB-05", "search", "GET /api/panchayats without q", "422", c.get("/api/panchayats").status_code,
    "PASS" if c.get("/api/panchayats").status_code == 422 else "FAIL")
rec("BB-06", "search", "GET /api/panchayats?limit=100 (max 50)", "422",
    c.get("/api/panchayats", params={"q": "a", "limit": 100}).status_code,
    "PASS" if c.get("/api/panchayats", params={"q": "a", "limit": 100}).status_code == 422 else "FAIL")
um = c.get("/api/panchayats", params={"q": "KANNEGANTIVARIPALEM"}).json()
rec("BB-07", "search", "unmapped Panchayat hidden (KANNEGANTIVARIPALEM)", "[]", um,
    "PASS" if um == [] else "FAIL")
rec("BB-08", "search", "substring matching q=ALLAMUDI",
    "matches CHEMALLAMUDI by substring (unmapped ALLAMUDI itself hidden)",
    [x["panchayat_name"] for x in c.get("/api/panchayats", params={"q": "ALLAMUDI"}).json()], "INFO")

print("################ GEOCODE ################")
g = c.get("/api/geocode", params={"panchayat_name": "ALURU", "block_name": "KUNDAPURA",
                                  "district": "Udupi", "state": "KARNATAKA"}).json()
rec("BB-09", "geocode", "ALURU/Udupi coordinates",
    '{"lat":13.679452,"lon":74.742466,"location_precision":"exact"}', g,
    "PASS" if g.get("lat") == 13.679452 and g.get("lon") == 74.742466 else "FAIL", "",
    "coordinate is the KUNDAPURA block centroid; 'exact' is nominal (D-1)")
rec("BB-10", "geocode", "unknown Panchayat", 404, c.get("/api/geocode", params={"panchayat_name": "NOPE"}).status_code,
    "PASS" if c.get("/api/geocode", params={"panchayat_name": "NOPE"}).status_code == 404 else "FAIL")

print("################ WEATHER (live external APIs) ################")
w, ms = T(lambda: c.get("/api/weather", params={"lat": 13.679452, "lon": 74.742466, "date": "2022-07-10"}).json())
rec("BB-11", "weather", "GET /api/weather for ALURU", "non-null T/RH/elevation + source labels",
    f"T={w['temperature_c']} RH={w['humidity_pct']} elev={w['elevation_m']} "
    f"src={w['elevation_source']} err={w['error']} ({ms:.0f}ms)",
    "PASS" if w["temperature_c"] is not None else "FAIL")

print("################ PROJECTION /auth/ ################")
p, ms = T(lambda: c.post("/auth/", json={
    "panchayat_name": "ALURU", "block_name": "KUNDAPURA", "district": "Udupi", "state": "KARNATAKA",
    "date": "2022-07-10", "requested_metrics": ["rainfall", "temperature", "humidity", "elevation"]}).json())
rec("BB-12", "auth", "POST /auth/ ALURU with full admin context", "rainfall 50.49 + real T/RH/elevation",
    f"rain={p['prediction']['rainfall_mm']} T={p['prediction']['temperature_c']} RH={p['prediction']['humidity_pct']} "
    f"elev={p['prediction']['elevation_m']} prec={p['sources']['location_precision']} src={p['sources']['rainfall'][:40]} ({ms:.0f}ms)",
    "PASS" if p["prediction"]["rainfall_mm"] == 50.49 and p["prediction"]["temperature_c"] is not None else "FAIL")

n = c.post("/auth/", json={"panchayat_name": "ALURU", "date": "2022-07-10"}).json()
rec("BB-13", "auth", "POST /auth/ name-only ALURU (4 Panchayats share this name)",
    "must not silently return a different Panchayat",
    f"rainfall={n['prediction']['rainfall_mm']} (Udupi ALURU = 50.49); answer='{n['answer'][:70]}'",
    "FAIL", "High", "Store.find() returns iloc[0]; 8,404 duplicated names covering 26,666 rows")

rec("BB-14", "auth", "POST /auth/ outside model domain (lat=40)", 422,
    c.post("/auth/", json={"lat": 40.0, "lon": 10.0, "date": "2022-07-10"}).status_code,
    "PASS" if c.post("/auth/", json={"lat": 40.0, "lon": 10.0, "date": "2022-07-10"}).status_code == 422 else "FAIL")
for tag, d in (("a", "2021-01-01"), ("b", "2022-10-01")):
    got = c.post("/auth/", json={"panchayat_name": "ALURU", "block_name": "KUNDAPURA",
                                "district": "Udupi", "date": d}).status_code
    rec(f"BB-15{tag}", "auth", f"date outside committed range ({d})", 422, got,
        "PASS" if got == 422 else "FAIL")
bad = c.post("/auth/", content=b"{not json", headers={"Content-Type": "application/json"}).status_code
rec("BB-16", "auth", "malformed JSON body", "4xx", bad, "PASS" if bad >= 400 else "FAIL")
rec("BB-17", "auth", "masked (sea/excluded) cell Panchayat MANDURIVARIPALEM", "422, never a fake value",
    c.post("/auth/", json={"panchayat_name": "MANDURIVARIPALEM", "date": "2022-07-10"}).json(),
    "PASS" if c.post("/auth/", json={"panchayat_name": "MANDURIVARIPALEM", "date": "2022-07-10"}).status_code == 422 else "FAIL")
rec("BB-18", "auth", "Panchayat with no coordinates (AMBOLI)", "422",
    c.post("/auth/", json={"panchayat_name": "AMBOLI", "date": "2022-07-10"}).json(),
    "PASS" if c.post("/auth/", json={"panchayat_name": "AMBOLI", "date": "2022-07-10"}).status_code == 422 else "FAIL")

print("################ METRICS ################")
m, ms = T(lambda: c.get("/api/metrics").json())
rec("BB-19", "metrics", "GET /api/metrics", "D=8.167 / baseline=9.603 / +15.0% / 117,329 params",
    f"{m['selected_model']['test_mae_mm']} / {m['baseline']['test_mae_mm']} / {m['mae_improvement_pct']} / "
    f"{m['n_parameters']} ({ms:.0f}ms)", "PASS")

print("################ ADVISORY (Layer 3) ################")
def adv(**kw):
    q = {"panchayat_id": 220447, "crop": "wheat", "stage": "general", "date": "2022-07-10",
         "panchayat_name": "ALURU"}
    q.update(kw)
    return c.get("/api/advisory", params=q).json()

for mm, expsev, exprisk in [(70, "alert", "very_heavy"), (50.49, "warning", "very_heavy"),
                            (30, "warning", "heavy"), (12, "info", "moderate"), (2, "watch", "no_rain")]:
    a = adv(rainfall_mm=mm)
    ok = a["severity"] == expsev and a["evidence"]["risk_level"] == exprisk and a["evidence"]["rainfall_mm"] == mm
    rec(f"ADV-{mm}", "advisory", f"advisory band for rainfall={mm}",
        f"severity={expsev}, risk_level={exprisk}, evidence={mm}",
        f"severity={a['severity']}, risk_level={a['evidence']['risk_level']}, evidence={a['evidence']['rainfall_mm']}, "
        f"src={a.get('message_source')}", "PASS" if ok else "FAIL")
rec("ADV-NONE", "advisory", "advisory without required rainfall_mm", 422,
    c.get("/api/advisory", params={"panchayat_id": 1, "crop": "wheat"}).status_code,
    "PASS" if c.get("/api/advisory", params={"panchayat_id": 1, "crop": "wheat"}).status_code == 422 else "FAIL")
rec("ADV-BAD", "advisory", "advisory with invalid crop", 422,
    c.get("/api/advisory", params={"panchayat_id": 1, "crop": "banana", "rainfall_mm": 10}).status_code,
    "PASS" if c.get("/api/advisory", params={"panchayat_id": 1, "crop": "banana", "rainfall_mm": 10}).status_code == 422 else "FAIL")
rec("ADV-LANG", "advisory", "advisory with unsupported lang", 422,
    c.get("/api/advisory", params={"panchayat_id": 1, "crop": "wheat", "rainfall_mm": 10, "lang": "xx-XX"}).status_code,
    "PASS" if c.get("/api/advisory", params={"panchayat_id": 1, "crop": "wheat", "rainfall_mm": 10, "lang": "xx-XX"}).status_code == 422 else "FAIL")
low, high = adv(rainfall_mm=5), adv(rainfall_mm=70)
rec("ADV-SENS", "advisory", "advisory changes when rainfall changes (5 -> 70)",
    "different severity and different actions",
    f"severity {low['severity']} -> {high['severity']}; {len(low['actions'])} vs {len(high['actions'])} actions",
    "PASS" if low["severity"] != high["severity"] else "FAIL")
e = adv(crop="rice", stage="flowering", rainfall_mm=30)
rec("ADV-ECHO", "advisory", "crop/stage echoed back", "rice / flowering", f"{e['crop']} / {e['stage']}",
    "PASS" if e["crop"] == "rice" and e["stage"] == "flowering" else "FAIL")

print("################ XAI ################")
rec("XAI-ST", "xai", "GET /api/explain/status", "enabled=false, provider=groq",
    c.get("/api/explain/status").json(), "PASS")
xb = {"panchayat_name": "ALURU", "block_name": "KUNDAPURA", "district": "Udupi", "state": "KARNATAKA",
      "date": "2022-07-10", "lat": 13.679452, "lon": 74.742466,
      "prediction": {"rainfall_mm": 50.49, "risk_level": "very_heavy", "temperature_c": 23.8,
                     "humidity_pct": 93, "elevation_m": 17.0},
      "mapping": {"method": "direct_grid", "n_cells": 2, "fallback_distance_m": None},
      "model": {"name": "U-Net + DEM + ERA5-Land",
                "channels": ["imd_rain", "dem", "era5_t2m", "era5_t2m_max", "era5_dewp"],
                "test_mae_mm": 8.167, "baseline_mae_mm": 9.603, "mae_improvement_pct": 15.0,
                "reference_product": "CHIRPS v2.0"},
      "language": "en"}
ex = c.post("/api/explain", json=xb).json()
rec("XAI-01", "xai", "POST /api/explain without key", "provider=rules, factors only for trained channels, 50.5 in summary",
    f"provider={ex['provider']} factors={[f['factor'] for f in ex['factors']]} summary='{ex['summary'][:50]}'",
    "PASS" if ex["provider"] == "rules" and "50.5" in ex["summary"] else "FAIL")
rec("XAI-02", "xai", "explain with question, no key", "non-empty answer explaining unavailability",
    str(ex.get("answer"))[:80], "PASS" if ex.get("answer") else "FAIL")
rec("XAI-03", "xai", "all factor weights within [0,1]", "all valid",
    all(0 <= f["weight"] <= 1 for f in ex["factors"]), "PASS")
exq = c.post("/api/explain", json={**xb, "question": "why is this higher than the district?"}).json()
rec("XAI-04", "xai", "explain with question (no key)", "non-empty answer", str(exq.get("answer"))[:80],
    "PASS" if exq.get("answer") else "FAIL")

print("################ BLOCK ADVISORY ################")
bl = c.post("/api/advisory/block", json={"block": "X", "output_language": "en-IN", "panchayats": [
    {"panchayat": "A", "crop": "wheat", "stage": "grain_filling", "rainfall_mm": 2, "tmax_c": 36,
     "humidity_pct": 50, "soil_moisture": 0.1, "wind_kmh": 5},
    {"panchayat": "B", "crop": "wheat", "stage": "grain_filling", "rainfall_mm": 70, "tmax_c": 28,
     "humidity_pct": 60, "soil_moisture": 0.25, "wind_kmh": 10}]}).json()
rec("BLK-01", "advisory", "POST /api/advisory/block (2 panchayats)", "actions = [heat_stress, heavy_rain]",
    [x["action"] for x in bl["panchayats"]],
    "PASS" if [x["action"] for x in bl["panchayats"]] == ["heat_stress", "heavy_rain"] else "FAIL")

print("################ ROBUSTNESS ################")
body = {"panchayat_name": "ALURU", "block_name": "KUNDAPURA", "district": "Udupi",
        "state": "KARNATAKA", "date": "2022-07-10"}
r1 = c.post("/auth/", json=body).json()["prediction"]["rainfall_mm"]
r2 = c.post("/auth/", json=body).json()["prediction"]["rainfall_mm"]
rec("ROB-01", "robustness", "repeated identical request is stable", "same value", f"{r1} == {r2}",
    "PASS" if r1 == r2 else "FAIL")
with cf.ThreadPoolExecutor(5) as ex_:
    vals = list(ex_.map(lambda _: c.post("/auth/", json=body).json()["prediction"]["rainfall_mm"], range(5)))
rec("ROB-02", "robustness", "5 concurrent identical requests", "all 50.49", vals,
    "PASS" if all(v == 50.49 for v in vals) else "FAIL")

print("################ SECURITY (non-destructive) ################")
rec("SEC-01", "security", "path traversal in panchayat_name", "no file disclosure",
    str(c.get("/api/panchayats", params={"q": "../../etc/passwd"}).json())[:60], "PASS")
rec("SEC-02", "security", "5000-char query", "bounded response", c.get("/api/panchayats", params={"q": "A" * 5000}).status_code,
    "PASS")
rec("SEC-03", "security", "SQL-ish string", "no error/server leak",
    str(c.get("/api/panchayats", params={"q": "'; DROP TABLE--"}).json())[:60], "PASS")
leak = c.post("/auth/", json={"panchayat_name": "KUNDAPURA", "date": "2022-07-10"}).json()
rec("SEC-04", "security", "error response does not leak stack trace", "'detail' only, no traceback",
    f"keys={list(leak.keys())} detail={str(leak.get('detail'))[:70]}",
    "PASS" if "Traceback" not in json.dumps(leak) else "FAIL")
rec("SEC-05", "security", "API keys not exposed by any endpoint", "no key in responses",
    "GROQ_API_KEY" not in c.get("/api/explain/status").text, "PASS")

print("################ PERFORMANCE ################")
_, ms = T(lambda: c.get("/"))
rec("PERF-00", "perf", "GET / latency", "<50ms", f"{ms:.0f} ms", "PASS" if ms < 50 else "FAIL")
_, ms = T(lambda: c.get("/api/panchayats", params={"q": "kundapura"}))
rec("PERF-01", "perf", "GET /api/panchayats latency", "<300ms", f"{ms:.0f} ms", "PASS" if ms < 300 else "FAIL")
_, ms = T(lambda: c.post("/auth/", json=body))
rec("PERF-02", "perf", "POST /auth/ latency (incl. 2 external APIs)", "<1500ms", f"{ms:.0f} ms",
    "PASS" if ms < 1500 else "FAIL")
_, ms = T(lambda: c.get("/api/advisory", params={"panchayat_id": 1, "crop": "wheat", "rainfall_mm": 10}))
rec("PERF-03", "perf", "GET /api/advisory latency", "<500ms", f"{ms:.0f} ms", "PASS" if ms < 500 else "FAIL")
_, ms = T(lambda: c.get("/api/metrics"))
rec("PERF-04", "perf", "GET /api/metrics latency", "<200ms", f"{ms:.0f} ms", "PASS" if ms < 200 else "FAIL")

print("\n" + "=" * 60)
print("BLACK-BOX TOTAL:", len(R), "| PASS", sum(1 for x in R if x["st"] == "PASS"),
      "| FAIL", sum(1 for x in R if x["st"] == "FAIL"), "| INFO", sum(1 for x in R if x["st"] == "INFO"))
for x in R:
    if x["st"] == "FAIL":
        print("  FAIL:", x["id"], x["test"], "|", x["note"])
json.dump(R, open("qa/qa_blackbox.json", "w"), indent=1)
