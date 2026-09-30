"""
Panchayat advisory engine  (Layer 3 of the project)  -  the *explainable* part.

Design rule: the DECISION is made here, by explicit rules. The LLM never decides
anything; it only rewrites the decision for the farmer (see main.py), and its
output is rejected if it introduces numbers that are not in the trace.

Every rule - fired or not - is recorded with the inputs it read, the threshold it
compared against, how far the value was from that threshold, and what would have
to change for the outcome to flip. That trace IS the explanation.

NOTE: all thresholds below are illustrative defaults for the prototype. Validate
them with an agronomist / KVK / ICAR crop calendar before any real-world use.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

Crop = Literal["general", "wheat", "rice", "maize", "mustard", "cotton", "bajra", "pulses"]
Stage = Literal["sowing", "vegetative", "flowering", "grain_filling", "maturity", "harvest"]
Severity = Literal["none", "low", "medium", "high"]
_SEV_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}

# --------------------------------------------------------------------------
# Crop parameters (placeholders - see note above)
# --------------------------------------------------------------------------
CROP_PARAMS: dict[str, dict] = {
    #            heat tmax (C)  heat-sensitive stages          dry soil (m3/m3)  disease (example)      tall crop
    "wheat":   dict(heat=34, stages={"flowering", "grain_filling"}, dry_sm=0.15, disease="rust",             tall=True),
    "rice":    dict(heat=35, stages={"flowering"},                  dry_sm=0.30, disease="blast",            tall=False),
    "maize":   dict(heat=35, stages={"flowering", "grain_filling"}, dry_sm=0.15, disease="leaf blight",      tall=True),
    "mustard": dict(heat=32, stages={"flowering", "grain_filling"}, dry_sm=0.12, disease="alternaria blight", tall=False),
    "cotton":  dict(heat=38, stages={"flowering", "grain_filling"}, dry_sm=0.15, disease="boll rot",         tall=False),
    "bajra":   dict(heat=40, stages={"flowering"},                  dry_sm=0.10, disease="downy mildew",     tall=True),
    "pulses":  dict(heat=35, stages={"flowering", "grain_filling"}, dry_sm=0.12, disease="wilt or blight",   tall=False),
    # "general" = no specific crop: conservative middle-of-the-road values
    "general": dict(heat=36, stages={"flowering", "grain_filling"}, dry_sm=0.13, disease="fungal disease",   tall=False),
}

# Frontend growth-stage names -> engine stages (None = unknown / general)
FRONTEND_STAGE = {"general": None, "sowing": "sowing", "vegetative": "vegetative",
                  "flowering": "flowering", "ripening": "maturity", "harvest": "harvest"}

# Short action bullets shown as "Recommended Actions" in the UI, per rule
RULE_ACTIONS = {
    "R1_HEAVY_RAIN": ["Delay all field operations", "Clear drainage channels immediately",
                      "Hold back irrigation and fertiliser"],
    "R1B_SUBSTANTIAL_RAIN": ["Check that field drainage is working", "Avoid spraying or fertiliser for about 48 hours"],
    "R2_IRRIGATION": ["Check field moisture", "Plan irrigation within the next few days"],
    "R3_HEAT_STRESS": ["Irrigate lightly in the evening if water is available", "Avoid spraying in the afternoon"],
    "R4_DISEASE": ["Scout the field for early symptoms", "Consult your local agriculture office about preventive spray"],
    "R5_LODGING": ["Avoid irrigating just before strong wind", "Support or earth-up tall plants where possible"],
}
SEVERITY_TO_UI = {"none": "info", "low": "watch", "medium": "warning", "high": "alert"}

HEAVY_RAIN_MM = 64.5        # IMD "heavy rainfall" lower bound (24 h)
SUBSTANTIAL_RAIN_MM = 24.5  # prototype value, matches the frontend fallback tier
LOW_RAIN_MM_PER_DAY = 3.0   # below this * window_days => "low rain"
HUMID_PCT = 85.0
DISEASE_TMAX_RANGE = (22.0, 32.0)
WIND_KMH = 40.0


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class DownscalingInfo(BaseModel):
    """Optional: why this Panchayat differs from the Block value.
    `contributions` are per-feature effects in mm (e.g. SHAP values from the
    downscaling model, computed offline) and should sum to panchayat - block."""
    block_rainfall_mm: float
    contributions: dict[str, float] = {}


class PanchayatInput(BaseModel):
    panchayat: str
    block: Optional[str] = None
    crop: Crop
    stage: Optional[Stage] = None
    rainfall_mm: float = Field(..., ge=0, description="Forecast total for the advisory window")
    window_days: int = Field(1, ge=1, le=7)
    tmax_c: Optional[float] = None
    tmean_c: Optional[float] = Field(None, description="daily mean; used only if tmax_c is missing")
    humidity_pct: Optional[float] = Field(None, ge=0, le=100)
    wind_kmh: Optional[float] = Field(None, ge=0)
    soil_moisture: Optional[float] = Field(None, ge=0, le=1, description="SMAP volumetric, m3/m3")
    downscaling: Optional[DownscalingInfo] = None


    def has_temp(self) -> bool:
        return self.tmax_c is not None or self.tmean_c is not None

    def temp(self) -> float:
        """Temperature used by heat/disease rules: tmax if known, else the mean."""
        t = self.tmax_c if self.tmax_c is not None else self.tmean_c
        if t is None:
            raise ValueError("provide tmax_c or tmean_c")
        return t


class RuleResult(BaseModel):
    rule_id: str
    name: str
    fired: bool
    evaluable: bool = True                 # False when a required input was missing
    severity: Severity = "none"
    inputs: dict[str, Optional[float | str]] = {}
    condition: str                         # human-readable test that was applied
    margin_pct: Optional[float] = None     # signed distance from threshold, % of threshold
    flip_hint: str = ""                    # what would change the outcome
    advice_en: str = ""                    # deterministic advice text (only when fired)


class DownscalingExplanation(BaseModel):
    panchayat_rainfall_mm: float
    block_rainfall_mm: float
    delta_mm: float
    drivers: list[dict]
    unexplained_mm: float
    text_en: str


class AdvisoryTrace(BaseModel):
    panchayat: str
    crop: str
    severity: Severity
    action: str                            # short machine-readable label
    headline_en: str                       # deterministic farmer message
    rules: list[RuleResult]
    confidence: float
    confidence_reasons: list[str]
    downscaling: Optional[DownscalingExplanation] = None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _margin(value: float, threshold: float) -> float:
    return round(100.0 * (value - threshold) / threshold, 1) if threshold else 0.0


def _n(x: float) -> str:
    """Compact number formatting so text and trace use identical strings."""
    return f"{x:.1f}".rstrip("0").rstrip(".")


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------
def _r_heavy_rain(i: PanchayatInput, p: dict) -> RuleResult:
    fired = i.rainfall_mm >= HEAVY_RAIN_MM
    return RuleResult(
        rule_id="R1_HEAVY_RAIN", name="Heavy rainfall", fired=fired,
        severity="high" if fired else "none",
        inputs={"rainfall_mm": i.rainfall_mm},
        condition=f"rainfall_mm >= {_n(HEAVY_RAIN_MM)} (IMD heavy-rain threshold)",
        margin_pct=_margin(i.rainfall_mm, HEAVY_RAIN_MM),
        flip_hint=(f"would not fire if forecast rainfall stayed below {_n(HEAVY_RAIN_MM)} mm"
                   if fired else f"would fire at {_n(HEAVY_RAIN_MM)} mm or more"),
        advice_en=(f"Heavy rain of about {_n(i.rainfall_mm)} mm is expected. Clear drainage channels, "
                   f"hold back irrigation and fertiliser, and avoid harvesting or spraying."
                   if fired else ""),
    )


def _r_substantial_rain(i: PanchayatInput, p: dict) -> RuleResult:
    fired = SUBSTANTIAL_RAIN_MM <= i.rainfall_mm < HEAVY_RAIN_MM
    return RuleResult(
        rule_id="R1B_SUBSTANTIAL_RAIN", name="Substantial rainfall", fired=fired,
        severity="medium" if fired else "none",
        inputs={"rainfall_mm": i.rainfall_mm},
        condition=f"{_n(SUBSTANTIAL_RAIN_MM)} <= rainfall_mm < {_n(HEAVY_RAIN_MM)}",
        margin_pct=_margin(i.rainfall_mm, SUBSTANTIAL_RAIN_MM),
        flip_hint=(f"would escalate to heavy-rain alert at {_n(HEAVY_RAIN_MM)} mm" if fired
                   else f"fires between {_n(SUBSTANTIAL_RAIN_MM)} and {_n(HEAVY_RAIN_MM)} mm"),
        advice_en=(f"Substantial rain of about {_n(i.rainfall_mm)} mm is expected. Low-lying fields may "
                   f"waterlog, so check drainage and postpone spraying and fertiliser."
                   if fired else ""),
    )


def _r_irrigation(i: PanchayatInput, p: dict) -> RuleResult:
    low_mm = LOW_RAIN_MM_PER_DAY * i.window_days
    dry = p["dry_sm"]
    if i.soil_moisture is None:
        # Rain-only fallback: weaker evidence, lower severity
        fired = i.rainfall_mm < low_mm
        return RuleResult(
            rule_id="R2_IRRIGATION", name="Irrigation need", fired=fired, evaluable=True,
            severity="low" if fired else "none",
            inputs={"rainfall_mm": i.rainfall_mm, "soil_moisture": None},
            condition=f"rainfall_mm < {_n(low_mm)} (soil moisture unavailable, rain-only check)",
            margin_pct=_margin(i.rainfall_mm, low_mm),
            flip_hint="add soil-moisture data to confirm; rain of "
                      f"{_n(low_mm)} mm or more would clear it",
            advice_en=(f"Little rain ({_n(i.rainfall_mm)} mm) is expected. Check field moisture and "
                       f"irrigate {i.crop} if the soil is dry." if fired else ""),
        )
    fired = i.rainfall_mm < low_mm and i.soil_moisture < dry
    return RuleResult(
        rule_id="R2_IRRIGATION", name="Irrigation need", fired=fired,
        severity="medium" if fired else "none",
        inputs={"rainfall_mm": i.rainfall_mm, "soil_moisture": i.soil_moisture},
        condition=(f"rainfall_mm < {_n(low_mm)} AND soil_moisture < {_n(dry)} "
                   f"(dry-soil level for {i.crop})"),
        margin_pct=_margin(i.soil_moisture, dry),
        flip_hint=(f"would not fire if rain reached {_n(low_mm)} mm or soil moisture reached {_n(dry)}"
                   if fired else "fires only when BOTH rain is low and soil is dry"),
        advice_en=(f"Low rainfall ({_n(i.rainfall_mm)} mm) and dry soil ({_n(i.soil_moisture)}). "
                   f"Irrigation may be needed for {i.crop} within the next few days."
                   if fired else ""),
    )


def _r_heat(i: PanchayatInput, p: dict) -> RuleResult:
    thr = p["heat"]
    if not i.has_temp():
        return RuleResult(
            rule_id="R3_HEAT_STRESS", name="Heat stress", fired=False, evaluable=False,
            inputs={"temperature_c": None, "stage": i.stage or "unknown"},
            condition=f"temperature_c >= {_n(thr)} AND stage in {sorted(p['stages'])} for {i.crop}",
            flip_hint="temperature missing - rule could not be evaluated",
        )
    stage_unknown = i.stage is None
    sensitive = stage_unknown or i.stage in p["stages"]
    fired = i.temp() >= thr and sensitive
    return RuleResult(
        rule_id="R3_HEAT_STRESS", name="Heat stress", fired=fired,
        severity="high" if fired and not stage_unknown else ("medium" if fired else "none"),
        inputs={"temperature_c": i.temp(), "stage": i.stage or "unknown"},
        condition=f"temperature_c >= {_n(thr)} AND stage in {sorted(p['stages'])} for {i.crop}",
        margin_pct=_margin(i.temp(), thr),
        flip_hint=(f"would not fire if temperature stayed below {_n(thr)} C or the crop were outside "
                   f"its sensitive stage" if fired
                   else ("stage is outside the heat-sensitive window" if not sensitive
                         else f"would fire at {_n(thr)} C or above")),
        advice_en=(f"High temperature ({_n(i.temp())} C) at a heat-sensitive stage of {i.crop}. "
                   f"Irrigate lightly in the evening if water is available and avoid spraying in the afternoon."
                   if fired else ""),
    )


def _r_disease(i: PanchayatInput, p: dict) -> RuleResult:
    if i.humidity_pct is None or not i.has_temp():
        missing = " and ".join(n for n, ok in (("humidity", i.humidity_pct is not None),
                                               ("temperature", i.has_temp())) if not ok)
        return RuleResult(
            rule_id="R4_DISEASE", name="Fungal disease risk", fired=False, evaluable=False,
            inputs={"humidity_pct": i.humidity_pct, "temperature_c": i.temp() if i.has_temp() else None},
            condition=f"humidity_pct >= {_n(HUMID_PCT)} AND temperature_c in {DISEASE_TMAX_RANGE}",
            flip_hint=f"{missing} missing - rule could not be evaluated",
        )
    lo, hi = DISEASE_TMAX_RANGE
    fired = i.humidity_pct >= HUMID_PCT and lo <= i.temp() <= hi
    return RuleResult(
        rule_id="R4_DISEASE", name="Fungal disease risk", fired=fired,
        severity="medium" if fired else "none",
        inputs={"humidity_pct": i.humidity_pct, "temperature_c": i.temp()},
        condition=f"humidity_pct >= {_n(HUMID_PCT)} AND {_n(lo)} <= temperature_c <= {_n(hi)}",
        margin_pct=_margin(i.humidity_pct, HUMID_PCT),
        flip_hint=("would not fire if humidity dropped below "
                   f"{_n(HUMID_PCT)}% or temperature left the {_n(lo)}-{_n(hi)} C band" if fired
                   else "fires only when humidity is high AND temperature is in the favourable band"),
        advice_en=(f"Humid ({_n(i.humidity_pct)}%) and mild weather favours {p['disease']} in {i.crop}. "
                   f"Scout the field and consult your local agriculture office about preventive spray."
                   if fired else ""),
    )


def _r_wind(i: PanchayatInput, p: dict) -> RuleResult:
    if i.wind_kmh is None:
        return RuleResult(
            rule_id="R5_LODGING", name="Wind / lodging risk", fired=False, evaluable=False,
            inputs={"wind_kmh": None}, condition=f"wind_kmh >= {_n(WIND_KMH)} for tall crops",
            flip_hint="wind missing - rule could not be evaluated",
        )
    fired = i.wind_kmh >= WIND_KMH and p["tall"]
    return RuleResult(
        rule_id="R5_LODGING", name="Wind / lodging risk", fired=fired,
        severity="medium" if fired else "none",
        inputs={"wind_kmh": i.wind_kmh},
        condition=f"wind_kmh >= {_n(WIND_KMH)} AND crop is tall ({i.crop}: {p['tall']})",
        margin_pct=_margin(i.wind_kmh, WIND_KMH),
        flip_hint=("would not fire below " f"{_n(WIND_KMH)} km/h" if fired
                   else ("crop is not a tall crop" if not p["tall"] else f"would fire at {_n(WIND_KMH)} km/h or more")),
        advice_en=(f"Strong wind ({_n(i.wind_kmh)} km/h) can flatten tall {i.crop}. "
                   f"Avoid irrigating just before the wind and support or earth-up where possible."
                   if fired else ""),
    )


_RULES = (_r_heavy_rain, _r_substantial_rain, _r_irrigation, _r_heat, _r_disease, _r_wind)


# --------------------------------------------------------------------------
# Downscaling explanation (why this Panchayat differs from the Block)
# --------------------------------------------------------------------------
def explain_downscaling(i: PanchayatInput) -> Optional[DownscalingExplanation]:
    d = i.downscaling
    if d is None:
        return None
    delta = round(i.rainfall_mm - d.block_rainfall_mm, 2)
    drivers = sorted(
        ({"feature": k, "effect_mm": round(v, 2)} for k, v in d.contributions.items()),
        key=lambda x: abs(x["effect_mm"]), reverse=True,
    )
    unexplained = round(delta - sum(x["effect_mm"] for x in drivers), 2)
    direction = "higher" if delta > 0 else "lower" if delta < 0 else "the same as"
    text = (f"{i.panchayat} is forecast {_n(i.rainfall_mm)} mm against {_n(d.block_rainfall_mm)} mm "
            f"for the block ({'+' if delta >= 0 else ''}{_n(delta)} mm, {direction} than the block).")
    top = [x for x in drivers if abs(x["effect_mm"]) >= 0.5][:3]
    if top:
        text += " Main local factors: " + ", ".join(
            f"{x['feature']} ({'+' if x['effect_mm'] >= 0 else ''}{_n(x['effect_mm'])} mm)" for x in top) + "."
    return DownscalingExplanation(
        panchayat_rainfall_mm=i.rainfall_mm, block_rainfall_mm=d.block_rainfall_mm,
        delta_mm=delta, drivers=drivers, unexplained_mm=unexplained, text_en=text,
    )


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def evaluate(i: PanchayatInput) -> AdvisoryTrace:
    p = CROP_PARAMS[i.crop]
    rules = [r(i, p) for r in _RULES]
    fired = sorted((r for r in rules if r.fired), key=lambda r: -_SEV_ORDER[r.severity])

    if fired:
        top = fired[0]
        severity, action = top.severity, top.rule_id.split("_", 1)[1].lower()
        headline = " ".join(r.advice_en for r in fired[:2])   # top two, keep it short for a farmer
    else:
        severity, action = "none", "normal"
        headline = (f"No weather risk found for {i.crop} in {i.panchayat}. "
                    f"Continue normal irrigation and field work.")

    # Confidence is computed from data quality and proximity to thresholds - NOT asked of the LLM.
    conf, reasons = 1.0, []
    for name, val in (("soil moisture", i.soil_moisture), ("humidity", i.humidity_pct),
                      ("wind", i.wind_kmh), ("crop stage", i.stage)):
        if val is None:
            conf -= 0.1
            reasons.append(f"{name} not provided")
    if not i.has_temp():
        conf -= 0.2
        reasons.append("temperature not provided; heat-stress and disease rules were not evaluated")
    elif i.tmax_c is None:
        conf -= 0.1
        reasons.append("only mean temperature available; heat stress may be under-detected")
    near = [r.name for r in rules if r.evaluable and r.margin_pct is not None and abs(r.margin_pct) <= 10]
    if near:
        conf -= 0.1
        reasons.append("value close to threshold for: " + ", ".join(near))
    if i.downscaling and abs(explain_downscaling(i).unexplained_mm) > 0.25 * max(abs(i.downscaling.block_rainfall_mm), 1):
        conf -= 0.1
        reasons.append("downscaling attribution leaves a large unexplained residual")
    conf = round(max(conf, 0.3), 2)

    return AdvisoryTrace(
        panchayat=i.panchayat, crop=i.crop, severity=severity, action=action,
        headline_en=headline, rules=rules, confidence=conf, confidence_reasons=reasons,
        downscaling=explain_downscaling(i),
    )


# --------------------------------------------------------------------------
# Faithfulness guard for LLM rewrites
# --------------------------------------------------------------------------
_NUM = re.compile(r"\d+(?:\.\d+)?")


def numbers_in(text: str) -> set[float]:
    return {round(float(x), 2) for x in _NUM.findall(text)}


def allowed_numbers(trace: AdvisoryTrace) -> set[float]:
    """Every number that appears anywhere in the deterministic trace."""
    blob = trace.model_dump_json()
    return numbers_in(blob)


def is_faithful(message: str, trace: AdvisoryTrace) -> bool:
    """A rewrite is accepted only if it introduces no numbers absent from the trace."""
    return numbers_in(message) <= allowed_numbers(trace)


def actions_for(trace: AdvisoryTrace) -> list[str]:
    """Deduplicated action bullets for the fired rules, most severe first."""
    fired = sorted((r for r in trace.rules if r.fired), key=lambda r: -_SEV_ORDER[r.severity])
    out: list[str] = []
    for r in fired:
        for a in RULE_ACTIONS.get(r.rule_id, []):
            if a not in out:
                out.append(a)
    return out or ["Continue scheduled operations", "Maintain regular irrigation if dry conditions persist"]