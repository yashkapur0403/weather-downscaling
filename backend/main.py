"""
Explainable-AI backend (text only)
- Groq   -> structured, explainable reasoning, with an ordered multi-model fallback chain
- Sarvam -> regional-language translation (in and out)

Run:  uvicorn main:app --reload --port 8000
"""
import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal, Optional

from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from groq import AsyncGroq
from pydantic import BaseModel, Field

import advisory as adv
import aux_layers

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("xai-backend")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Ordered fallback chain: first = preferred, rest = used only if earlier ones fail.
GROQ_MODELS = [
    m.strip() for m in os.getenv(
        "GROQ_MODELS",
        "openai/gpt-oss-120b,openai/gpt-oss-20b",
    ).split(",") if m.strip()
]
GROQ_TIMEOUT = float(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))
COOLDOWN_SECONDS = int(os.getenv("MODEL_COOLDOWN_SECONDS", "60"))   # skip a rate-limited model for this long

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_BASE = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai")
TRANSLATE_MODEL = os.getenv("SARVAM_TRANSLATE_MODEL", "mayura:v1")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

if not GROQ_API_KEY or not SARVAM_API_KEY:
    log.warning("GROQ_API_KEY or SARVAM_API_KEY is missing - set them in .env")

LangCode = Literal[
    "en-IN", "hi-IN", "bn-IN", "ta-IN", "te-IN", "kn-IN",
    "ml-IN", "mr-IN", "gu-IN", "od-IN", "pa-IN",
]
TRANSLATE_CHUNK = 900   # mayura:v1 accepts ~1000 chars per request

# max_retries=0 so a failing model falls through to the next one immediately
groq_client = AsyncGroq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT, max_retries=0)
http = httpx.AsyncClient(timeout=60)
_cooldown_until: dict[str, float] = {}

@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    await http.aclose()


app = FastAPI(title="Explainable AI + Regional Language API", version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Sarvam: translation
# --------------------------------------------------------------------------
def _split(text: str, limit: int) -> list[str]:
    """Split on sentence boundaries into chunks <= limit chars."""
    sentences = re.split(r"(?<=[.!?।])\s+", text.strip())
    chunks, cur = [], ""
    for s in sentences:
        while len(s) > limit:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(s[:limit])
            s = s[limit:]
        if len(cur) + len(s) + 1 > limit and cur:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks


async def _translate_chunk(chunk: str, src: str, tgt: str) -> str:
    last_err = ""
    for attempt in range(2):                       # one retry on transient failure
        try:
            r = await http.post(
                f"{SARVAM_BASE}/translate",
                headers={"api-subscription-key": SARVAM_API_KEY},
                json={
                    "input": chunk,
                    "source_language_code": src,    # "auto" lets Sarvam detect
                    "target_language_code": tgt,
                    "model": TRANSLATE_MODEL,
                    "mode": "formal",
                    "enable_preprocessing": True,
                },
            )
            if r.status_code < 400:
                return r.json()["translated_text"]
            last_err = f"{r.status_code}: {r.text[:200]}"
            if r.status_code < 500 and r.status_code != 429:
                break                               # client error, retrying won't help
        except httpx.HTTPError as e:
            last_err = str(e)
        await asyncio.sleep(0.5)
    log.error("Sarvam translate failed: %s", last_err)
    raise HTTPException(502, f"Translation service error: {last_err}")


async def sarvam_translate(text: str, src: str, tgt: str) -> str:
    if not text.strip() or src == tgt:
        return text
    parts = [await _translate_chunk(c, src, tgt) for c in _split(text, TRANSLATE_CHUNK)]
    return " ".join(parts)


# --------------------------------------------------------------------------
# Groq: explainable reasoning with fallback chain
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an explainable-AI assistant. Answer the user's question and \
make your reasoning fully transparent.

Return ONLY a JSON object with exactly this schema:
{
  "answer": "direct answer, 2-4 sentences",
  "reasoning_steps": ["step 1", "step 2", "..."],
  "key_factors": [
    {"factor": "short name", "impact": "positive|negative|neutral", "weight": 0.0-1.0,
     "explanation": "one sentence"}
  ],
  "confidence": 0.0-1.0,
  "limitations": ["what you are unsure about or could not verify"]
}

Rules:
- Be honest about uncertainty; lower confidence when information is missing.
- Reasoning steps must reflect how you actually reached the answer, not a post-hoc story.
- Keep each string concise and in plain language.
- Do not give medical, legal or financial decisions as final; flag when a professional is needed.
"""


class KeyFactor(BaseModel):
    factor: str
    impact: Literal["positive", "negative", "neutral"] = "neutral"
    weight: float = Field(0.5, ge=0, le=1)
    explanation: str = ""


class Explanation(BaseModel):
    answer: str
    reasoning_steps: list[str] = []
    key_factors: list[KeyFactor] = []
    confidence: float = Field(0.5, ge=0, le=1)
    limitations: list[str] = []


def _extract_json(raw: str) -> dict:
    """Tolerate <think> blocks, code fences and stray text around the JSON."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```(?:json)?", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(raw[start:end + 1])


async def _call_model(model: str, user_msg: str, json_mode: bool = True) -> Explanation:
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = await groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=4000,        # reasoning models spend part of this budget on hidden thinking
        **kwargs,
    )
    return Explanation(**_extract_json(resp.choices[0].message.content or ""))


async def run_with_fallback(call) -> tuple[object, str, list[dict]]:
    """Try each Groq model in order. `call(model)` returns a result or raises.
    Returns (result, model_used, attempts)."""
    attempts: list[dict] = []
    now = time.monotonic()
    # Models still cooling down go to the back so they're only used as a last resort
    ready = [m for m in GROQ_MODELS if _cooldown_until.get(m, 0) <= now]
    cooling = [m for m in GROQ_MODELS if m not in ready]

    for model in ready + cooling:
        t0 = time.monotonic()
        try:
            result = await call(model)
            attempts.append({"model": model, "status": "ok", "ms": int((time.monotonic() - t0) * 1000)})
            return result, model, attempts
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status == 429:
                _cooldown_until[model] = time.monotonic() + COOLDOWN_SECONDS
            reason = f"{type(e).__name__}" + (f" ({status})" if status else "")
            log.warning("Groq model %s failed: %s - %s", model, reason, str(e)[:200])
            attempts.append({"model": model, "status": "failed", "error": reason,
                             "ms": int((time.monotonic() - t0) * 1000)})
    raise HTTPException(503, detail={"message": "All Groq models failed", "attempts": attempts})


async def groq_explain(question: str, context: Optional[str]) -> tuple[Explanation, str, list[dict]]:
    user_msg = question if not context else f"Context:\n{context}\n\nQuestion:\n{question}"

    async def call(model: str) -> Explanation:
        try:
            return await _call_model(model, user_msg)
        except Exception as e:
            # Some models reject JSON mode with a 400; retry once without it
            if getattr(e, "status_code", None) == 400:
                return await _call_model(model, user_msg, json_mode=False)
            raise

    return await run_with_fallback(call)


async def translate_explanation(exp: Explanation, tgt: str) -> Explanation:
    """Translate every human-readable string in the explanation, concurrently."""
    sem = asyncio.Semaphore(5)

    async def tr(t: str) -> str:
        async with sem:
            return await sarvam_translate(t, "en-IN", tgt)

    answer, steps, limits, factors, expls = await asyncio.gather(
        tr(exp.answer),
        asyncio.gather(*[tr(s) for s in exp.reasoning_steps]),
        asyncio.gather(*[tr(s) for s in exp.limitations]),
        asyncio.gather(*[tr(f.factor) for f in exp.key_factors]),
        asyncio.gather(*[tr(f.explanation) for f in exp.key_factors]),
    )
    new_factors = [
        f.model_copy(update={"factor": n, "explanation": e})
        for f, n, e in zip(exp.key_factors, factors, expls)
    ]
    return exp.model_copy(update={
        "answer": answer,
        "reasoning_steps": list(steps),
        "limitations": list(limits),
        "key_factors": new_factors,
    })


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    context: Optional[str] = Field(None, max_length=6000)
    input_language: str = "auto"                  # e.g. "hi-IN", "ta-IN" or "auto"
    output_language: LangCode = "en-IN"


class ExplainResponse(BaseModel):
    input_language: str
    output_language: str
    english_question: str
    explanation: Explanation                     # in output_language
    explanation_en: Explanation                  # original English, for auditing
    model_used: str
    attempts: list[dict] = []                    # fallback trail, useful for debugging/UI


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    source: str = "auto"
    target: LangCode


@app.get("/health")
async def health():
    return {"status": "ok", "groq_models": GROQ_MODELS}


@app.post("/api/explain/generic", response_model=ExplainResponse)
async def explain_generic(req: ExplainRequest):
    # 1) regional language -> English
    if req.input_language in ("en-IN", "en"):
        english_q = req.text
    else:
        english_q = await sarvam_translate(req.text, req.input_language, "en-IN")

    # 2) explainable reasoning on Groq (with fallback)
    exp_en, model_used, attempts = await groq_explain(english_q, req.context)

    # 3) English -> user's language
    exp_local = exp_en if req.output_language == "en-IN" \
        else await translate_explanation(exp_en, req.output_language)

    return ExplainResponse(
        input_language=req.input_language,
        output_language=req.output_language,
        english_question=english_q,
        explanation=exp_local,
        explanation_en=exp_en,
        model_used=model_used,
        attempts=attempts,
    )


@app.post("/api/translate")
async def translate(req: TranslateRequest):
    return {"translated_text": await sarvam_translate(req.text, req.source, req.target)}


# --------------------------------------------------------------------------
# Panchayat advisory: rules decide -> Groq rephrases -> Sarvam translates
# --------------------------------------------------------------------------
PHRASE_PROMPT = """You rewrite an agricultural advisory for a farmer in simple, friendly English.

You are given a JSON "trace" that already contains the decision. Do NOT change the decision.
Rules:
- Use ONLY facts and numbers that appear in the trace. Never add new numbers, crops, chemicals or dates.
- Maximum 3 short sentences. Put the most urgent action first.
- If the trace says severity "none", just say conditions look normal.
Return ONLY JSON: {"message": "..."}"""


async def _phrase_call(model: str, trace: adv.AdvisoryTrace) -> str:
    payload = {"panchayat": trace.panchayat, "crop": trace.crop, "severity": trace.severity,
               "advice": trace.headline_en,
               "why": [f"{r.name}: {r.condition}" for r in trace.rules if r.fired]}
    resp = await groq_client.chat.completions.create(
        model=model, temperature=0.2, max_tokens=1500,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": PHRASE_PROMPT},
                  {"role": "user", "content": json.dumps(payload)}],
    )
    msg = str(_extract_json(resp.choices[0].message.content or "")["message"]).strip()
    if not msg:
        raise ValueError("empty message")
    return msg


async def farmer_message(trace: adv.AdvisoryTrace) -> tuple[str, str, list[dict]]:
    """Returns (message_en, source, attempts). Falls back to the deterministic text if the
    LLM is unavailable or its rewrite adds numbers that are not in the trace."""
    if not GROQ_API_KEY:
        return trace.headline_en, "template", []
    try:
        msg, model, attempts = await run_with_fallback(lambda m: _phrase_call(m, trace))
    except HTTPException as e:
        return trace.headline_en, "template", e.detail.get("attempts", [])
    if not adv.is_faithful(msg, trace):
        log.warning("LLM rewrite rejected (unsupported numbers): %s", msg)
        attempts.append({"model": model, "status": "rejected", "error": "unfaithful_numbers"})
        return trace.headline_en, "template", attempts
    return msg, f"llm:{model}", attempts


class AdvisoryResponse(BaseModel):
    output_language: str
    message: str                                   # farmer-facing, in output_language
    message_en: str
    message_source: str                            # "template" or "llm:<model>"
    severity: str
    action: str
    confidence: float
    confidence_reasons: list[str]
    fired_rules: list[dict]                        # [{rule_id, name, condition, inputs, flip_hint, reason}]
    downscaling_explanation: Optional[str] = None  # in output_language
    trace: adv.AdvisoryTrace                       # full audit trail (English)
    attempts: list[dict] = []


class AdvisoryRequest(BaseModel):
    data: adv.PanchayatInput
    output_language: LangCode = "hi-IN"


class BlockAdvisoryRequest(BaseModel):
    block: str
    panchayats: list[adv.PanchayatInput] = Field(..., min_length=1, max_length=60)
    output_language: LangCode = "hi-IN"


async def build_advisory(data: adv.PanchayatInput, lang: str) -> AdvisoryResponse:
    trace = adv.evaluate(data)
    msg_en, source, attempts = await farmer_message(trace)
    fired = [r for r in trace.rules if r.fired]
    ds_en = trace.downscaling.text_en if trace.downscaling else None

    async def tr(t: str) -> str:
        try:
            return await sarvam_translate(t, "en-IN", lang)
        except HTTPException:
            return t                                # never lose the advice because translation failed

    msg, ds, reasons = await asyncio.gather(
        tr(msg_en),
        tr(ds_en) if ds_en else asyncio.sleep(0, result=None),
        asyncio.gather(*[tr(r.advice_en) for r in fired]),
    )
    return AdvisoryResponse(
        output_language=lang, message=msg, message_en=msg_en, message_source=source,
        severity=trace.severity, action=trace.action, confidence=trace.confidence,
        confidence_reasons=trace.confidence_reasons,
        fired_rules=[{"rule_id": r.rule_id, "name": r.name, "condition": r.condition,
                      "inputs": r.inputs, "margin_pct": r.margin_pct, "flip_hint": r.flip_hint,
                      "reason": rs} for r, rs in zip(fired, reasons)],
        downscaling_explanation=ds, trace=trace, attempts=attempts,
    )


@app.post("/api/advisory", response_model=AdvisoryResponse)
async def advisory_one(req: AdvisoryRequest):
    return await build_advisory(req.data, req.output_language)


@app.post("/api/advisory/block")
async def advisory_block(req: BlockAdvisoryRequest):
    sem = asyncio.Semaphore(3)

    async def one(p: adv.PanchayatInput):
        async with sem:
            return await build_advisory(p, req.output_language)

    results = await asyncio.gather(*[one(p) for p in req.panchayats])
    return {"block": req.block, "output_language": req.output_language, "panchayats": results}


# ==========================================================================
# Frontend-compatible routes (match Frontend/src/types/index.ts exactly)
# ==========================================================================
def _risk_level(mm: float) -> str:            # same bands as classifyRisk() in the frontend
    return ("no_rain" if mm < 2.5 else "light" if mm < 10 else "moderate" if mm < 25
            else "heavy" if mm < 50 else "very_heavy")


_AUX = None


def aux_layers_ctx():
    """Lazy singleton for the committed soil/NDVI/land-cover arrays (None if absent)."""
    global _AUX
    if _AUX is None:
        root = os.getenv("REPO_ROOT") or str(Path(__file__).resolve().parents[1])
        try:
            _AUX = aux_layers.AuxLayers(root)
        except Exception as e:                       # never break the advisory because aux data is missing
            log.warning("aux layers unavailable: %s", e)
            _AUX = False
    return _AUX or None


CropQ = Literal["general", "rice", "wheat", "cotton", "maize", "pulses", "mustard", "bajra"]
StageQ = Literal["general", "sowing", "vegetative", "flowering", "ripening", "harvest"]
UI_DISCLAIMER = ("Rule-based prototype advisory. Thresholds are illustrative defaults and have not been "
                 "validated for field use; consult your local agriculture office.")


class AdvisoryUIResponse(BaseModel):
    advisory_text: str
    severity: Literal["info", "watch", "warning", "alert"]
    actions: list[str]
    evidence: dict
    data_date: str
    disclaimer: str
    crop: str
    stage: str
    # extras the current UI ignores but a "why?" drawer can show
    confidence: float
    confidence_reasons: list[str]
    message_source: str
    fired_rules: list[dict]
    trace: adv.AdvisoryTrace
    # what was CHECKED before the rules ran (see advisory_ui):
    #   rainfall "verified_against_layer1" | "resolved_from_layer1"
    #            | "unverified_no_data_source"
    #   temperature/humidity "caller_supplied" | "absent"
    verification: dict = {}


# Values are rounded to 2 dp at the API boundary, so a faithful caller can differ
# from the stored value by at most half a rounding step.
RAINFALL_MATCH_TOL_MM = 0.011


def _rainfall_resolver():
    """The data layer publishes this on app.state (routes_data.register).

    Absent when the XAI service is mounted without the data routes (text-only
    use), in which case nothing can be verified and the caller's value is
    accepted but explicitly labelled unverified.
    """
    return getattr(app.state, "rainfall_resolver", None)


@app.get("/api/advisory", response_model=AdvisoryUIResponse)
async def advisory_ui(
    panchayat_id: int,
    crop: CropQ = "general",
    stage: StageQ = "general",
    rainfall_mm: Optional[float] = Query(None, ge=0),
    temperature_c: Optional[float] = None,
    humidity_pct: Optional[float] = Query(None, ge=0, le=100),
    date: str = "",
    panchayat_name: str = "this panchayat",
    irrigation_available: Optional[bool] = None,
    lang: str = "en-IN",
):
    if lang != "en-IN" and lang not in LangCode.__args__:
        raise HTTPException(422, f"unsupported lang {lang}")

    # ---- the rainfall must be the Layer-1 value for THIS panchayat and date ----
    # The advisory is only meaningful if it is answering the question the user
    # asked. A supplied value that disagrees with the stored field is a hard
    # error (409) rather than something to quietly accept: otherwise a stale
    # client value, a wrong Panchayat or a wrong date produces confident advice
    # about a different place or day with nothing in the trace to show for it.
    supplied_mm = rainfall_mm
    verification = {
        "panchayat_id": panchayat_id,
        "date": date,
        "rainfall": "unverified_no_data_source",
        "supplied_mm": supplied_mm,
        "expected_mm": None,
        "source": None,
        "cell": None,
        "location_precision": None,
        "temperature_c": "caller_supplied" if temperature_c is not None else "absent",
        "humidity_pct": "caller_supplied" if humidity_pct is not None else "absent",
        "aux": "unavailable",
    }
    resolver = _rainfall_resolver()
    if resolver is not None:
        try:
            ref = resolver(panchayat_id, date)
        except HTTPException:
            raise
        except Exception as e:                     # artefacts unreadable
            raise HTTPException(
                503, f"cannot verify rainfall for panchayat_id={panchayat_id}, date={date}: {e}")
        expected = ref.get("rainfall_mm")
        verification.update(expected_mm=expected, source=ref.get("source"),
                            cell=ref.get("cell"),
                            location_precision=ref.get("location_precision"),
                            reason=ref.get("reason"))
        if expected is None:
            raise HTTPException(422, {
                "error": (f"cannot verify rainfall for panchayat_id={panchayat_id}, "
                          f"date={date!r}: {ref.get('reason')}"),
                "reason": ref.get("reason"),
                "panchayat_id": panchayat_id, "date": date,
                "hint": ("the advisory is rule-based on the Layer-1 field, so it refuses to run "
                         "against an unverifiable rainfall; check the Panchayat id and that the "
                         "date is inside the served range (GET /api/metrics -> layer1_provenance)"),
            })
        if supplied_mm is None:
            rainfall_mm = expected
            verification["rainfall"] = "resolved_from_layer1"
        elif abs(supplied_mm - expected) > RAINFALL_MATCH_TOL_MM:
            raise HTTPException(409, {
                "error": ("supplied rainfall_mm does not match the Layer-1 value for this "
                          "panchayat and date"),
                "supplied_mm": supplied_mm, "expected_mm": expected,
                "difference_mm": round(supplied_mm - expected, 3),
                "panchayat_id": panchayat_id, "date": date,
                "source": ref.get("source"),
                "hint": ("re-read the value from POST /auth/ for this Panchayat and date, or "
                         "omit rainfall_mm and let the server resolve it"),
            })
        else:
            rainfall_mm = expected               # normalise onto the stored value
            verification["rainfall"] = "verified_against_layer1"
    elif supplied_mm is None:
        raise HTTPException(422, "rainfall_mm is required when no data source is registered")

    aux_ctx = None
    _aux = aux_layers_ctx()
    if _aux is not None:
        d = _aux.lookup_panchayat(panchayat_id, date or None)
        if d:
            aux_ctx = adv.AuxContext.model_validate(d)
    verification["aux"] = "verified_against_layer2_cell" if aux_ctx else "unavailable"
    data = adv.PanchayatInput(
        panchayat=panchayat_name, crop=crop, stage=adv.FRONTEND_STAGE[stage],
        rainfall_mm=rainfall_mm, tmean_c=temperature_c, humidity_pct=humidity_pct,
        aux=aux_ctx,
    )
    trace = adv.evaluate(data)
    text_en, source, _ = await farmer_message(trace)
    actions = adv.actions_for(trace)
    if irrigation_available is False and any(r.fired and r.rule_id == "R2_IRRIGATION" for r in trace.rules):
        actions = ["Irrigation is not available: ask your agriculture office about water options"
                   if a == "Plan irrigation within the next few days" else a for a in actions]

    async def tr(t: str) -> str:
        if lang == "en-IN":
            return t
        try:
            return await sarvam_translate(t, "en-IN", lang)
        except HTTPException:
            return t

    text, actions_t = await asyncio.gather(tr(text_en), asyncio.gather(*[tr(a) for a in actions]))
    fired = [r for r in trace.rules if r.fired]
    return AdvisoryUIResponse(
        advisory_text=text, severity=adv.SEVERITY_TO_UI[trace.severity], actions=list(actions_t),
        evidence={"rainfall_mm": rainfall_mm, "risk_level": _risk_level(rainfall_mm),
                  "temperature_c": temperature_c, "humidity_pct": humidity_pct,
                  "aux": aux_ctx.model_dump() if aux_ctx else None},
        data_date=date, disclaimer=UI_DISCLAIMER, crop=crop, stage=stage,
        confidence=trace.confidence, confidence_reasons=trace.confidence_reasons,
        message_source=source,
        fired_rules=[{"rule_id": r.rule_id, "name": r.name, "condition": r.condition, "inputs": r.inputs,
                      "margin_pct": r.margin_pct, "flip_hint": r.flip_hint} for r in fired],
        trace=trace,
        verification=verification,
    )


# ---- Explainable AI for the downscaled rainfall number ---------------------
class XPrediction(BaseModel):
    rainfall_mm: float
    risk_level: str
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    elevation_m: Optional[float] = None


class XMapping(BaseModel):
    method: Literal["direct_grid", "area_weighted", "nearest_fallback"]
    n_cells: int
    fallback_distance_m: Optional[float] = None


class XModel(BaseModel):
    name: str
    channels: list[str] = []
    test_mae_mm: Optional[float] = None
    baseline_mae_mm: Optional[float] = None
    mae_improvement_pct: Optional[float] = None
    reference_product: str = "CHIRPS v2.0"


class UIExplainRequest(BaseModel):
    panchayat_name: str
    block_name: str = ""
    district: str = ""
    state: str = ""
    date: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    prediction: XPrediction
    mapping: XMapping
    model: XModel
    question: Optional[str] = Field(None, max_length=300)
    language: Literal["en", "hi"] = "en"


class UIFactor(BaseModel):
    factor: str
    effect: Literal["increases", "decreases", "neutral"]
    weight: float = Field(..., ge=0, le=1)
    detail: str


class UIExplainResponse(BaseModel):
    summary: str
    explanation: str
    factors: list[UIFactor]
    confidence: Literal["low", "medium", "high"]
    confidence_note: str
    answer: Optional[str] = None
    provider: str
    model: str
    generated_at: str
    fallback_reason: Optional[str] = None   # why the rule-based text was used instead of the LLM


_METHOD_NOTE = {
    "direct_grid": "Grid cell centres fall directly inside the panchayat polygon.",
    "area_weighted": "Weighted average of overlapping 0.05° cells (small polygon).",
    "nearest_fallback": "Nearest grid cell centroid used (< 15 km fallback).",
}


def ui_facts(r: UIExplainRequest) -> dict:
    """Deterministic, auditable part of the explanation. Only channels the model was
    actually trained on can appear as factors. Weights are heuristic (not SHAP)."""
    pr, ch = r.prediction, set(r.model.channels or ["imd_rain", "dem", "era5_t2m", "era5_t2m_max", "era5_dewp"])
    f: list[UIFactor] = []
    if "imd_rain" in ch:
        f.append(UIFactor(factor="IMD coarse rainfall (0.25°)", effect="increases" if pr.rainfall_mm >= 2.5 else "neutral",
                          weight=0.9, detail="The bilinear-upsampled IMD value is the starting point; the U-Net only "
                                             "learns a correction on top of it."))
    if "dem" in ch and pr.elevation_m is not None:
        hi = pr.elevation_m > 500
        f.append(UIFactor(factor="Elevation (SRTM DEM)", effect="increases" if hi else "neutral", weight=0.6 if hi else 0.2,
                          detail=(f"At {pr.elevation_m:g} m, orographic lift on windward slopes tends to add rainfall." if hi
                                  else f"Terrain is relatively flat ({pr.elevation_m:g} m), so elevation adds little.")))
    if ("era5_dewp" in ch) and pr.humidity_pct is not None:
        h = pr.humidity_pct
        f.append(UIFactor(factor="Moisture (ERA5 dewpoint)", effect="increases" if h > 70 else "decreases" if h < 50 else "neutral",
                          weight=0.5 if h > 70 else 0.3,
                          detail=f"Relative humidity of {h:g}% is used as a proxy for available moisture."))
    if ({"era5_t2m", "era5_t2m_max"} & ch) and pr.temperature_c is not None:
        f.append(UIFactor(factor="Temperature (ERA5 T2m)", effect="neutral", weight=0.2,
                          detail=f"Mean temperature of {pr.temperature_c:g}°C gives the model thermal context for convection."))
    conf = "low" if r.mapping.method == "nearest_fallback" else "high" if r.mapping.n_cells > 1 else "medium"
    n = r.mapping.n_cells
    text = (f"The model starts from IMD's 28 km rainfall, upsamples it to a 5 km grid, and then a U-Net adds a local "
            f"correction using terrain and ERA5 weather. The panchayat value is then taken from {n} grid cell"
            f"{'' if n == 1 else 's'}.")
    if r.model.mae_improvement_pct is not None:
        text += (f" On the 2022 test season this model's error was {r.model.mae_improvement_pct:g}% lower than the "
                 f"IMD baseline (scored against {r.model.reference_product}).")
    return {"summary": f"{pr.rainfall_mm:.1f} mm/day estimated for {r.panchayat_name} on {r.date}.",
            "explanation": text, "factors": f, "confidence": conf,
            "confidence_note": _METHOD_NOTE[r.mapping.method]}


EXPLAIN_PROMPT = """You explain a downscaled rainfall estimate to a non-expert.
You get JSON "facts" that are already correct. Rewrite the explanation clearly in plain English.
Rules:
- Use ONLY facts and numbers present in the facts (or in the user's question). Never invent numbers, causes or sources.
- The factor weights are heuristic, not measured attributions; do not present them as exact.
- If a question is given, answer it using only the facts; if the facts cannot answer it, say so plainly.
- Explanation: at most 4 sentences. Answer: at most 3 sentences, or null if no question.
Return ONLY JSON: {"explanation": "...", "answer": "..." or null}"""


async def _explain_call(model: str, facts_json: str, question: Optional[str]) -> dict:
    resp = await groq_client.chat.completions.create(
        model=model, temperature=0.2, max_tokens=2000, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": EXPLAIN_PROMPT},
                  {"role": "user", "content": json.dumps({"facts": json.loads(facts_json), "question": question})}],
    )
    out = _extract_json(resp.choices[0].message.content or "")
    if not str(out.get("explanation", "")).strip():
        raise ValueError("empty explanation")
    return out


def _unsupported_numbers(text: str, allowed: set) -> set:
    """Numbers in `text` that are not in `allowed`. A number that is just a rounding of an allowed value
    (4.87 -> 4.9 or 5, 15.9 -> 16) is fine; anything else is treated as invented."""
    ok = set(allowed) | {round(a, 1) for a in allowed} | {float(round(a)) for a in allowed}
    return {n for n in adv.numbers_in(text) if n not in ok}


@app.get("/api/explain/status")
async def explain_status():
    return {"enabled": bool(GROQ_API_KEY), "provider": "groq", "model": GROQ_MODELS[0] if GROQ_MODELS else ""}


@app.post("/api/explain", response_model=UIExplainResponse)
async def explain_ui(req: UIExplainRequest):
    facts = ui_facts(req)
    explanation, answer, provider, model_used = facts["explanation"], None, "rules", "deterministic"
    fallback_reason: Optional[str] = None if GROQ_API_KEY else "GROQ_API_KEY is not set on the backend"

    if GROQ_API_KEY:
        facts_json = json.dumps({
            "place": f"{req.panchayat_name}, {req.block_name}, {req.district}, {req.state}", "date": req.date,
            "summary": facts["summary"], "baseline_explanation": facts["explanation"],
            "factors": [f.model_dump() for f in facts["factors"]],
            "confidence": facts["confidence"], "confidence_note": facts["confidence_note"],
            "risk_level": req.prediction.risk_level, "model": req.model.model_dump()}, ensure_ascii=False)
        try:
            out, model_used, _ = await run_with_fallback(lambda m: _explain_call(m, facts_json, req.question))
            allowed = adv.numbers_in(facts_json) | adv.numbers_in(req.question or "")
            e_txt, a_txt = str(out["explanation"]).strip(), (str(out.get("answer") or "").strip() or None)
            bad = _unsupported_numbers(e_txt, allowed) | (_unsupported_numbers(a_txt, allowed) if a_txt else set())
            if not bad:
                explanation, answer, provider = e_txt, a_txt, "groq"
            else:
                log.warning("Explain rewrite rejected - numbers not in the facts: %s | text: %.300s",
                            sorted(bad), f"{e_txt} || {a_txt}")
                fallback_reason = f"LLM reply rejected: it contained numbers not in the facts {sorted(bad)}"
                model_used = "deterministic"
        except HTTPException as e:
            d = e.detail if isinstance(e.detail, dict) else {}
            errs = "; ".join(f"{a.get('model')}: {a.get('error')}" for a in d.get("attempts", []) if a.get("status") != "ok")
            fallback_reason = f"All Groq models failed - {errs or e.detail}"
            model_used = "deterministic"
        except Exception as e:                       # e.g. bad JSON shape - never turn this into a 500
            log.exception("Explain LLM step crashed")
            fallback_reason = f"LLM step crashed: {type(e).__name__}: {str(e)[:150]}"
            model_used = "deterministic"

    if req.question and answer is None and provider == "groq":
        answer = "The available facts do not answer that question."
    elif req.question and answer is None:
        answer = ("The language model is unavailable or its answer could not be verified against the inputs, so "
                  "follow-up questions cannot be answered right now. The factors shown are rule-based.")

    resp = UIExplainResponse(
        summary=facts["summary"], explanation=explanation, factors=facts["factors"],
        confidence=facts["confidence"], confidence_note=facts["confidence_note"], answer=answer,
        provider=provider, model=model_used, generated_at=datetime.now(timezone.utc).isoformat(),
        fallback_reason=None if provider == "groq" else fallback_reason)

    if req.language == "hi":
        async def tr(t: str) -> str:
            try:
                return await sarvam_translate(t, "en-IN", "hi-IN")
            except HTTPException:
                return t
        s_, e_, cn_, a_, fs_, ds_ = await asyncio.gather(
            tr(resp.summary), tr(resp.explanation), tr(resp.confidence_note),
            tr(resp.answer) if resp.answer else asyncio.sleep(0, result=None),
            asyncio.gather(*[tr(f.factor) for f in resp.factors]),
            asyncio.gather(*[tr(f.detail) for f in resp.factors]))
        resp = resp.model_copy(update={
            "summary": s_, "explanation": e_, "confidence_note": cn_, "answer": a_,
            "factors": [f.model_copy(update={"factor": a, "detail": b}) for f, a, b in zip(resp.factors, fs_, ds_)]})
    return resp