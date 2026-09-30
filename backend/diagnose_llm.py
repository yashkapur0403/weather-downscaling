"""Run from backend/:  python diagnose_llm.py
Tests every model in GROQ_MODELS the same way /api/explain does, and shows the exact error."""
import asyncio, json, os
from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()
key = os.getenv("GROQ_API_KEY", "")
models = [m.strip() for m in os.getenv("GROQ_MODELS", "openai/gpt-oss-120b,qwen/qwen3.6-27b,openai/gpt-oss-20b").split(",") if m.strip()]
print("GROQ_API_KEY loaded:", bool(key), "| models:", models)

async def main():
    c = AsyncGroq(api_key=key, timeout=30, max_retries=0)
    for m in models:
        try:
            r = await c.chat.completions.create(
                model=m, temperature=0.2, max_tokens=2000, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": 'Return ONLY JSON: {"explanation": "...", "answer": null}'},
                          {"role": "user", "content": json.dumps({"facts": {"rain_mm": 4.9}, "question": "why"})}])
            print(f"OK   {m}: {(r.choices[0].message.content or '')[:120]!r}")
        except Exception as e:
            print(f"FAIL {m}: {type(e).__name__} status={getattr(e, 'status_code', None)} {str(e)[:200]}")

asyncio.run(main())