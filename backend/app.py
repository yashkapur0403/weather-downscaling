"""
Single entry point:   uvicorn app:app --reload --port 8000

  main.py        XAI service (Groq explain/advisory wording + Sarvam translation), owns the FastAPI app + CORS
  routes_data.py data routes (/, /api/panchayats, /api/geocode, /auth/, /api/metrics, /api/weather)
"""
from main import app
from routes_data import register

register(app)
