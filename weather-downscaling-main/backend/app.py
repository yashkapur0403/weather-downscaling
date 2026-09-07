import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers.query_router import query_router
from backend.utils.model_loader import ModelLoader
from backend.utils.predictor import Predictor
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-api")

MODEL_CHECKPOINT_PATH = os.environ.get("MODEL_CHECKPOINT_PATH", "models/best_model.pt")
LAYER2_CSV_PATH = os.environ.get("LAYER2_CSV_PATH", "outputs/layer2/panchayat_weather.csv")
LAYER2_PICKLE_PATH = os.environ.get("LAYER2_PICKLE_PATH")

model_loader = ModelLoader.get_instance()
predictor = Predictor(
    model_loader,
    layer2_csv=LAYER2_CSV_PATH,
    layer2_pickle=LAYER2_PICKLE_PATH,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting background model load from %s", MODEL_CHECKPOINT_PATH)
    model_loader.load_async(MODEL_CHECKPOINT_PATH)
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Weather APi", version="1.0.0", lifespan=lifespan)

frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3001")
origins = list({
    frontend_url,
    "http://localhost:3001",
    "http://localhost:3000",
    "http://localhost:80",
    "http://localhost",
})

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router, prefix="/auth", tags=["Auth"])


@app.get("/", tags=["Health"])
def health_check():
    return {"message": "Weather API running"}


@app.get("/api/metrics", tags=["Metrics"])
def get_metrics():
    """Return model performance metrics for the frontend."""
    return {
        "all_variants": [
            {
                "split": "test",
                "model": "Bilinear IMD baseline",
                "MAE_mm": 7.67,
                "RMSE_mm": 12.45,
                "corr": 0.304
            },
            {
                "split": "test", 
                "model": "U-Net + DEM + ERA5-Land",
                "MAE_mm": 6.45,
                "RMSE_mm": 11.32,
                "corr": 0.376
            }
        ],
        "test_variants": [
            {
                "split": "test",
                "model": "Bilinear IMD baseline", 
                "MAE_mm": 7.67,
                "RMSE_mm": 12.45,
                "corr": 0.304
            },
            {
                "split": "test",
                "model": "U-Net + DEM + ERA5-Land",
                "MAE_mm": 6.45, 
                "RMSE_mm": 11.32,
                "corr": 0.376
            }
        ],
        "selected_model": {
            "name": "U-Net + DEM + ERA5-Land",
            "variant_key": "model_d",
            "test_mae_mm": 6.45,
            "test_rmse_mm": 11.32,
            "test_correlation": 0.376
        },
        "baseline": {
            "name": "Bilinear IMD baseline",
            "test_mae_mm": 7.67
        },
        "mae_improvement_pct": 15.9,
        "evaluation_period": "2022 monsoon (Jun-Sep, 122 days)",
        "reference_product": "CHIRPS v2.0",
        "reference_note": "Reference product, not absolute ground truth",
        "channels": ["IMD rain", "DEM", "ERA5-Land T/Tmax/dewpoint"],
        "n_parameters": "~150k"
    }


@app.get("/api/panchayats", tags=["Search"])
def search_panchayats(q: str = "", limit: int = 20):
    """Search panchayats by name, block, or district."""
    import pandas as pd
    from pathlib import Path
    
    csv_path = Path("outputs/layer2/panchayat_weather.csv")
    if not csv_path.exists():
        return []
    
    try:
        df = pd.read_csv(csv_path)
    except:
        return []
    
    if not q:
        return df.head(limit).to_dict(orient="records")
    
    # Search across multiple columns
    mask = (
        df["panchayat_name"].astype(str).str.contains(q, case=False, na=False) |
        df["block_name"].astype(str).str.contains(q, case=False, na=False) |
        df["district"].astype(str).str.contains(q, case=False, na=False) |
        df["state"].astype(str).str.contains(q, case=False, na=False)
    )
    
    results = df[mask].head(limit)
    
    # Convert to format expected by frontend
    return [
        {
            "panchayat_id": int(row.get("panchayat_id", 0)),
            "panchayat_name": str(row.get("panchayat_name", "")),
            "block_name": str(row.get("block_name", "")),
            "block_id": int(row.get("block_id", 0)),
            "district": str(row.get("district", "")),
            "state": str(row.get("state", "")),
            "date": "2022-07-10",  # Default date since CSV doesn't have it
            "rainfall_mm": float(row.get("rainfall_mm", 0)),
            "n_cells": 1,
            "mapping_method": "direct_grid",
            "fallback_distance_m": None,
            "lat": None,
            "lon": None,
        }
        for _, row in results.iterrows()
    ]


@app.get("/api/advisory", tags=["Advisory"])
def get_advisory(
    panchayat_id: int = 0,
    crop: str = "general",
    stage: str = "general",
    irrigation_available: bool = False
):
    """Generate crop advisory based on weather conditions."""
    # This is a simplified advisory response - in production this would use
    # actual weather data and crop models
    return {
        "advisory_text": f"Based on current conditions for {crop} in {stage} stage, maintain normal irrigation schedules and monitor for pest activity.",
        "severity": "info",
        "actions": [
            "Continue regular irrigation",
            "Monitor for pest and disease activity",
            "Apply nitrogen fertilizer if needed",
            "Monitor weather forecasts for heavy rain"
        ],
        "evidence": {
            "rainfall_mm": 2.2,
            "risk_level": "light",
            "temperature_c": 28.5,
            "humidity_pct": 75
        },
        "data_date": "2022-07-10",
        "disclaimer": "This advisory is based on historical data and should be used with local knowledge.",
        "crop": crop,
        "stage": stage
    }