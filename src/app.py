# src/app.py
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from contextlib import asynccontextmanager

src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from .predict import PricingPredictor
except ImportError:
    from predict import PricingPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ListingInput(BaseModel):
    accommodates: int
    bathrooms: float
    bedrooms: int
    beds: int
    minimum_nights: int
    neighbourhood_cleansed: str
    property_type: str
    room_type: str
    amenities: str

model_wrapper = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ML Model...")
    try:
        model_wrapper["predictor"] = PricingPredictor()
        logger.info("Model loaded.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    yield
    model_wrapper.clear()

app = FastAPI(title="Real Estate Pricing Engine", lifespan=lifespan)

@app.post("/predict")
def predict_price(listing: ListingInput):
    if "predictor" not in model_wrapper:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"predicted_price": model_wrapper["predictor"].predict(listing.dict()), "currency": "USD"}