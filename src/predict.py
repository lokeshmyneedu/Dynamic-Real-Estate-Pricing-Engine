# src/predict.py
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from .config import Config
except ImportError:
    from config import Config

class PricingPredictor:
    def __init__(self, model_path=None):
        self.model_path = model_path or Config.MODEL_SAVE_PATH
        self.model = self._load_model()
        
    def _load_model(self):
        try:
            return joblib.load(self.model_path)
        except FileNotFoundError:
            logger.error(f"Model file not found at {self.model_path}")
            raise
    
    def predict(self, input_data: dict) -> float:
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Handle User Aliases (Mapping API input to Training Columns)
        column_mapping = {
            'min_nights': 'minimum_nights',
            'neighborhood': 'neighbourhood_cleansed',
            'cleaning_fee': 'beds' # Preserved your logic, though usually these are separate
        }
        df = df.rename(columns=column_mapping)
        
        # Ensure all columns exist (fill with defaults if missing)
        # The Pipeline's SimpleImputer will handle the NaNs, we just need the columns to exist
        required_cols = Config.NUMERICAL_FEATURES + Config.CATEGORICAL_FEATURES + ['amenities']
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan 
        
        # Predict
        prediction = self.model.predict(df)
        return round(float(prediction[0]), 2)

if __name__ == "__main__":
    predictor = PricingPredictor()
    test_input = {
        "accommodates": 4, "bathrooms": 2.0, "bedrooms": 2, "beds": 2,
        "minimum_nights": 3, "neighbourhood_cleansed": "Downtown",
        "property_type": "Apartment", "room_type": "Entire home/apt",
        "amenities": "{TV,Wifi}"
    }
    print(f"Predicted price: ${predictor.predict(test_input)}")