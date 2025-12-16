# src/features.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Logarithmically transforms skewed numerical features.
    Enterprise Standard: Handles potential div by zero or negative values.
    Works with both DataFrames (column names) and numpy arrays (column indices).
    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Handle numpy arrays (from sklearn transformers like SimpleImputer)
        if isinstance(X, np.ndarray):
            X_copy = X.copy()
            # Apply log1p to all columns in numpy array
            X_copy = np.log1p(np.clip(X_copy, 0, None))
            return X_copy
        
        # Handle DataFrames
        X_copy = X.copy()
        if self.columns:
            for col in self.columns:
                if col in X_copy.columns:
                    X_copy[col] = np.log1p(X_copy[col].clip(lower=0))
        return X_copy

class AmenityScoreEngine(BaseEstimator, TransformerMixin):
    """
    Example of Custom Feature Engineering:
    Parses a string of amenities (e.g., "{TV,Wifi,Pool}") into a numerical score.
    """
    def __init__(self, amenity_col='amenities'):
        self.amenity_col = amenity_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Check if column exists to prevent pipeline crash
        if self.amenity_col not in X.columns:
            logger.warning(f"{self.amenity_col} not found in data. Returning original.")
            return X
            
        X_copy = X.copy()
        # Simple logic: count number of items in the list string
        X_copy['amenity_score'] = X_copy[self.amenity_col].apply(lambda x: len(str(x).split(',')))
        return X_copy.drop(columns=[self.amenity_col])