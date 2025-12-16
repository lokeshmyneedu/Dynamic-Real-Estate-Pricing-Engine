# src/config.py
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class Config:
    # Paths
    RAW_DATA_PATH = Path("data/raw/listings.csv")
    MODEL_SAVE_PATH = Path("models/pricing_model_v1.pkl")
    
    # Feature Configs
    NUMERICAL_FEATURES = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'minimum_nights']
    CATEGORICAL_FEATURES = ['neighbourhood_cleansed', 'property_type', 'room_type']
    target = 'price'
    
    RANDOM_STATE = 42

    # --- THE MODEL ZOO ---
    MODEL_CONFIGS = {
        'Polynomial_Ridge': {
            'model': Ridge(),
            'feature_options': {'use_poly': True},
            'params': {
                'model__alpha': [0.1, 1.0, 10.0],
                # --- FIX 2: Use only sparse-compatible solvers ---
                'model__solver': ['auto', 'sparse_cg', 'lsqr'] 
            }
        },
        'Standard_Ridge': {
            'model': Ridge(),
            'feature_options': {'use_poly': False},
            'params': {
                'model__alpha': [0.1, 1.0, 10.0],
                'model__solver': ['auto'] # 'auto' usually picks the best one safely
            }
        },
        'Lasso': {
            'model': Lasso(),
            'feature_options': {'use_poly': False},
            'params': {
                'model__alpha': [0.01, 0.1, 1.0],
                # --- FIX 3: Increase iterations to prevent ConvergenceWarning ---
                'model__max_iter': [10000] 
            }
        },
        'SVR': {
            'model': SVR(),
            'feature_options': {'use_poly': False},
            'params': {
                'model__C': [0.1, 1.0],
                'model__kernel': ['rbf'],
                'model__epsilon': [0.1, 0.2]
            }
        },
        'XGBoost': {
            'model': XGBRegressor(random_state=42),
            'feature_options': {'use_poly': False},
            'params': {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.05, 0.1], 
                'model__max_depth': [3, 6]
            }
        },
        'Random_Forest': {
            'model': RandomForestRegressor(random_state=42),
            'feature_options': {'use_poly': False},
            'params': {
                'model__n_estimators': [100],
                'model__max_depth': [10, 20],
                'model__min_samples_split': [2, 5]
            }
        }
    }