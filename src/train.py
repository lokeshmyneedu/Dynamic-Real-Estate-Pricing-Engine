# src/train.py
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from .config import Config
    from .pipeline import PipelineFactory
except ImportError:
    from config import Config
    from pipeline import PipelineFactory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def run_training():
    # 1. Load Data
    logger.info(f"Loading data from {Config.RAW_DATA_PATH}...")
    try:
        df = pd.read_csv(Config.RAW_DATA_PATH)
    except FileNotFoundError:
        logger.error("Data file not found! Please check data/raw/ folder.")
        return

    # --- FIX 1: Use raw string r'' for regex to silence SyntaxWarning ---
    if df[Config.target].dtype == object:
        df[Config.target] = df[Config.target].replace(r'[\$,]', '', regex=True).astype(float)
    
    df = df.dropna(subset=[Config.target])
    
    # 2. Split Data
    X = df.drop(columns=[Config.target])
    y = df[Config.target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=Config.RANDOM_STATE
    )

    # 3. Model Zoo Loop
    best_overall_model = None
    best_overall_score = float('inf') 
    best_model_name = ""

    for name, model_conf in Config.MODEL_CONFIGS.items():
        logger.info(f"\n--- Training {name} ---")
        
        use_poly = model_conf.get('feature_options', {}).get('use_poly', False)
        
        pipeline = PipelineFactory.create_pipeline(
            model=model_conf['model'], 
            use_poly=use_poly
        )
        
        grid = GridSearchCV(
            pipeline,
            model_conf['params'],
            cv=3,
            scoring='neg_mean_absolute_error',
            n_jobs=-2, 
            verbose=1
        )
        
        grid.fit(X_train, y_train)
        
        mae = -grid.best_score_
        logger.info(f"Best {name} MAE: {mae:.2f}")
        
        if mae < best_overall_score:
            best_overall_score = mae
            best_overall_model = grid.best_estimator_
            best_model_name = name

    # 4. Final Champion
    print("\n" + "="*30)
    print(f"CHAMPION MODEL: {best_model_name}")
    print("="*30)
    
    predictions = best_overall_model.predict(X_test)
    final_mae = mean_absolute_error(y_test, predictions)
    final_r2 = r2_score(y_test, predictions)
    
    print(f"Test Set MAE: {final_mae:.2f}")
    print(f"Test Set R2: {final_r2:.2f}")
    
    logger.info(f"Saving champion to {Config.MODEL_SAVE_PATH}...")
    joblib.dump(best_overall_model, Config.MODEL_SAVE_PATH)

if __name__ == "__main__":
    run_training()