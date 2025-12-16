# src/pipeline.py
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression

try:
    from .config import Config
    from .features import LogTransformer, AmenityScoreEngine
except ImportError:
    from config import Config
    from features import LogTransformer, AmenityScoreEngine

logger = logging.getLogger(__name__)

class PipelineFactory:
    
    @staticmethod
    def create_pipeline(model, use_poly=False):
        """
        Args:
            model: The sklearn estimator to add at the end.
            use_poly (bool): If True, inserts PolynomialFeatures step.
        """
        # 1. Numerical Preprocessing
        steps_num = [
            ('imputer', SimpleImputer(strategy='median')),
            ('log_transform', LogTransformer(columns=['minimum_nights'])), 
            ('scaler', StandardScaler())
        ]
        
        # Add Polynomial Features if requested by Config
        if use_poly:
            steps_num.append(('poly', PolynomialFeatures(degree=2, include_bias=False)))

        num_transformer = Pipeline(steps=steps_num)

        # 2. Categorical Preprocessing
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # 3. Column Transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, Config.NUMERICAL_FEATURES),
                ('cat', cat_transformer, Config.CATEGORICAL_FEATURES)
            ])

        # 4. Final Assembly
        full_pipeline = Pipeline(steps=[
            ('amenity_eng', AmenityScoreEngine()),
            ('preprocessor', preprocessor),
            ('feature_selection', SelectKBest(score_func=f_regression, k='all')),
            ('model', model)
        ])
        
        return full_pipeline