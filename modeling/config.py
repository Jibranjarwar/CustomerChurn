"""
Configuration management for churn prediction system.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class FeatureConfig:
    """Feature engineering parameters."""
    
    HIGH_VALUE_PERCENTILE: float = 0.8
    SUPPORT_HEAVY_PERCENTILE: float = 0.75
    DAYS_IN_MONTH: int = 30
    ORIGINAL_FEATURE_COUNT: int = 16
    MIN_TENURE_DAYS: int = 1
    
    FEATURE_DTYPES: Dict[str, np.dtype] = None
    
    def __post_init__(self):
        if self.FEATURE_DTYPES is None:
            self.FEATURE_DTYPES = {
                # Raw features only - no engineered features to prevent data leakage
                "monthly_spend": np.float64,
                "spend_change": np.float64,
                "monthly_spend_last_month": np.float64,
                
                "logins_last_30_days": np.float64,
                "logins_last_60_days": np.float64,
                "logins_per_day": np.float64,
                "login_trend": np.float64,
                
                "support_tickets": np.float64,
                "tickets_per_month": np.float64,
                "tickets_per_tenure": np.float64,
                
                "tenure_days": np.float64,
                "tenure_bucket_<3m": np.int32,
                "tenure_bucket_3-6m": np.int32,
                "tenure_bucket_6-12m": np.int32,
                "tenure_bucket_1-2y": np.int32,
                "tenure_bucket_2y+": np.int32
            }

@dataclass
class ModelConfig:
    """Configuration for model training parameters."""
    
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.3
    CV_FOLDS: int = 10
    N_ESTIMATORS: int = 100
    CLASS_WEIGHT: str = "balanced"
    
    OPTUNA_N_TRIALS: int = 30
    GRID_SEARCH_CV: int = 10
    
    MAX_CORRELATION_THRESHOLD: float = 0.5
    
    PERFECT_SCORE_THRESHOLD: float = 1.0
    MIN_TEST_SIZE: int = 2

@dataclass
class ValidationConfig:
    """Data validation parameters."""
    
    MIN_TEST_SIZE: int = 2
    MAX_MISSING_VALUES: int = 0
    
    REQUIRED_CONFIG_SECTIONS: list = None
    
    def __post_init__(self):
        if self.REQUIRED_CONFIG_SECTIONS is None:
            self.REQUIRED_CONFIG_SECTIONS = ['data', 'model']

feature_config = FeatureConfig()
model_config = ModelConfig()
validation_config = ValidationConfig() 