from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
import logging
from .config import feature_config
from .exceptions import FeatureEngineeringError, DataValidationError

logger = logging.getLogger(__name__)

RAW_FEATURE_COLUMNS = [
    "monthly_spend", "spend_change", "monthly_spend_last_month",
    "logins_last_30_days", "logins_last_60_days", "logins_per_day", "login_trend",
    "support_tickets", "tickets_per_month", "tickets_per_tenure",
    "tenure_days", "tenure_bucket_<3m", "tenure_bucket_3-6m", 
    "tenure_bucket_6-12m", "tenure_bucket_1-2y", "tenure_bucket_2y+"
]

FEATURE_DTYPES = feature_config.FEATURE_DTYPES
TARGET_COLUMN = "churned"

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features for churn prediction"""
    df = df.copy()
    
    df['spend_login_interaction'] = df['spend_change'] * df['login_trend']
    df['spend_support_interaction'] = df['monthly_spend'] * df['tickets_per_month']
    df['tenure_engagement_interaction'] = df['tenure_days'] * df['logins_per_day']
    
    return df

def create_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features for churn prediction"""
    df = df.copy()
    
    df['login_frequency_trend'] = (df['logins_last_30_days'] - df['logins_last_60_days'] / 2) / 30
    
    return df

def create_customer_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Create customer segments based on behavior patterns"""
    df = df.copy()
    
    spend_threshold = df['monthly_spend'].quantile(feature_config.HIGH_VALUE_PERCENTILE)
    login_threshold = df['logins_per_day'].median()
    support_threshold = df['tickets_per_month'].quantile(feature_config.SUPPORT_HEAVY_PERCENTILE)
    
    df['is_high_value'] = (df['monthly_spend'] >= spend_threshold).astype(int)
    df['is_active_user'] = (df['logins_per_day'] >= login_threshold).astype(int)
    df['is_support_heavy'] = (df['tickets_per_month'] >= support_threshold).astype(int)
    
    df['customer_segment'] = 0
    
    high_risk_mask = (df['is_high_value'] == 1) & (df['login_trend'] < 0)
    df.loc[high_risk_mask, 'customer_segment'] = 2
    
    medium_risk_mask = (df['is_support_heavy'] == 1) | (df['is_active_user'] == 0)
    df.loc[medium_risk_mask, 'customer_segment'] = 1
    
    return df

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering"""
    raw_df = df[RAW_FEATURE_COLUMNS].copy()
    
    raw_df = create_interaction_features(raw_df)
    raw_df = create_time_based_features(raw_df)
    raw_df = create_customer_segments(raw_df)
    
    result_df = pd.concat([df[RAW_FEATURE_COLUMNS], raw_df.drop(columns=RAW_FEATURE_COLUMNS)], axis=1)
    
    return result_df

def get_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract features and target from DataFrame"""
    df_engineered = apply_feature_engineering(df)
    
    all_features = RAW_FEATURE_COLUMNS + [
        'spend_login_interaction', 'spend_support_interaction', 'tenure_engagement_interaction',
        'login_frequency_trend', 'is_high_value', 'is_active_user', 'is_support_heavy', 'customer_segment'
    ]
    
    available_features = [f for f in all_features if f in df_engineered.columns]
    
    if not available_features:
        raise ValueError("No features found in dataframe")
    
    missing_raw_features = set(RAW_FEATURE_COLUMNS) - set(df.columns)
    if missing_raw_features:
        raise DataValidationError(f"Missing required raw features: {missing_raw_features}")
    
    dtype_mapping = {}
    for feature in available_features:
        if feature in FEATURE_DTYPES:
            dtype_mapping[feature] = FEATURE_DTYPES[feature]
        elif feature in ['is_high_value', 'is_active_user', 'is_support_heavy', 'customer_segment']:
            dtype_mapping[feature] = np.int32
        else:
            dtype_mapping[feature] = np.float64
    
    X = df_engineered[available_features].astype(dtype_mapping)
    y = df[TARGET_COLUMN].astype(np.int32)
    
    if not set(y.unique()).issubset({0, 1}):
        raise DataValidationError(f"Target must be binary (0, 1), found: {set(y.unique())}")
    
    return X, y