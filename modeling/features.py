from typing import Tuple, List
import pandas as pd
import numpy as np



# Define feature data types
FEATURE_DTYPES = {
    "monthly_spend": np.float64,
    "tenure_days": np.float64,
    "tickets_per_month": np.float64,
    "support_tickets": np.float64,
    "logins_last_30_days": np.float64
}

FEATURE_COLUMNS = list(FEATURE_DTYPES.keys())
TARGET_COLUMN = "churned"

def get_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target variables with proper dtypes.
    
    Args:
        df: Input DataFrame containing all features and target
        
    Returns:
        Tuple containing:
            - Features DataFrame with correct dtypes
            - Target Series
    """
    X = df[FEATURE_COLUMNS].astype(FEATURE_DTYPES)
    y = df[TARGET_COLUMN].astype(np.int32)
    return X, y