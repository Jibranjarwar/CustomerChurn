import os
import random
import numpy as np
import joblib
import pandas as pd
import logging
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import yaml
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_training.log')
    ]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELING_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(PROJECT_ROOT)

from mlflow.models.signature import infer_signature

from modeling.features import get_features_and_target, RAW_FEATURE_COLUMNS, TARGET_COLUMN, FEATURE_DTYPES
from modeling.utils import validate_data, log_metrics_and_artifacts
from modeling.config import model_config
from modeling.exceptions import ModelTrainingError, DataLeakageError, FeatureEngineeringError

config_path = os.path.join(MODELING_DIR, "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

random.seed(config["model"]["random_state"])
np.random.seed(config["model"]["random_state"])

try:
    data_path = os.path.join(PROJECT_ROOT, config["data"]["path"])
    df = pd.read_csv(data_path)
except FileNotFoundError:
    raise ModelTrainingError(f"Data file not found at {data_path}")
except pd.errors.EmptyDataError:
    raise ModelTrainingError("Data file is empty")
except pd.errors.ParserError as e:
    raise ModelTrainingError(f"Error parsing data file: {e}")
except Exception as e:
    raise ModelTrainingError(f"Unexpected error loading data: {e}")

missing_raw_features = set(RAW_FEATURE_COLUMNS) - set(df.columns)
if missing_raw_features:
    raise FeatureEngineeringError(f"Missing required raw features: {missing_raw_features}")

engineered_features_in_csv = [
    'spend_login_interaction', 'spend_support_interaction', 'tenure_engagement_interaction',
    'login_frequency_trend', 'is_high_value', 'is_active_user', 'is_support_heavy', 'customer_segment'
]

leaked_features = [f for f in engineered_features_in_csv if f in df.columns]
if leaked_features:
    logger.warning(f"Found pre-engineered features in CSV (will be re-created): {leaked_features}")

X, y = get_features_and_target(df)

if X.isnull().any().any():
    raise FeatureEngineeringError(f"Missing values found in features:\n{X.isnull().sum()[X.isnull().sum() > 0]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=config["model"]["test_size"], 
    random_state=config["model"]["random_state"], 
    stratify=y
)

logger.info(f"Training set size: {len(X_train)}")
logger.info(f"Validation set size: {len(X_test)}")
logger.info(f"Training target distribution: {pd.Series(y_train).value_counts(normalize=True).to_dict()}")
logger.info(f"Validation target distribution: {pd.Series(y_test).value_counts(normalize=True).to_dict()}")

mlflow.set_experiment("churn-prediction")

with mlflow.start_run() as run:
    param_grid = config["model"]["grid_search"]
    if 'max_depth' in param_grid:
        param_grid['max_depth'] = [None if x == 'null' else x for x in param_grid['max_depth']]
    
    base_model = RandomForestClassifier(
        random_state=config["model"]["random_state"],
        class_weight="balanced"
    )
    
    grid_search = GridSearchCV(
        base_model, 
        param_grid, 
        cv=10,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    try:
        grid_search.fit(X_train, y_train)
    except ValueError as e:
        raise ModelTrainingError(f"Invalid model parameters: {e}")
    except MemoryError as e:
        raise ModelTrainingError(f"Insufficient memory for model training: {e}")
    except Exception as e:
        raise ModelTrainingError(f"Model training failed: {e}")

    model = grid_search.best_estimator_
    
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    
    cv_scores = cross_val_score(
        model, 
        X_train, 
        y_train, 
        cv=config["model"]["cv_folds"], 
        scoring='roc_auc'
    )
    
    mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
    mlflow.log_metric("cv_roc_auc_std", cv_scores.std())

    signature = infer_signature(X_train, model.predict(X_train))
    
    mlflow.sklearn.log_model(
        model, 
        "churn_model",
        signature=signature,
        input_example=X_train.iloc[:5].astype({col: FEATURE_DTYPES.get(col, np.float64) for col in X_train.columns})
    )

    log_metrics_and_artifacts(model, X_test, y_test, X.columns, run)

    os.makedirs(config["model"]["dir"], exist_ok=True)
    model_path = os.path.join(config["model"]["dir"], "model.pkl")
    joblib.dump(model, model_path)