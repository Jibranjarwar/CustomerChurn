import os
import random
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import yaml
import sys

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELING_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the project root to Python path
sys.path.append(PROJECT_ROOT)
# Add cross-validation
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score
from mlflow.models.signature import infer_signature


from modeling.features import get_features_and_target, FEATURE_COLUMNS, TARGET_COLUMN, FEATURE_DTYPES
from modeling.utils import validate_data, log_metrics_and_artifacts

# Load config from modeling directory
config_path = os.path.join(MODELING_DIR, "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

# Access nested config values
random.seed(config["model"]["random_state"])
np.random.seed(config["model"]["random_state"])

try:
    # Load already processed data
    df = pd.read_csv(config["data"]["path"])
except FileNotFoundError:
    raise RuntimeError(f"Data file not found at {config['data']['path']}")

# Validate and split data
validate_data(df, FEATURE_COLUMNS, TARGET_COLUMN)
X, y = get_features_and_target(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config["model"]["test_size"], 
    random_state=config["model"]["random_state"], 
    stratify=y
)

mlflow.set_experiment("churn-prediction")

with mlflow.start_run() as run:
    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    
    base_model = RandomForestClassifier(
        random_state=config["model"]["random_state"],
        class_weight="balanced"
    )
    
    # Use GridSearchCV with stratified k-fold
    grid_search = GridSearchCV(
        base_model, 
        param_grid, 
        cv=10,  # Increased from 5 to 10 folds
        scoring='roc_auc',
        n_jobs=-1
    )
    
    try:
        grid_search.fit(X_train, y_train)
    except Exception as e:
        raise RuntimeError(f"Model training failed: {e}")

    # Use best model from grid search
    model = grid_search.best_estimator_
    
    # Log best parameters
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    
    # Get cross validation scores for best model
    cv_scores = cross_val_score(
        model, 
        X_train, 
        y_train, 
        cv=config["model"]["cv_folds"], 
        scoring='roc_auc'
    )
    
    mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
    mlflow.log_metric("cv_roc_auc_std", cv_scores.std())

    # Infer model signature
    signature = infer_signature(X_train, model.predict(X_train))
    
    # Log the model with signature
    mlflow.sklearn.log_model(
        model, 
        "churn_model",
        signature=signature,
        input_example=X_train.iloc[:5].astype(FEATURE_DTYPES)
    )

    # Log metrics and artifacts
    log_metrics_and_artifacts(model, X_test, y_test, X.columns, run)

    os.makedirs(config["model"]["dir"], exist_ok=True)
    joblib.dump(model, os.path.join(config["model"]["dir"], "model.pkl"))