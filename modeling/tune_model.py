import os
import sys
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score
from modeling.features import get_features_and_target, RAW_FEATURE_COLUMNS, TARGET_COLUMN
from modeling.utils import validate_data, log_metrics_and_artifacts
from modeling.config import model_config
from modeling.exceptions import ModelTrainingError, FeatureEngineeringError
import numpy as np
import random
import yaml
import logging
import time
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_training.log')
    ]
)
logger = logging.getLogger(__name__)

MODELING_DIR = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(MODELING_DIR, "config.yaml")

logger.info("Loading config")
with open(config_path) as f:
    config = yaml.safe_load(f)

logger.info(f"Setting random seed: {config['model']['random_state']}")
random.seed(config["model"]["random_state"])
np.random.seed(config["model"]["random_state"])

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and prepare training data"""
    start_time = time.time()
    logger.info("Loading data")
    
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(PROJECT_ROOT, config["data"]["path"])
    
    logger.info(f"Data path: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded: {df.shape}")
        
        logger.info(f"Target distribution: {df[TARGET_COLUMN].value_counts().to_dict()}")
        
        X, y = get_features_and_target(df)
        logger.info(f"Features extracted: {X.shape}")
        
        if X.isnull().any().any():
            missing_cols = X.columns[X.isnull().any()].tolist()
            logger.error(f"Missing values in features: {missing_cols}")
            raise FeatureEngineeringError(f"Missing values found in features:\n{X.isnull().sum()[X.isnull().sum() > 0]}")
        
        logger.info("Data validation passed")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config["model"]["test_size"],
            random_state=config["model"]["random_state"],
            stratify=y
        )
        
        logger.info(f"Data split: train={X_train.shape}, test={X_test.shape}")
        
        loading_time = time.time() - start_time
        logger.info(f"Data loading completed: {loading_time:.2f}s")
        
        return X_train, X_test, y_train, y_test
        
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise ModelTrainingError(f"Failed to load data: {str(e)}")

def objective(trial: optuna.Trial) -> float:
    """Optuna objective function"""
    trial_start_time = time.time()
    trial_number = trial.number
    
    logger.info(f"Trial {trial_number}")
    
    optuna_config = config["model"]["optuna"]
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 
            optuna_config["n_estimators_range"][0],
            optuna_config["n_estimators_range"][1]),
        "max_depth": trial.suggest_int("max_depth",
            optuna_config["max_depth_range"][0],
            optuna_config["max_depth_range"][1]),
        "min_samples_split": trial.suggest_int("min_samples_split",
            optuna_config["min_samples_split_range"][0],
            optuna_config["min_samples_split_range"][1]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf",
            optuna_config["min_samples_leaf_range"][0],
            optuna_config["min_samples_leaf_range"][1]),
        "random_state": config["model"]["random_state"],
        "class_weight": "balanced"
    }
    
    logger.info(f"Trial {trial_number} parameters: {params}")
    
    try:
        model = RandomForestClassifier(**params)
        
        X_train, X_test, y_train, y_test = load_data()
        
        cv_scores = cross_val_score(
            model, 
            X_train, 
            y_train, 
            cv=config["model"]["cv_folds"],
            scoring='roc_auc'
        )
        
        mean_cv_score = cv_scores.mean()
        std_cv_score = cv_scores.std()
        
        logger.info(f"Trial {trial_number} CV score: {mean_cv_score:.4f} (+/- {std_cv_score * 2:.4f})")
        
        trial_time = time.time() - trial_start_time
        logger.info(f"Trial {trial_number} completed: {trial_time:.2f}s")
        
        return mean_cv_score
        
    except Exception as e:
        logger.error(f"Trial {trial_number} failed: {str(e)}")
        return 0.0

def main():
    """Main function for Optuna"""
    start_time = time.time()
    logger.info("Optuna optimization")
    
    try:
        X_train, X_test, y_train, y_test = load_data()
        logger.info(f"Data loaded successfully: train={X_train.shape}, test={X_test.shape}")
        
        study = optuna.create_study(
            direction="maximize",
            study_name="churn_prediction_optimization"
        )
        
        n_trials = config["model"]["optuna"]["n_trials"]
        logger.info(f"Running {n_trials} trials")
        
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best CV score: {study.best_trial.value:.4f}")
        logger.info(f"Best parameters: {study.best_trial.params}")
        
        best_params = study.best_trial.params.copy()
        best_params.update({
            "random_state": config["model"]["random_state"],
            "class_weight": "balanced"
        })
        
        final_model = RandomForestClassifier(**best_params)
        final_model.fit(X_train, y_train)
        
        test_score = final_model.score(X_test, y_test)
        test_auc = roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1])
        
        logger.info(f"Final model test accuracy: {test_score:.4f}")
        logger.info(f"Final model test AUC: {test_auc:.4f}")
        
        os.makedirs(config["model"]["dir"], exist_ok=True)
        model_path = os.path.join(config["model"]["dir"], "best_model.pkl")
        joblib.dump(final_model, model_path)
        logger.info(f"Best model saved to: {model_path}")
        
        mlflow.set_experiment("churn-prediction-optuna")
        with mlflow.start_run() as run:
            mlflow.log_params(best_params)
            mlflow.log_metric("best_cv_score", study.best_trial.value)
            mlflow.log_metric("test_accuracy", test_score)
            mlflow.log_metric("test_auc", test_auc)
            
            mlflow.sklearn.log_model(final_model, "best_churn_model")
            
            log_metrics_and_artifacts(final_model, X_test, y_test, X_train.columns, run)
        
        total_time = time.time() - start_time
        logger.info(f"Optimization completed: {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        raise ModelTrainingError(f"Optimization failed: {str(e)}")

if __name__ == "__main__":
    main()