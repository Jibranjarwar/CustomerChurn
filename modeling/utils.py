import logging
import os
from typing import List, Dict, Any, Optional
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import mlflow
import matplotlib.pyplot as plt
import yaml
import time
from datetime import datetime
from .config import model_config, validation_config
from .exceptions import (
    ConfigurationError, 
    DataValidationError, 
    ModelEvaluationError, 
    InsufficientDataError
)

logger = logging.getLogger(__name__)

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters"""
    logger.info("Validating config")
    start_time = time.time()
    
    try:
        for key in validation_config.REQUIRED_CONFIG_SECTIONS:
            if key not in config:
                logger.error(f"Missing required config section: {key}")
                raise ConfigurationError(f"Missing required config section: {key}")
            else:
                logger.debug(f"Config section '{key}' found")
        
        test_size = config['model']['test_size']
        if not 0 < test_size < 1:
            logger.error(f"Invalid test_size: {test_size}. Must be between 0 and 1")
            raise ConfigurationError("test_size must be between 0 and 1")
        else:
            logger.debug(f"Test size validation passed: {test_size}")
        
        validation_time = time.time() - start_time
        logger.info(f"Config validation completed: {validation_time:.3f}s")
        
    except Exception as e:
        logger.error(f"Config validation failed: {str(e)}")
        raise

def validate_data(df: pd.DataFrame, feature_columns: List[str], target_column: str) -> None:
    """Validate input data for missing values and required columns"""
    logger.info("Validating data")
    start_time = time.time()
    
    try:
        logger.debug(f"Checking for missing values in {len(feature_columns)} features and target")
        missing = df[feature_columns + [target_column]].isnull().sum()
        
        if missing.any():
            missing_cols = missing[missing > 0].to_dict()
            logger.error(f"Missing values found: {missing_cols}")
            raise DataValidationError(f"Missing values found:\n{missing[missing > 0]}")
        else:
            logger.debug("No missing values found in features or target")
        
        missing_features = set(feature_columns) - set(df.columns)
        if missing_features:
            logger.error(f"Missing feature columns: {missing_features}")
            raise DataValidationError("Some feature columns are missing from the DataFrame")
        else:
            logger.debug("All required feature columns present")
        
        logger.info(f"Data validation passed: {df.shape}")
        logger.info(f"Features validated: {len(feature_columns)}")
        logger.info(f"Target column: {target_column}")
        
        if target_column in df.columns:
            target_dist = df[target_column].value_counts().to_dict()
            logger.info(f"Target distribution: {target_dist}")
        
        validation_time = time.time() - start_time
        logger.info(f"Data validation completed: {validation_time:.3f}s")
        
    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        raise

def log_metrics_and_artifacts(model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                             feat_names: List[str], mlflow_run: Any, prefix: str = "") -> None:
    """Log model metrics and visualization artifacts to MLflow"""
    logger.info("Evaluating model and generating artifacts")
    start_time = time.time()
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        min_test_size = config["validation"]["min_test_size"]
        if len(y_test) < min_test_size:
            logger.error(f"Test set too small: {len(y_test)} < {min_test_size}")
            raise InsufficientDataError(f"Test set too small. Need at least {min_test_size} samples.")
        
        logger.info(f"Evaluating model on {len(y_test)} test samples")
        
        logger.debug("Generating predictions")
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        
        logger.debug("Calculating performance metrics")
        acc = (preds == y_test).mean()
        auc = roc_auc_score(y_test, proba)
        cm = confusion_matrix(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)
        
        metrics = {
            f"{prefix}accuracy": acc,
            f"{prefix}roc_auc": auc,
            f"{prefix}f1": report["1"]["f1-score"],
            f"{prefix}precision": report["1"]["precision"],
            f"{prefix}recall": report["1"]["recall"]
        }
        
        logger.info(f"Model performance metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        perfect_threshold = config["validation"]["perfect_score_threshold"]
        if acc == perfect_threshold or auc == perfect_threshold:
            logger.warning("Perfect scores seen, overfitting")
        
        mlflow.log_metrics(metrics)
        logger.info(f"Metrics logged to MLflow: {list(metrics.keys())}")
        
        logger.debug("Creating confusion matrix visualization")
        plt.figure(figsize=(4,3))
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.imshow(cm, cmap="Blues", interpolation="nearest")
        plt.colorbar()
        plt.xticks([0,1], ["No Churn", "Churn"])
        plt.yticks([0,1], ["No Churn", "Churn"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
        plt.tight_layout()
        
        artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        
        confusion_path = os.path.join(artifacts_dir, "confusion_matrix.png")
        plt.savefig(confusion_path)
        mlflow.log_artifact(confusion_path)
        plt.close()
        logger.info(f"Confusion matrix saved: {confusion_path}")
        
        if hasattr(model, 'feature_importances_'):
            logger.debug("Creating feature importance visualization")
            importances = model.feature_importances_
            
            feature_importance_df = pd.DataFrame({
                'feature': feat_names,
                'importance': importances
            }).sort_values('importance', ascending=True)
            
            plt.figure(figsize=(6,4))
            plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
            plt.title("Feature Importances")
            plt.tight_layout()
            
            feature_path = os.path.join(artifacts_dir, "feature_importances.png")
            plt.savefig(feature_path)
            mlflow.log_artifact(feature_path)
            plt.close()
            
            top_features = feature_importance_df.tail(5)
            logger.info("Top 5 most important features:")
            for _, row in top_features.iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            logger.info(f"Feature importance plot saved: {feature_path}")
        else:
            logger.warning("Model does not have feature_importances_ attribute")
        
        processing_time = time.time() - start_time
        logger.info(f"Model evaluation completed: {processing_time:.3f}s")
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        raise ModelEvaluationError(f"Failed to evaluate model: {str(e)}")