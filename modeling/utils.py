import logging
from typing import List, Dict, Any
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import mlflow
import matplotlib.pyplot as plt
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters."""
    required_keys = ['data', 'model']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    
    if not 0 < config['model']['test_size'] < 1:
        raise ValueError("test_size must be between 0 and 1")

def validate_data(df: pd.DataFrame, feature_columns: List[str], target_column: str) -> None:
    """
    Validate input data for missing values and required columns.
    
    Args:
        df: Input DataFrame
        feature_columns: List of required feature columns
        target_column: Name of target column
    
    Raises:
        ValueError: If validation fails
    """
    logger.info("Validating input data...")
    missing = df[feature_columns + [target_column]].isnull().sum()
    if missing.any():
        raise ValueError(f"Missing values found:\n{missing[missing > 0]}")
    if not set(feature_columns).issubset(df.columns):
        raise ValueError("Some feature columns are missing from the DataFrame.")
    logger.info("Data validation successful")

def log_metrics_and_artifacts(model, X_test, y_test, feat_names, mlflow_run, prefix=""):
    """
    Log model metrics and visualization artifacts to MLflow.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        feat_names: Feature names
        mlflow_run: Active MLflow run
        prefix: Prefix for metric names
    """
    logger.info("Calculating and logging metrics...")
    
    # Add validation checks
    if len(y_test) < 2:
        raise ValueError("Test set too small for meaningful metrics")
    
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    acc = (preds == y_test).mean()
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    
    # Add sanity checks for perfect scores
    if acc == 1.0 or auc == 1.0:
        logger.warning("Perfect scores detected - possible overfitting!")
    
    # Log metrics using mlflow directly
    metrics = {
        f"{prefix}accuracy": acc,
        f"{prefix}roc_auc": auc,
        f"{prefix}f1": report["1"]["f1-score"],
        f"{prefix}precision": report["1"]["precision"],
        f"{prefix}recall": report["1"]["recall"]
    }
    mlflow.log_metrics(metrics)  # Change this line
    logger.info(f"Logged metrics: accuracy={acc:.3f}, roc_auc={auc:.3f}")
    
    # Log confusion matrix
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
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")  # Change this line
    plt.close()
    
    # Log feature importances
    importances = model.feature_importances_
    plt.figure(figsize=(6,4))
    plt.barh(feat_names, importances)
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importances.png")
    mlflow.log_artifact("feature_importances.png")  # Change this line
    plt.close()