import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score
from features import get_features_and_target, FEATURE_COLUMNS, TARGET_COLUMN
from modeling.utils import validate_data, log_metrics_and_artifacts
import numpy as np
import random
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

random.seed(config["random_state"])
np.random.seed(config["random_state"])

def load_data():
    df = pd.read_csv(config["data_path"])
    validate_data(df, FEATURE_COLUMNS, TARGET_COLUMN)
    X, y = get_features_and_target(df)
    return train_test_split(X, y, test_size=config["test_size"], random_state=config["random_state"], stratify=y)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "class_weight": "balanced",
        "random_state": config["random_state"]
    }
    with mlflow.start_run(nested=True) as run:
        model = RandomForestClassifier(**params)
        f1_cv = cross_val_score(model, X_train, y_train, cv=5, scoring="f1").mean()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)
        mlflow.log_params(params)
        mlflow.log_metrics({"f1_score": f1, "f1_cv": f1_cv, "roc_auc": auc})
        log_metrics_and_artifacts(model, X_test, y_test, X_train.columns, mlflow)
        if not hasattr(objective, "best_f1") or f1 > objective.best_f1:
            objective.best_f1 = f1
            joblib.dump(model, os.path.join(config["model_dir"], "best_model.pkl"))
            mlflow.sklearn.log_model(model, "best_churn_model")
        return f1

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    mlflow.set_experiment("churn-prediction-optuna")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    print("Best trial:", study.best_trial.params)