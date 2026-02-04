import sys
import yaml
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from models import get_model_module

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def read_csv_checked(path: str | Path) -> pd.DataFrame:
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path.resolve()}")

        return pd.read_csv(path)

def make_preprocess_pipeline(X):
    numeric_cols = X.select_dtypes(include=["number"]).columns

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
        ],
    )

def encode_target(y):
    le = LabelEncoder()
    return le.fit_transform(y), le

def decode_target(y_pred, encoder):
    return encoder.inverse_transform(y_pred)
    

def main():
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: python train.py <config_path>")
    
    cfg_path = sys.argv[1]
    cfg = load_config(cfg_path)

    df = read_csv_checked("data/raw/train.csv")

    target_col = "Heart Disease"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=float(cfg["data"]["test_size"]),
        random_state=int(cfg["data"]["random_state"]),
        stratify=y,
    )
    random_state = int(cfg["data"]["random_state"])

    X_transformer = make_preprocess_pipeline(X_train)
    X_train_scaled = X_transformer.fit_transform(X_train)
    X_test_scaled = X_transformer.transform(X_test)

    y_train_encoded, encoder = encode_target(y_train)
    y_test_encoded = encoder.transform(y_test)

    def evaluate(model):
        model.fit(X_train_scaled, y_train_encoded)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test_encoded, preds)

        scores = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_test)
        )
        auc = roc_auc_score(y_test, scores)
        return acc, auc
    
    model_type = cfg["model"]["type"]
    mod = get_model_module(model_type)

    def objective(trial):
        params = mod.suggest(trial, cfg)
        model = mod.build(params, random_state)

        acc, auc = evaluate(model)

        with mlflow.start_run(nested=True):
            mlflow.log_params({"model_type": model_type, **params})
            mlflow.log_metrics({"accuracy": acc, "roc_auc": auc})

        return auc
    
    mlflow.set_experiment(cfg["experiment"]["name"])

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    x_path = artifacts_dir / "X_transform.pkl"
    y_path = artifacts_dir / "y_transform.pkl"

    with open(x_path, "wb") as f:
        pickle.dump(X_transformer, f)

    with open(y_path, "wb") as f:
        pickle.dump(encoder, f)
    
    with mlflow.start_run(run_name=f"{model_type}_optuna") as parent:
        mlflow.log_artifact(cfg_path)
        mlflow.log_params(cfg["search"])

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=int(cfg["search"]["n_trials"]),
            timeout=None if cfg["search"]["timeout_seconds"] == 0 else cfg["search"]["timeout_seconds"],
        )

        best_params = study.best_params
        best_model = mod.build(best_params, random_state)
        best_acc, best_auc = evaluate(best_model)

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics({"best_accuracy": best_acc, "best_roc_auc": best_auc})

        model_path = artifacts_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        mlflow.log_artifact(str(model_path), artifact_path="best_model_artifact")
        mlflow.log_artifact(str(x_path), artifact_path="x_transformer_artifact")
        mlflow.log_artifact(str(y_path), artifact_path="y_transformer_artifact")
        mlflow.sklearn.log_model(best_model, artifact_path="best_model_mlflow")

        print("Best model saved")

if __name__ == "__main__":
    main()


    
