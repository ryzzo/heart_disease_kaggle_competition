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

    random_state = int(cfg["data"]["random_state"])

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=float(cfg["data"]["test_size"]),
        random_state=random_state,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.2,
        random_state=int(cfg["data"]["random_state"]),
        stratify=y_trainval,
    )
    
    y_train_encoded, encoder = encode_target(y_train)
    y_val_encoded = encoder.transform(y_val)
    y_test_encoded = encoder.transform(y_test)

    model_type = cfg["model"]["type"]
    mod = get_model_module(model_type)

    def evaluate_on(model, X_eval, y_eval_encoded):
        if len(X_eval) != len(y_eval_encoded):
            raise ValueError(f"Length mismatch: X_eval={len(X_eval)} vs y_eval={len(y_eval_encoded)}")

        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval_encoded, preds)

        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_eval)[:, 1]
        else:
            scores = model.decision_function(X_eval)

        auc = roc_auc_score(y_eval_encoded, scores)
        return acc, auc

    def objective(trial):
        params = mod.suggest(trial, cfg)
        model = mod.build(params, random_state)
        
        if hasattr(mod, "fit"):
            mod.fit(model, X_train, y_train_encoded, X_val, y_val_encoded, trial, cfg)
        else:
            model.fit(X_train, y_train_encoded)

        acc, auc = evaluate_on(model, X_val, y_val_encoded)

        with mlflow.start_run(nested=True):
            mlflow.log_params({"model_type": model_type, **params})
            mlflow.log_metrics({"accuracy": acc, "val_roc_auc": auc})

        return auc
    
    mlflow.set_experiment(cfg["experiment"]["name"])

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    y_path = artifacts_dir / "y_encoder.pkl"
    with open(y_path, "wb") as f:
        pickle.dump(encoder, f)
    
    with mlflow.start_run(run_name=f"{model_type}_optuna") as parent:
        mlflow.log_artifact(cfg_path)
        mlflow.log_artifact(str(y_path), artifact_path="encoders")
        mlflow.log_params(cfg.get("search", {}))

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=int(cfg["search"]["n_trials"]),
            timeout=None if cfg["search"]["timeout_seconds"] == 0 else int(cfg["search"]["timeout_seconds"]),
        )

        best_params = study.best_params.copy()

        # unpack combined choice into fields build() expects
        if "solver_penalty" in best_params:
            solver, penalty = best_params.pop("solver_penalty")
            best_params["solver"] = solver
            best_params["penalty"] = penalty

        best_model = mod.build(best_params, random_state)

        best_model.fit(X_trainval, encoder.transform(y_trainval))
        test_acc, test_auc = evaluate_on(best_model, X_test, y_test_encoded)

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics({"test_accuracy": test_acc, "best_roc_auc": test_auc})

        model_path = artifacts_dir / "best_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        mlflow.log_artifact(str(model_path), artifact_path="best_model_artifact")
        mlflow.sklearn.log_model(best_model, artifact_path="best_model_mlflow")

        print("Best model saved")

if __name__ == "__main__":
    main()


    
