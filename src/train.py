import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from .config import TrainConfig
from .data import load_data, split_xy_and_encode, make_train_test
from .features import build_preprocessor

import os
print("MLFLOW_TRACKING_URI env:", os.getenv("MLFLOW_TRACKING_URI"))
print("mlflow tracking uri:", mlflow.get_tracking_uri())

Path("/app/reports/figures").mkdir(parents=True, exist_ok=True)

def train_baseline(cfg: TrainConfig):
    df = load_data(cfg.data_path)
    X, y = split_xy_and_encode(df, cfg.target_col, cfg.positive_label, cfg.negative_label)
    X_train, X_test, y_train, y_test = make_train_test(X, y, cfg.test_size, cfg.random_state)

    preprocessor = build_preprocessor(X_train)
    model = LogisticRegression(max_iter=200, n_jobs=None)

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("target_col", cfg.target_col)
        mlflow.log_param("positive_label", cfg.positive_label)
        mlflow.log_param("negative_label", cfg.negative_label)
        mlflow.log_param("test_size", cfg.test_size)
        mlflow.log_param("random_state", cfg.random_state)

        pipeline.fit(X_train, y_train)

        proba = pipeline.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("roc_auc", float(roc_auc))
        mlflow.log_metric("accuracy", float(acc))

        # Artifact
        cm = confusion_matrix(y_test, preds)
        fig = plt.figure()
        plt.imshow(cm)
        plt.title("Confusion matrix (threshold=0.5)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        for (i, j), v in __import__("numpy").ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center")

        ART_DIR = Path("/app/reports/figures")
        ART_DIR.mkdir(parents=True, exist_ok=True)

        artifact_path = ART_DIR / "confusion_matrix.png"

        fig.savefig(artifact_path, bbox_inches="tight")
        plt.close(fig)

        mlflow.log_artifact(artifact_path, artifact_path="figures")

        # Log + register model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=cfg.registered_model_name
        )

        print({"roc_auc": roc_auc, "accuracy": acc, "cm": cm.tolist()})

if __name__ == "__main__":
    cfg = TrainConfig()
    train_baseline(cfg)