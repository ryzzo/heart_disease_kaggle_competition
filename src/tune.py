import mlflow
import mlflow.sklearn
import optuna

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .config import TrainConfig
from .data import load_data, split_xy, make_train_test
from .features import build_preprocessor

def tune(cfg: TrainConfig, n_trials: int = 25):
    df = load_data(cfg.data_path)
    X, y = split_xy(df, cfg.target_col)
    X_train, X_test, y_train, y_test = make_train_test(X, y, cfg.test_size, cfg.random_state)

    preprocessor = build_preprocessor(X_train)

    mlflow.set_experiment(cfg.experiment_name)

    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 1e-3, 100.0, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
        penalty = "l2" if solver == "lbfgs" else trial.suggest_categorical("penalty", ["l1", "l2"])

        model = LogisticRegression(
            C=C,
            solver=solver,
            penalty=penalty,
            max_iter=800,
        )

        pipeline = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ])

        with mlflow.start_run(nested=True):
            mlflow.log_param("C", C)
            mlflow.log_param("solver", solver)
            mlflow.log_param("penalty", penalty)

            pipeline.fit(X_train, y_train)
            proba = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, proba)

            mlflow.log_metric("roc_auc", float(roc_auc))
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

        return roc_auc
    
    with mlflow.start_run(run_name="optuna_tuning"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        mlflow.log_param("n_trials", n_trials)
        mlflow.log_metric("best_roc_auc", float(study.best_value))
        for k, v in study.best_params.items():
            mlflow.log_param(f"best_{k}", v)

        print("Best ROC-AUC:", study.best_value)
        print("Best params:", study.best_params)

if __name__ == "__main__":
    cfg = TrainConfig()
    tune(cfg, n_trials=25)