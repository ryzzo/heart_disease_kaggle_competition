import mlflow
from mlflow.tracking import MlflowClient

from .config import TrainConfig

def register_best_model(cfg: TrainConfig):
    client = MlflowClient()

    # find best run in the experiments by metric
    exp = client.get_experiment_by_name(cfg.experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{cfg.experiment_name}' not found.")
    
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="",
        order_by=[f"metrics.{cfg.primary_metric} DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No runs found to register.")
    
    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_metric = best_run.data.metrics.get(cfg.primary_metric)

    model_uri = f"runs:/{best_run_id}/model"

    result = mlflow.register_model(model_uri=model_uri, name=cfg.registered_model_name)

    print(f"Registered model: {cfg.registered_model_name}")
    print(f"Version: {result.version}")
    print(f"Best run id: {best_run_id}")
    print(f"Best {cfg.primary_metric}: {best_metric}")

    return result.name, result.version, best_run_id

if __name__ == "__main__":
    cfg = TrainConfig()
    register_best_model(cfg)