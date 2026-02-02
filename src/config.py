from pydantic import BaseModel

class TrainConfig(BaseModel):
    data_path: str = "data/raw/train.csv"
    target_col: str = "Heart Disease"

    positive_label: str = "Presence"
    negative_label: str = "Absence"

    test_size: float = 0.2
    random_state: int = 42

    experiment_name: str = "heart-disease"
    run_name: str = "baseline"

    registered_model_name: str = "LocalClassifier"
    primary_metric: str = "roc_auc"
