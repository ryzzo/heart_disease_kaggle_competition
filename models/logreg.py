from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def suggest(trial, cfg):
    if trial is None:
        return cfg["logreg"]

    C = trial.suggest_float("C", cfg["search"]["C_min"], cfg["search"]["C_max"], log=True)
    max_iter = trial.suggest_int("max_iter", 500, 5000, log=True)

    return {
        "C": C,
        "max_iter": max_iter,
        "class_weight": None,
    }

def build(params, random_state):
    clf = LogisticRegression(
        C=float(params["C"]),
        max_iter=int(params["max_iter"]),
        class_weight=params.get("class_weight"),
        random_state=random_state,
    )

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", clf),
    ])