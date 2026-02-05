from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def suggest(trial, cfg):
    search_cfg = cfg["search"]

    valid_solver_penalty = [
        ("lbfgs", "l2"),
        ("liblinear", "l1"),
        ("liblinear", "l2"),
        ("newton-cg", "l2"),
        ("sag", "l2"),
        ("saga", "l1"),
        ("saga", "l2"),
        ("saga", "elasticnet"),
    ]

    # choose solver depending on penalty
    solver, penalty = trial.suggest_categorical(
        "solver_penalty",
        valid_solver_penalty,
    )

    # Elastic-net ratio only if needed
    l1_ratio = None
    if penalty == "elasticnet":
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

    # continous ranges from YAML
    C = trial.suggest_float(
        "C",
        float(search_cfg["C_min"]),
        float(search_cfg["C_max"]),
        log=True,
    )
    
    max_iter = trial.suggest_int(
        "max_iter",
        int(search_cfg["max_iter_min"]),
        int(search_cfg["max_iter_max"]),
        log=True,
    )

    tol = trial.suggest_float(
        "tol",
        float(search_cfg["tol_min"]),
        float(search_cfg["tol_max"]),
        log=True,
    )

    # categorical choices from YAML
    class_weight = trial.suggest_categorical("class_weight", search_cfg["class_weight"])

    fit_intercept = trial.suggest_categorical("fit_intercept", search_cfg["fit_intercept"])

    warm_start = trial.suggest_categorical("warm_start", search_cfg["warm_start"])

    params = {
        "solver": solver,
        "C": C,
        "penalty": penalty,
        "l1_ratio": l1_ratio,
        "max_iter": max_iter,
        "class_weight": class_weight,
        "tol": tol,
        "fit_intercept": fit_intercept,
        "warm_start": warm_start,
    }

    if penalty == "elasticnet":
        params["l1_ratio"] = l1_ratio

    return params

def build(params, random_state):
    # Allow either ("solver_penalty") or separate keys
    if "solver_penalty" in params and ("solver" not in params or "penalty" not in params):
        solver, penalty = params["solver_penalty"]
    else:
        solver = params["solver"]
        penalty = params["penalty"]

    lr_kwargs = dict(
        solver=solver,
        penalty=penalty,
        C=float(params["C"]),
        tol=float(params["tol"]),
        max_iter=int(params["max_iter"]),
        class_weight=params.get("class_weight"),
        fit_intercept=bool(params.get("fit_intercept", True)),
        warm_start=bool(params.get("warm_start", False)),
        random_state=random_state,
    )

    # Only include l1_ratio for elasticnet
    if penalty == "elasticnet":
        lr_kwargs["l1_ratio"] = float(params["l1_ratio"])

    clf = LogisticRegression(**lr_kwargs)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", clf),
    ])

def fit(model, X_train, y_train, X_val, y_val, trial, cfg):
    model.fit(X_train, y_train)