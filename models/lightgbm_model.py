import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def suggest(trial, cfg):
    search_cfg = cfg["search"]
    
    params = {
        # Core
        "n_estimators": trial.suggest_int(
            "n_estimators", 
            int(search_cfg["n_estimators_min"]), int(search_cfg["n_estimators_max"]),
            log=True
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 
            float(search_cfg["learning_rate_min"]), float(search_cfg["learning_rate_max"]), 
            log=True
        ),

        # Tree structure
        "num_leaves": trial.suggest_int(
            "num_leaves", 
            int(search_cfg["num_leaves_min"]), int(search_cfg["num_leaves_max"]),
            log=True
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            int(search_cfg["max_depth_min"]), int(search_cfg["max_depth_max"]),
        ),
        "min_child_samples": trial.suggest_int(
            "min_child_samples",
            int(search_cfg["min_child_samples_min"]), int(search_cfg["min_child_samples_max"]),
        ),

        # sampling
        "subsample": trial.suggest_float(
            "subsample", 
            float(search_cfg["subsample_min"]), float(search_cfg["subsample_max"])
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 
            float(search_cfg["colsample_bytree_min"]), float(search_cfg["colsample_bytree_max"])
        ),

        # regularization
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 
            float(search_cfg["reg_alpha_min"]), float(search_cfg["reg_alpha_max"]), 
            log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 
            float(search_cfg["reg_lambda_min"]), float(search_cfg["reg_lambda_max"]), 
            log=True
        ),
        "min_split_gain": trial.suggest_float(
            "min_split_gain",
            float(search_cfg["min_split_gain_min"]), float(search_cfg["min_split_gain_max"]),
        ),

        # Misc
        "class_weight": trial.suggest_categorical("class_weight", search_cfg["class_weight"]),
        "boosting_type": trial.suggest_categorical("boosting_type", search_cfg["boosting_type"]),
    }

    return params

def build(params, random_state):
    clf = lgb.LGBMClassifier(
        **params,
        random_state=random_state,
        n_jobs=-1,
        objective="binary",
        metric="auc",
        device_type="gpu",
        gpu_platform_id=0,
        gpu_device_id=0,
    )

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", clf),
    ])


def fit(model, X_train, y_train, X_val, y_val, trial, cfg):
    es = int(cfg.get("training", {}).get("early_stopping_rounds", 50))

    # callback fot early stopping and logging
    callbacks = [lgb.early_stopping(es, verbose=False)]

    try:
        from optuna.integration import LightGBMPruningCallback
        callbacks.append(LightGBMPruningCallback(trial, "auc"))
    except Exception:
        pass

    model.fit(
        X_train, y_train,
        model__eval_set = [(X_val, y_val)],
        model__eval_metric = "auc",
        model__callbacks = callbacks
    )

