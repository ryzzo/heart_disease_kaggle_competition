from catboost import CatBoostClassifier

def suggest(trial, cfg):
    # Fixed training (mode: train)
    if trial is None:
        return cfg["catboost"]

    search_cfg = cfg["search"]

    return {
        "iterations": trial.suggest_int(
            "iterations", 
            int(search_cfg["iterations_min"]), int(search_cfg["iterations_max"]), 
            log=True
        ),
        "depth": trial.suggest_int(
            "depth",
            int(search_cfg["depth_min"]), int(search_cfg["depth_max"])
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 
            float(search_cfg["learning_rate_min"]), float(search_cfg["learning_rate_max"]), 
            log=True
        ),
        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg", 
            float(search_cfg["l2_leaf_reg_min"]), float(search_cfg["l2_leaf_reg_max"]), 
            log=True
        ),
        "subsample": trial.suggest_float(
            "subsample", 
            float(search_cfg["subsample_min"]), float(search_cfg["subsample_max"])
        ),
        "random_strength": trial.suggest_float(
            "random_strength", 
            float(search_cfg["random_strength_min"]), float(search_cfg["random_strength_max"])
        ),
        "bagging_temperature": trial.suggest_float(
            "bagging_temperature",
            float(search_cfg["bagging_temperature_min"]), float(search_cfg["bagging_temperature_max"]),
        ),
    }

def build(params, random_state):
    return CatBoostClassifier(
        **params,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_state,
        verbose=False,
        thread_count=-1,
    )