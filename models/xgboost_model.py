from xgboost import XGBClassifier

def suggest(trial, cfg):
    search_cfg = cfg["search"]

    # for fixed training
    if trial is None:
        return cfg["xgboost"]
    
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators", 
            search_cfg["n_estimators_min"], search_cfg["n_estimators_max"],
            log=True
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            search_cfg["max_depth_min"], search_cfg["max_depth_max"],
            log=True
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 
            search_cfg["learning_rate_min"], search_cfg["learning_rate_max"], 
            log=True
        ),
        "subsample": trial.suggest_float(
            "subsample", 
            search_cfg["subsample_min"], search_cfg["subsample_max"]
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 
            search_cfg["colsample_bytree_min"], search_cfg["colsample_bytree_max"]
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 
            search_cfg["min_child_weight_min"], search_cfg["min_child_weight_max"], 
            log=True
        ),
        "gamma": trial.suggest_float(
            "gamma", 
            search_cfg["gamma_min"], search_cfg["gamma_max"]
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 
            search_cfg["reg_alpha_min"], search_cfg["reg_alpha_max"], 
            log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 
            search_cfg["reg_lambda_min"], search_cfg["reg_lambda_max"], 
            log=True
        ),
    }

def build(params, random_state):
    return XGBClassifier(
        **params,
        random_state=random_state,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1
    )