from . import logreg, xgboost_model, catboost_model

REGISTRY = {
    "logreg": logreg,
    "xgboost": xgboost_model,
    "catboost": catboost_model,
}

def get_model_module(name: str):
    return REGISTRY[name]