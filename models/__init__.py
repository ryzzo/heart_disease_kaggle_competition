from . import logreg, xgboost_model

REGISTRY = {
    "logreg": logreg,
    "xgboost": xgboost_model
}

def get_model_module(name: str):
    return REGISTRY[name]