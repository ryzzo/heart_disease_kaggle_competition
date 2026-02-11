from . import logreg 
from . import xgboost_model
from . import catboost_model
from . import lightgbm_model

REGISTRY = {
    "logreg": logreg,
    "xgboost": xgboost_model,
    "catboost": catboost_model,
    "lightgbm": lightgbm_model,
}

def get_model_module(name: str):
    return REGISTRY[name]