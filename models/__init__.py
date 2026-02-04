from . import logreg

REGISTRY = {
    "logreg": logreg,
}

def get_model_module(name: str):
    return REGISTRY[name]