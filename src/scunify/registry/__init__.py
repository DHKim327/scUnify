# Lazy imports to avoid loading model dependencies
__all__ = ["dataset", "models"]

def __getattr__(name):
    if name == "dataset":
        from . import dataset
        return dataset
    elif name == "models":
        from . import models
        return models
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
