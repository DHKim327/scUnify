from importlib.metadata import version
import importlib

from .config import ScUnifyConfig, setup


__all__ = [
    "ScUnifyConfig",
    "ScUnifyRunner",
    "setup",
    "evaluation",  # Use 'evaluation' instead of 'eval' to avoid shadowing the Python builtin
]

__version__ = version("scUnify")


# Lazy imports for modules that require torch/heavy dependencies
# - core, inferencer, registry, utils: require torch (via ray.train.torch)
# - evaluation: requires scUnify[eval] dependencies
_LAZY_MODULES = {}

def __getattr__(name):
    _LAZY_MAP = {
        "core": ".core",
        "inferencer": ".inferencer",
        "registry": ".registry",
        "utils": ".utils",
        "ScUnifyRunner": ".core.runner",
        "evaluation": ".eval",
    }
    if name in _LAZY_MAP:
        if name not in _LAZY_MODULES:
            mod = importlib.import_module(_LAZY_MAP[name], package="scunify")
            _LAZY_MODULES[name] = mod
        # For ScUnifyRunner, return the class from the module
        if name == "ScUnifyRunner":
            return _LAZY_MODULES[name].ScUnifyRunner
        return _LAZY_MODULES[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
