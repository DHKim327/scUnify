from importlib.metadata import version
import importlib

from . import core, inferencer, registry, utils
from .config import ScUnifyConfig, setup
from .core.runner import ScUnifyRunner


__all__ = [
    "ScUnifyConfig",
    "ScUnifyRunner",
    "setup",
    "evaluation",  # Use 'evaluation' instead of 'eval' to avoid shadowing the Python builtin
]

__version__ = version("scUnify")


# Lazy import for evaluation module (requires scUnify[eval] dependencies)
# Named 'evaluation' to avoid collision with the Python builtin 'eval'
_LAZY_MODULES = {}

def __getattr__(name):
    if name == "evaluation":
        if "evaluation" not in _LAZY_MODULES:
            _LAZY_MODULES["evaluation"] = importlib.import_module(".eval", package="scunify")
        return _LAZY_MODULES["evaluation"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
