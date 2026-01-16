from importlib.metadata import version
import importlib

from . import core, inferencer, registry, utils
from .config import ScUnifyConfig, setup
from .core.runner import ScUnifyRunner


__all__ = [
    "ScUnifyConfig",
    "ScUnifyRunner",
    "setup",
    "evaluation",  # 'eval' 대신 'evaluation' 사용 (Python 내장 함수와 충돌 방지)
]

# __version__ = "0.1.0"
__version__ = version("scUnify")


# Lazy import for evaluation module (requires scUnify[eval] dependencies)
# Python 내장 함수 'eval'과 충돌 방지를 위해 'evaluation'으로 명명
_LAZY_MODULES = {}

def __getattr__(name):
    if name == "evaluation":
        if "evaluation" not in _LAZY_MODULES:
            # importlib을 사용하여 명시적으로 import
            _LAZY_MODULES["evaluation"] = importlib.import_module(".eval", package="scunify")
        return _LAZY_MODULES["evaluation"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
