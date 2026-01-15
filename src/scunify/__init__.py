from importlib.metadata import version

from . import core, inferencer, registry, utils
from .config import ScUnifyConfig, setup
from .core.runner import ScUnifyRunner


__all__ = [
    "ScUnifyConfig",
    "ScUnifyRunnersetup",
]

# __version__ = "0.1.0"
__version__ = version("scUnify")
