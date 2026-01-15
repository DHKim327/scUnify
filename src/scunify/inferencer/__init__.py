import re
from typing import Type

# Lazy imports
__all__ = ["resolve_inferencer", "ScFoundationInferencer", "ScGPTInferencer", "UCEInferencer"]

def _to_camel(name: str) -> str:
    parts = re.sub(r"[-_]+", " ", str(name)).strip().split()
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


_ALIAS = {
    "scfoundation": "ScFoundationInferencer",
    "scgpt": "ScGPTInferencer",
    "uce": "UCEInferencer",
}

_IMPORTED = {}


def __getattr__(name):
    """Lazy import inferencer classes"""
    if name == "ScFoundationInferencer":
        if name not in _IMPORTED:
            from ._scfoundation_inferencer import ScFoundationInferencer
            _IMPORTED[name] = ScFoundationInferencer
        return _IMPORTED[name]
    elif name == "ScGPTInferencer":
        if name not in _IMPORTED:
            from ._scgpt_inferencer import ScGPTInferencer
            _IMPORTED[name] = ScGPTInferencer
        return _IMPORTED[name]
    elif name == "UCEInferencer":
        if name not in _IMPORTED:
            from ._uce_inferencer import UCEInferencer
            _IMPORTED[name] = UCEInferencer
        return _IMPORTED[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def resolve_inferencer(cfg) -> Type:
    """
    cfg.model_name 기준으로 inferencer 클래스를 반환.
    Lazy import로 필요할 때만 로드.
    """
    model_name = cfg.get("model_name")
    if not model_name:
        raise KeyError("cfg must contain 'model_name' or 'model.name'")

    key = model_name.replace(" ", "").lower()
    cls_name = _ALIAS.get(key) or f"{_to_camel(model_name)}Inferencer"

    try:
        return __getattr__(cls_name)
    except AttributeError:
        raise ImportError(f"Inferencer class '{cls_name}' not found.")
