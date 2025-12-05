import re
import sys as _sys
from typing import Any, Dict, Type

from ._scfoundation_inferencer import ScFoundationInferencer
from ._scgpt_inferencer import ScGPTInferencer
from ._uce_inferencer import UCEInferencer


def _to_camel(name: str) -> str:
    parts = re.sub(r"[-_]+", " ", str(name)).strip().split()
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


_ALIAS = {
    "scfoundation": "ScFoundationInferencer",
    "scgpt": "ScGPTInferencer",
    "uce": "UCEInferencer",
}


def resolve_inferencer(cfg) -> type:
    """
    cfg.model_name 기준으로 inferencer 클래스를 반환.
    - inferencer 패키지 __init__ 에서 이미 클래스를 import 해두었다는 전제.
    - cfg에 'inferencer_class'가 있으면 그걸 우선 사용.
    """
    model_name = cfg.get("model_name")
    if not model_name:
        raise KeyError("cfg must contain 'model_name' or 'model.name'")

    key = model_name.replace(" ", "").lower()
    cls_name = _ALIAS.get(key) or f"{_to_camel(model_name)}Inferencer"

    try:
        return getattr(_sys.modules[__name__], cls_name)
    except AttributeError:
        raise ImportError(f"Inferencer class '{cls_name}' not found. Make sure scunify.inferencer.__init__ imports it.")
