from __future__ import annotations

from typing import Type


__all__ = ["resolve_trainer"]

_ALIAS = {
    "geneformer": "GeneformerTrainer",
    "scgpt": "ScGPTTrainer",
    "scfoundation": "ScFoundationTrainer",
    "uce": "UCETrainer",
    "nicheformer": "NicheformerTrainer",
}

_IMPORTED: dict[str, type] = {}


def __getattr__(name: str):
    """Lazy import trainer classes."""
    if name == "GeneformerTrainer":
        if name not in _IMPORTED:
            from ._geneformer_trainer import GeneformerTrainer
            _IMPORTED[name] = GeneformerTrainer
        return _IMPORTED[name]
    if name == "NicheformerTrainer":
        if name not in _IMPORTED:
            from ._nicheformer_trainer import NicheformerTrainer
            _IMPORTED[name] = NicheformerTrainer
        return _IMPORTED[name]
    if name == "ScFoundationTrainer":
        if name not in _IMPORTED:
            from ._scfoundation_trainer import ScFoundationTrainer
            _IMPORTED[name] = ScFoundationTrainer
        return _IMPORTED[name]
    if name == "ScGPTTrainer":
        if name not in _IMPORTED:
            from ._scgpt_trainer import ScGPTTrainer
            _IMPORTED[name] = ScGPTTrainer
        return _IMPORTED[name]
    if name == "UCETrainer":
        if name not in _IMPORTED:
            from ._uce_trainer import UCETrainer
            _IMPORTED[name] = UCETrainer
        return _IMPORTED[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def resolve_trainer(cfg) -> Type:
    """Return the trainer class for the given config's model_name."""
    model_name = cfg.get("model_name")
    if not model_name:
        raise KeyError("cfg must contain 'model_name'")

    key = model_name.replace(" ", "").lower()
    cls_name = _ALIAS.get(key)
    if cls_name is None:
        raise ValueError(f"No trainer registered for model: {model_name!r}")

    try:
        return __getattr__(cls_name)
    except AttributeError:
        raise ImportError(
            f"Trainer class '{cls_name}' not yet implemented. "
            f"Currently available: GeneformerTrainer, NicheformerTrainer, "
            f"ScFoundationTrainer, ScGPTTrainer, UCETrainer"
        )
