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
            f"Currently available: GeneformerTrainer"
        )
