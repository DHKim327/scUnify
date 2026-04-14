from __future__ import annotations

from typing import Type


__all__ = ["resolve_trainer", "register_mixin"]

_ALIAS = {
    "geneformer": "GeneformerTrainer",
    "scgpt": "ScGPTTrainer",
    "scfoundation": "ScFoundationTrainer",
    "uce": "UCETrainer",
    "nicheformer": "NicheformerTrainer",
}

_IMPORTED: dict[str, type] = {}

# --- Mixin registry ---
_MIXIN_REGISTRY: dict[str, type] = {}

# Built-in task → default Mixin mapping
_TASK_DEFAULT_MIXIN = {
    "pretraining": "PretrainingMixin",
    "classification": "ClassificationMixin",
    "perturbation": "PerturbationMixin",
    "regression": "RegressionMixin",
    "integration": "IntegrationMixin",
}


def register_mixin(name: str, mixin_cls: type) -> None:
    """Register a custom Mixin class for use in config-driven task resolution.

    Usage (e.g. in Jupyter)::

        from scunify.trainer import register_mixin

        class MyMixin:
            def compute_loss(self, model, batch):
                ...

        register_mixin("MyMixin", MyMixin)
        # Then in YAML: task_param.mixin: "MyMixin"
    """
    _MIXIN_REGISTRY[name] = mixin_cls


def _resolve_mixin(name: str) -> type:
    """Look up a Mixin by name: user-registered first, then built-in."""
    if name in _MIXIN_REGISTRY:
        return _MIXIN_REGISTRY[name]

    # Lazy-load built-in mixins
    if name == "PretrainingMixin":
        from .tasks._pretraining import PretrainingMixin
        return PretrainingMixin
    if name == "ClassificationMixin":
        from .tasks._classification import ClassificationMixin
        return ClassificationMixin
    if name == "PerturbationMixin":
        from .tasks._perturbation import PerturbationMixin
        return PerturbationMixin
    if name == "RegressionMixin":
        from .tasks._regression import RegressionMixin
        return RegressionMixin
    if name == "IntegrationMixin":
        from .tasks._integration import IntegrationMixin
        return IntegrationMixin

    raise ValueError(
        f"Unknown Mixin: {name!r}. Use register_mixin() to register custom Mixins."
    )


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
    """Return the trainer class for the given config, composed with task Mixin.

    Resolution order for Mixin:
    1. ``training.task_param.mixin`` (explicit name) — highest priority
    2. ``training.task`` → default Mixin mapping
    3. Fallback to PretrainingMixin
    """
    model_name = cfg.get("model_name")
    if not model_name:
        raise KeyError("cfg must contain 'model_name'")

    key = model_name.replace(" ", "").lower()
    cls_name = _ALIAS.get(key)
    if cls_name is None:
        raise ValueError(f"No trainer registered for model: {model_name!r}")

    try:
        base_cls = __getattr__(cls_name)
    except AttributeError:
        raise ImportError(
            f"Trainer class '{cls_name}' not yet implemented. "
            f"Currently available: GeneformerTrainer, NicheformerTrainer, "
            f"ScFoundationTrainer, ScGPTTrainer, UCETrainer"
        )

    # --- Resolve task Mixin ---
    training_cfg = cfg.get("training", {})
    task = training_cfg.get("task", "pretraining")
    task_param = training_cfg.get("task_param", {})

    # Priority 1: explicit mixin name in task_param
    mixin_name = task_param.get("mixin") if task_param else None

    # Priority 2: default mixin for the task
    if not mixin_name:
        mixin_name = _TASK_DEFAULT_MIXIN.get(task, "PretrainingMixin")

    mixin_cls = _resolve_mixin(mixin_name)

    # Compose: Mixin(compute_loss) + ModelTrainer(everything else)
    composed_name = f"{cls_name}_{mixin_name}"
    return type(composed_name, (mixin_cls, base_cls), {})
