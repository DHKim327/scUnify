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


def _ensure_user_codes_on_path() -> None:
    """Ensure all ``Codes/`` directories registered by ScUnifyConfig
    (via ``SCUNIFY_USER_CODES_PATHS`` env var) are on ``sys.path``.

    Called inside Ray workers: workers inherit env vars but start with a
    fresh ``sys.path``, so we re-apply on the worker side.
    """
    import os
    import sys

    paths = os.environ.get("SCUNIFY_USER_CODES_PATHS", "")
    for p in paths.split(os.pathsep):
        if p and p not in sys.path:
            sys.path.insert(0, p)


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


_LEGACY_CLS_ALIASES = {
    # v1 backbone-specific classification mixin names → backbone-agnostic v2.
    # The classifier head is now sourced from ``BaseTrainer.default_head`` per
    # backbone (paper-faithful), so the same ``ClassificationMixin`` covers all
    # 5 backbones. Old yaml ``mixin: ScGPTClassificationMixin`` keeps working.
    "ScGPTClassificationMixin",
    "GeneformerClassificationMixin",
    "NicheformerClassificationMixin",
    "ScFoundationClassificationMixin",
    "UCEClassificationMixin",
}


def _resolve_mixin(name: str) -> type:
    """Look up a Mixin by name.

    Resolution order:
    1. ``register_mixin()`` registry (user-registered, highest priority)
    2. Python import path (``my_mil.MILMixin``) — auto-imported from any
       ``Codes/`` dir registered by ScUnifyConfig
    3. Framework built-in (``ClassificationMixin``, ``ScGPTIntegrationMixin``, ...)
    """
    _ensure_user_codes_on_path()

    if name in _MIXIN_REGISTRY:
        return _MIXIN_REGISTRY[name]

    # Python import path form (e.g. ``my_mil.MILMixin``)
    if "." in name:
        import importlib

        module_path, class_name = name.rsplit(".", 1)
        try:
            mod = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            raise ValueError(
                f"Could not import mixin module {module_path!r} for "
                f"yaml ``mixin: {name}``. Ensure the file lives in a "
                f"``Codes/`` dir next to your yaml. ({exc})"
            ) from exc
        try:
            return getattr(mod, class_name)
        except AttributeError as exc:
            raise ValueError(
                f"Module {module_path!r} has no class {class_name!r} "
                f"(yaml ``mixin: {name}``)."
            ) from exc

    # Backbone-agnostic classification (paper-faithful via BaseTrainer hooks)
    if name == "ClassificationMixin" or name in _LEGACY_CLS_ALIASES:
        from .tasks._classification import ClassificationMixin
        return ClassificationMixin

    # Backbone-agnostic regression (paper-faithful: Linear head + MSE)
    if name == "RegressionMixin":
        from .tasks._regression import RegressionMixin
        return RegressionMixin

    # Task-specific framework-built-in Mixins
    if name == "ScGPTIntegrationMixin":
        from .tasks._integration_scgpt import ScGPTIntegrationMixin
        return ScGPTIntegrationMixin
    if name == "ScGPTPerturbationMixin":
        from .tasks._perturbation_scgpt import ScGPTPerturbationMixin
        return ScGPTPerturbationMixin
    if name == "ScFoundationPerturbationMixin":
        from .tasks._perturbation_scfoundation import ScFoundationPerturbationMixin
        return ScFoundationPerturbationMixin

    raise ValueError(
        f"Unknown Mixin: {name!r}. Either:\n"
        f"  (a) use a built-in name (e.g. 'ClassificationMixin', 'ScGPTIntegrationMixin'),\n"
        f"  (b) write your mixin in ``Codes/<file>.py`` next to your yaml and "
        f"reference it as ``mixin: <file>.<ClassName>``,\n"
        f"  (c) call ``scunify.trainer.register_mixin(name, cls)`` before runner.run()."
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
    task_param = training_cfg.get("task_param", {})

    # Explicit mixin name in task_param is required
    mixin_name = task_param.get("mixin") if task_param else None
    if not mixin_name:
        raise ValueError(
            "training.task_param.mixin must be specified. "
            "Use a built-in Mixin name (e.g. 'ClassificationMixin') "
            "or register a custom one via scunify.trainer.register_mixin()."
        )

    mixin_cls = _resolve_mixin(mixin_name)

    # Compose: Mixin(compute_loss) + ModelTrainer(everything else)
    composed_name = f"{cls_name}_{mixin_name}"
    return type(composed_name, (mixin_cls, base_cls), {})
