"""Task Mixins for scUnify training.

Layer 3 (user-facing) — built on Layer 2 (per-model BaseTrainer hooks).

- ``TaskMixin`` (`_base.py`)              — base class: declarative attrs +
                                            smart defaults. Inherit and override
                                            ``compute_loss`` for a new task.
- ``ClassificationMixin`` (`_classification.py`) — backbone-agnostic
                                            cross-entropy classification.
                                            Works on all 5 backbones via the
                                            BaseTrainer ``default_head`` /
                                            ``attach_task_head`` hooks.
- ``RegressionMixin`` (`_regression.py`)  — backbone-agnostic MSE regression
                                            (scalar via obs / vector via obsm).
- ``ScGPTIntegrationMixin``  (`_integration_scgpt.py`)      — paper INT task
- ``ScGPTPerturbationMixin`` (`_perturbation_scgpt.py`)     — paper Pert task
- ``ScFoundationPerturbationMixin`` (`_perturbation_scfoundation.py`) — paper Pert
"""

from ._base import TaskMixin
from ._classification import ClassificationMixin
from ._regression import RegressionMixin

__all__ = [
    "TaskMixin",
    "ClassificationMixin",
    "RegressionMixin",
    "ScGPTIntegrationMixin",
    "ScGPTPerturbationMixin",
    "ScFoundationPerturbationMixin",
]


def __getattr__(name):
    """Lazy import task-specific Mixins (avoid cross-model dependency issues)."""
    if name == "ScGPTIntegrationMixin":
        from ._integration_scgpt import ScGPTIntegrationMixin
        return ScGPTIntegrationMixin
    if name == "ScGPTPerturbationMixin":
        from ._perturbation_scgpt import ScGPTPerturbationMixin
        return ScGPTPerturbationMixin
    if name == "ScFoundationPerturbationMixin":
        from ._perturbation_scfoundation import ScFoundationPerturbationMixin
        return ScFoundationPerturbationMixin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
