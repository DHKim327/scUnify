"""Task Mixins for scUnify training.

Each Mixin provides ``compute_loss()`` and ``get_task_output()`` for a
specific downstream task. All inherit from ``BaseMixin``.

Built-in Mixins:
- PretrainingMixin: Self-supervised pretraining loss (MLM/GEP/MAE/BEP)
- ClassificationMixin: Cell type/state classification (CrossEntropy)
- IntegrationMixin: Batch correction via adversarial training (GRL)
- PerturbationMixin: Perturbation response prediction (MSE)
- RegressionMixin: Continuous target regression (MSE)
"""

from ._base import BaseMixin
from ._pretraining import PretrainingMixin
from ._classification import ClassificationMixin
from ._integration import IntegrationMixin
from ._perturbation import PerturbationMixin
from ._regression import RegressionMixin

__all__ = [
    "BaseMixin",
    "PretrainingMixin",
    "ClassificationMixin",
    "IntegrationMixin",
    "PerturbationMixin",
    "RegressionMixin",
]
