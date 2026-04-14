"""PretrainingMixin — delegates compute_loss to model-specific pretraining loss.

This is the default task Mixin. It calls ``self.compute_pretraining_loss()``
which each model trainer implements (MLM for Geneformer/Nicheformer,
GEP+MVC for scGPT, MAE for scFoundation, BEP for UCE).
"""

import torch
import torch.nn as nn

from ._base import BaseMixin


class PretrainingMixin(BaseMixin):
    """Default task: self-supervised pretraining loss."""

    def compute_loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        return self.compute_pretraining_loss(model, batch)
