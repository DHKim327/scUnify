"""RegressionMixin — continuous target regression via LoRA fine-tuning.

Applies to:
- scFoundation: Drug Response Prediction (Ref: scFoundation Fig.3a, DeepCDR)
- Nicheformer: Niche Regression / Density Regression (Ref: Schaar et al. 2025)

Usage::

    training:
      task: regression
      task_param:
        n_targets: 1
        head_hidden: 128
      label_keys: [drug_response]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import BaseMixin


class RegressionMixin(BaseMixin):
    """Continuous regression: cell_embedding → MLP → MSELoss."""

    def build_model(self):
        model = super().build_model()

        task_cfg = self.training_cfg.get("task_param", {})
        n_targets = int(task_cfg.get("n_targets", 1))
        head_hidden = int(task_cfg.get("head_hidden", 128))
        emb_dim = self._infer_emb_dim()

        model.regression_head = nn.Sequential(
            nn.Linear(emb_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, n_targets),
        )
        return model

    def compute_loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        cell_emb = self.get_cell_embedding(model, batch)
        m = self._unwrap(model)
        pred = m.regression_head(cell_emb)

        label_keys = self.training_cfg.get("label_keys", [])
        if not label_keys:
            raise ValueError(
                "RegressionMixin requires training.label_keys "
                "to specify the target column."
            )
        target = batch[label_keys[0]].float()
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        return F.mse_loss(pred, target)

    def get_task_output(self, model: nn.Module, batch: dict) -> dict:
        cell_emb = self.get_cell_embedding(model, batch)
        m = self._unwrap(model)
        pred = m.regression_head(cell_emb)
        return {
            "regression_pred": {"data": pred.squeeze(-1), "storage": "obs"},
        }
