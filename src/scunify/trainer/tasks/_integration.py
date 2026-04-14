"""IntegrationMixin — batch correction via adversarial training (GRL).

Applies to:
- scGPT: Batch Correction (Ref: Tutorial_Integration.ipynb, Cui et al. 2024)

Uses Gradient Reversal Layer (GRL) to learn batch-invariant embeddings:
  total_loss = task_loss + λ * reversed_batch_loss

Ref: Ganin & Lempitsky 2015 (Domain-Adversarial Training of Neural Networks)

Usage::

    training:
      task: integration
      task_param:
        n_classes: 10
        n_batches: 2
        lambda_adv: 1.0
      label_keys: [celltype, batch_id]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from ._base import BaseMixin


class _GradientReversal(Function):
    """Gradient Reversal Layer: identity forward, negated gradient backward.
    Ref: Ganin & Lempitsky, ICML 2015."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return _GradientReversal.apply(x, lambda_)


class IntegrationMixin(BaseMixin):
    """Batch correction via adversarial training."""

    def build_model(self):
        model = super().build_model()

        task_cfg = self.training_cfg.get("task_param", {})
        n_classes = int(task_cfg["n_classes"])
        n_batches = int(task_cfg["n_batches"])
        head_hidden = int(task_cfg.get("head_hidden", 128))
        emb_dim = self._infer_emb_dim()

        model.task_head = nn.Sequential(
            nn.Linear(emb_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, n_classes),
        )
        model.batch_head = nn.Sequential(
            nn.Linear(emb_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, n_batches),
        )
        return model

    def compute_loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        cell_emb = self.get_cell_embedding(model, batch)
        m = self._unwrap(model)

        label_keys = self.training_cfg.get("label_keys", [])
        if len(label_keys) < 2:
            raise ValueError(
                "IntegrationMixin requires at least 2 label_keys: "
                "[celltype_key, batch_key]"
            )

        task_cfg = self.training_cfg.get("task_param", {})
        lambda_adv = float(task_cfg.get("lambda_adv", 1.0))

        task_logits = m.task_head(cell_emb)
        task_loss = F.cross_entropy(task_logits, batch[label_keys[0]].long())

        reversed_emb = grad_reverse(cell_emb, lambda_adv)
        batch_logits = m.batch_head(reversed_emb)
        batch_loss = F.cross_entropy(batch_logits, batch[label_keys[1]].long())

        return task_loss + batch_loss

    def get_task_output(self, model: nn.Module, batch: dict) -> dict:
        cell_emb = self.get_cell_embedding(model, batch)
        m = self._unwrap(model)
        task_logits = m.task_head(cell_emb)
        batch_logits = m.batch_head(cell_emb)
        return {
            "celltype_logits": {"data": task_logits, "storage": "obsm"},
            "celltype_pred": {"data": task_logits.argmax(dim=-1), "storage": "obs"},
            "batch_logits": {"data": batch_logits, "storage": "obsm"},
        }
