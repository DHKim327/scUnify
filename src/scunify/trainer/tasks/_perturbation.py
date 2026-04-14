"""PerturbationMixin — perturbation response prediction via LoRA.

Applies to:
- scGPT: control → perturbed expression prediction
  (Ref: Tutorial_Perturbation.ipynb, Cui et al. 2024)
- scFoundation: GEARS decoder connection
  (Ref: Hao et al. 2024, GEARS snap-stanford/GEARS)

Usage::

    training:
      task: perturbation
      task_param:
        n_genes: 2000
        decoder_hidden: 256
      label_keys: [target_expr]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import BaseMixin


class PerturbationMixin(BaseMixin):
    """Perturbation response prediction: gene_embedding → decoder → expression."""

    def build_model(self):
        model = super().build_model()

        task_cfg = self.training_cfg.get("task_param", {})
        decoder_hidden = int(task_cfg.get("decoder_hidden", 256))
        emb_dim = self._infer_emb_dim()

        model.pert_decoder = nn.Sequential(
            nn.Linear(emb_dim, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, 1),
        )
        return model

    def compute_loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        gene_emb = self.get_gene_embedding(model, batch)
        m = self._unwrap(model)
        pred = m.pert_decoder(gene_emb).squeeze(-1)

        label_keys = self.training_cfg.get("label_keys", [])
        if not label_keys:
            raise ValueError(
                "PerturbationMixin requires training.label_keys "
                "to specify the target expression column."
            )
        target = batch[label_keys[0]].float()
        min_len = min(pred.size(1), target.size(1))
        return F.mse_loss(pred[:, :min_len], target[:, :min_len])

    def get_task_output(self, model: nn.Module, batch: dict) -> dict:
        gene_emb = self.get_gene_embedding(model, batch)
        m = self._unwrap(model)
        pred = m.pert_decoder(gene_emb).squeeze(-1)
        return {
            "pert_prediction": {"data": pred, "storage": "obsm"},
        }
