"""RegressionMixin v2 — backbone-agnostic MSE regression on cell embeddings.

Architecture per backbone (paper-faithful default head — Linear(emb_dim, n_outputs)):
- Nicheformer    : ``Linear(dim_model, n_outputs, bias=False)``  — Schaar et al. 2025
                   (``_fine_tune_model.py`` ``density_regression`` /
                   ``niche_regression`` recipe)
- Other backbones: not yet covered (intentional — only Nicheformer paper has
                   regression task in scUnify Tool paper R4 scope).

Label sourcing — the dataset's label processing routes by where the label is
defined in the input adata:

    label key in ``adata.obs[k]``       → ``batch[k]`` is a (B,) tensor
    label key in ``adata.obsm[k]``      → ``batch[k]`` is a (B, D) tensor
                                          (sparse → dense at __getitem__)

The mixin's ``compute_loss`` casts to float and squeezes head output when the
target is scalar, so the same code handles both ``density`` (scalar) and
``X_niche_N`` (vector) without yaml changes other than ``label_keys``.

Usage::

    # YAML — scalar regression (cell density)
    training:
      task: regression
      task_param:
        mixin: RegressionMixin
        n_classes: 1                 # output dim (= 1 scalar)
      label_keys: [density]

    # YAML — vector regression (niche composition)
    training:
      task: regression
      task_param:
        mixin: RegressionMixin
        n_classes: 22                # output dim (= n_cell_types)
      label_keys: [X_niche_3]        # paper niche size index
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import TaskMixin


class RegressionMixin(TaskMixin):
    """MSE regression on cell embeddings.

    Override ``label_keys`` / ``n_classes`` (output dim) in subclass or
    via yaml. Override ``compute_loss`` only for custom loss weighting.
    """

    task_type = "regression"
    label_keys: list[str] = []
    n_classes: int = 1   # output dim — scalar by default

    monitor = "val_loss"
    monitor_direction = "min"

    def compute_loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        keys = self._label_keys
        if not keys:
            raise ValueError(
                f"{type(self).__name__}.label_keys is empty. "
                f"Set as class attr (e.g. label_keys=['density']) or in yaml "
                f"under training.label_keys."
            )

        cell_emb = self.encode(model, batch)
        head = self._head(model)
        pred = head(cell_emb)                 # (B, n_outputs)
        target = batch[keys[0]].float()       # (B,) or (B, D)

        # Shape-align: scalar target (B,) vs (B, 1) head output → squeeze
        if pred.ndim == 2 and pred.shape[-1] == 1 and target.ndim == 1:
            pred = pred.squeeze(-1)
        elif pred.ndim == 1 and target.ndim == 2 and target.shape[-1] == 1:
            target = target.squeeze(-1)

        return F.mse_loss(pred, target)

    def predict(self, model: nn.Module, batch: dict) -> dict:
        """Per-cell prediction (B, n_outputs) — final-epoch extraction."""
        cell_emb = self.encode(model, batch)
        head = self._head(model)
        pred = head(cell_emb)
        return {
            "prediction": {"data": pred, "storage": "obsm"},
        }

    def inference_adata(self, full_adata):
        """Restrict ``save.outputs`` extraction to the test split.

        Paper convention (Suppl. Table 5 evaluation): regression metrics are
        reported on held-out test cells only. Output shapes will therefore be
        ``(n_test, ...)`` instead of ``(n_full, ...)``.
        Falls back to ``full_adata`` if the fold column is absent.
        """
        fold_keys = self.training_cfg.get("split", {}).get("fold_keys") or ["fold_0"]
        col = fold_keys[0]
        if col in full_adata.obs.columns:
            mask = full_adata.obs[col].astype(str) == "test"
            if mask.any():
                return full_adata[mask].copy()
        return full_adata
