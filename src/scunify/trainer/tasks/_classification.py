"""ClassificationMixin — cell type/state classification via LoRA fine-tuning.

Applies to:
- scGPT: Cell Type Identification (Ref: Tutorial_Annotation.ipynb)
- Geneformer: Cell State Classification (Ref: Theodoris et al., Nature 2023)
- Nicheformer: Niche Classification (Ref: Schaar et al., Nature Methods 2025)

Usage::

    training:
      task: classification
      task_param:
        n_classes: 10
        head_hidden: 128
      label_keys: [celltype]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import BaseMixin


class ClassificationMixin(BaseMixin):
    """Cell-level classification: cell_embedding → Linear → CrossEntropyLoss."""

    def build_model(self):
        model = super().build_model()

        task_cfg = self.training_cfg.get("task_param", {})
        n_classes = int(task_cfg["n_classes"])
        head_hidden = int(task_cfg.get("head_hidden", 128))
        emb_dim = self._infer_emb_dim()

        model.classifier_head = nn.Sequential(
            nn.Linear(emb_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, n_classes),
        )
        return model

    def compute_loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        cell_emb = self.get_cell_embedding(model, batch)
        m = self._unwrap(model)
        logits = m.classifier_head(cell_emb)

        label_keys = self.training_cfg.get("label_keys", [])
        if not label_keys:
            raise ValueError(
                "ClassificationMixin requires training.label_keys "
                "to specify the target column name."
            )
        return F.cross_entropy(logits, batch[label_keys[0]].long())

    def get_task_output(self, model: nn.Module, batch: dict) -> dict:
        cell_emb = self.get_cell_embedding(model, batch)
        m = self._unwrap(model)
        logits = m.classifier_head(cell_emb)
        preds = logits.argmax(dim=-1)
        return {
            "logits": {"data": logits, "storage": "obsm"},
            "predictions": {"data": preds, "storage": "obs"},
        }
