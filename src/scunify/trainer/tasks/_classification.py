"""ClassificationMixin v2 — backbone-agnostic.

Works with any BaseTrainer that provides paper-faithful hooks:
``default_head``, ``attach_task_head``, ``classifier_logits``.

Architecture per backbone (paper-faithful):
- scGPT          : ``ClsDecoder`` (3-layer Linear-ReLU-LN + Linear)         — Cui et al. 2024
- scFoundation   : ``BatchNorm1d(affine=False) → Linear-ReLU-Linear``        — Hao et al. 2024
- Nicheformer    : ``Linear(dim, n_cls, bias=False)``                       — Schaar et al. 2025
- Geneformer     : ``BertForSequenceClassification.from_pretrained(...)``    — Theodoris et al. 2023
- UCE            : ``Linear(emb, n_cls)``                                    — scunify default (UCE has no fine-tune)

Usage::

    import scunify as scu

    # Path A: yaml only — no Python file at all
    # YAML: training.task_param.mixin: "ClassificationMixin"
    #       training.label_keys: [celltype]
    #       training.task_param.n_classes: 14

    # Path B: subclass for declarative defaults
    class MyCellTypeMixin(ClassificationMixin):
        label_keys = ["celltype"]
        n_classes = 14

    scu.trainer.register_mixin("MyCellType", MyCellTypeMixin)

5-line task definition. Same code works on all 5 backbones.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import TaskMixin


class ClassificationMixin(TaskMixin):
    """Cross-entropy classification on cell embeddings (or integrated classifier
    output for HF-style backbones).

    Override ``label_keys`` / ``n_classes`` in subclass or via yaml. Override
    ``compute_loss`` only for class weighting / label smoothing / multi-label.
    """

    task_type = "classification"
    label_keys: list[str] = []
    n_classes: int = 0

    def compute_loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        keys = self._label_keys
        if not keys:
            raise ValueError(
                f"{type(self).__name__}.label_keys is empty. "
                f"Set it as class attr (e.g. label_keys=['celltype']) or "
                f"in yaml under training.label_keys."
            )
        logits = self.classifier_logits(model, batch)  # trainer-routed (paper-faithful)
        target = batch[keys[0]].long()
        return F.cross_entropy(logits, target)
