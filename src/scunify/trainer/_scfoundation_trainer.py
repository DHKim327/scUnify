"""scFoundation LoRA trainer — downstream tasks only.

No pretraining MAE support. Uses fixed-length (B, 19266) input
with batch-wise gatherData for batch_size > 1.
"""

import torch
import torch.nn as nn

from .base._basetrainer import BaseTrainer
from .dataset._scfoundation_dataset import ScFoundationTrainingDataset
from .lora._injection import inject_lora_to_model
from .models._scfoundation_wrapper import ScFoundationTrainingWrapper


class _ScFoundationCls(nn.Module):
    """scFoundation paper-faithful classification head.

    Verbatim copy of original ``LinearProbingClassifier`` head body in
    ``Foundations/scFoundation/model/finetune_model.py:35-40``::

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_class),
        )
        self.norm = nn.BatchNorm1d(hidden_dim, affine=False, eps=1e-6)
        # forward: norm(x) → fc1

    Sub-module names (``norm`` / ``fc1``) match the paper exactly.
    """

    def __init__(self, emb_dim: int, n_classes: int):
        super().__init__()
        self.norm = nn.BatchNorm1d(emb_dim, affine=False, eps=1e-6)
        self.fc1 = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(self.norm(x))


class ScFoundationTrainer(BaseTrainer):
    """LoRA trainer for scFoundation (downstream tasks, batch_size > 1)."""

    def build_dataset(self, adata):
        return ScFoundationTrainingDataset(adata, self.cfg)

    def build_model(self):
        return ScFoundationTrainingWrapper(self.cfg)

    def inject_lora(self, model: nn.Module) -> nn.Module:
        return inject_lora_to_model(
            model, "scfoundation", self.lora_cfg, self.freeze_cfg, self.is_backbone_param,
        )

    def compute_pretraining_loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """Not supported — scFoundation pretraining removed."""
        raise NotImplementedError(
            "scFoundation pretraining MAE is not supported. "
            "Use downstream task Mixins (Classification, Perturbation, etc.)."
        )

    def get_cell_embedding(self, model: nn.Module, batch: dict) -> torch.Tensor:
        m = self._unwrap(model)
        return m.get_cell_embedding(batch["pretrain_gene_x"])

    def get_gene_embedding(self, model: nn.Module, batch: dict) -> torch.Tensor:
        m = self._unwrap(model)
        return m.get_gene_embedding(batch["pretrain_gene_x"])

    def _build_inference_dataset(self, adata):
        from ..registry.dataset import ScFoundationDataset
        return ScFoundationDataset(adata, self.cfg)

    def forward_embed_step(self, model, batch):
        if isinstance(batch, dict):
            m = self._unwrap(model) if hasattr(model, 'module') else model
            emb = m.get_cell_embedding(batch["pretrain_gene_x"])
            return emb, batch["cid"]
        else:
            values, pad, pos, cid = batch
            emb = model(values, pad, pos)
            return emb, cid

    # ------------------------------------------------------------------ #
    #  Gene-level helpers (standard interface)
    # ------------------------------------------------------------------ #
    def get_gene_token_ids(self, batch):
        """scFoundation uses a fixed gene order — token id ``i`` corresponds
        to ``gene_list[i]`` for ``i`` in ``[0, n_genes)``. The last 2 entries
        of ``pretrain_gene_x`` are auxiliary scalars (resolution, logtc), not
        genes, so they are excluded.
        """
        x = batch["pretrain_gene_x"]
        n_genes = x.shape[1] - 2  # last 2 = resolution + logtc
        return torch.arange(n_genes, device=x.device).unsqueeze(0).expand(x.shape[0], -1)

    def gene_vocab(self):
        """``{i: gene_name}`` from scFoundation's ``gene_list.csv`` (cached)."""
        if getattr(self, "_gene_vocab_cache", None) is None:
            import pandas as pd

            path = self.cfg.get("resources", {})["gene_list"]
            df = pd.read_csv(path, sep="\t", header=0)
            self._gene_vocab_cache = {i: g for i, g in enumerate(df["gene_name"])}
        return self._gene_vocab_cache

    # ------------------------------------------------------------------ #
    #  Paper-faithful head factory (TaskMixin v2)
    # ------------------------------------------------------------------ #
    def default_head(self, task_type: str, emb_dim: int, n_classes: int) -> nn.Module | None:
        """scFoundation paper-faithful head — ``LinearProbingClassifier``.

        Ref: Hao et al. 2024, ``Foundations/scFoundation/model/finetune_model.py``.
        Architecture identical to original (BatchNorm1d + Linear-ReLU-Linear,
        forward = norm → fc1). Sub-modules are named ``norm`` / ``fc1`` to
        match the paper exactly.
        """
        if task_type == "classification":
            return _ScFoundationCls(emb_dim, max(n_classes, 1))
        return None
