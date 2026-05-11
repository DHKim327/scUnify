"""Nicheformer LoRA trainer — HF PEFT (QKV unfused before injection).

MLM loss via Nicheformer's built-in ``classifier_head``.
"""

from pathlib import Path

import torch
import torch.nn as nn

from .base._basetrainer import BaseTrainer
from .dataset._nicheformer_dataset import NicheformerTrainingDataset
from .lora._injection import inject_lora_to_model
from .models._nicheformer_wrapper import NicheformerTrainingWrapper


class NicheformerTrainer(BaseTrainer):
    """LoRA trainer for Nicheformer (fused QKV, unfused before PEFT)."""

    def build_dataset(self, adata):
        return NicheformerTrainingDataset(adata, self.cfg)

    def build_model(self):
        return NicheformerTrainingWrapper(self.cfg)

    def inject_lora(self, model: nn.Module) -> nn.Module:
        return inject_lora_to_model(
            model, "nicheformer", self.lora_cfg, self.freeze_cfg, self.is_backbone_param,
        )

    def compute_pretraining_loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """MLM loss — Nicheformer's classifier_head predicts masked tokens."""
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

    def get_cell_embedding(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """Mean-pooled embedding (B, D), skip 3 contextual tokens.
        Ref: Schaar et al., Nature Methods 2025."""
        m = self._unwrap(model)
        return m.get_cell_embedding(batch["input_ids"], batch["attention_mask"])

    def get_gene_embedding(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """Per-gene hidden states (B, S, D)."""
        m = self._unwrap(model)
        return m.get_gene_embedding(batch["input_ids"], batch["attention_mask"])

    def _build_inference_dataset(self, adata):
        """Inference dataset (no MLM masking) for embedding extraction."""
        from ..registry.dataset import NicheformerDataset

        return NicheformerDataset(adata, self.cfg)

    def forward_embed_step(self, model, batch):
        """Single-batch embedding. labels=None → embedding mode."""
        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        emb = model(input_ids, attn_mask)
        return emb, batch["cid"]

    # ------------------------------------------------------------------ #
    #  Gene-level helpers (standard interface)
    # ------------------------------------------------------------------ #
    #  Nicheformer's gene tokens are simply ``adata.var index + AUX_TOKENS``
    #  (the first 3 tokens of every sequence are species/assay/modality
    #  context tokens). Vocab depends on the dataset (the ref-gene list used
    #  during preprocessing), so we override ``align_to_adata_var`` directly
    #  rather than maintaining a global token→symbol dict.

    _AUX_TOKENS = 30

    def get_gene_token_ids(self, batch):
        """Raw input_ids (B, context_length). First 3 tokens are species/
        assay/modality context; the rest are ``var_idx + AUX_TOKENS``."""
        return batch["input_ids"]

    def gene_vocab(self):
        raise NotImplementedError(
            "Nicheformer's vocab is dataset-dependent (token_id - AUX_TOKENS = "
            "adata.var index). Use ``trainer.align_to_adata_var(gene_emb, batch, adata)`` "
            "directly — it does the offset lookup without needing a global dict."
        )

    def align_to_adata_var(self, gene_emb, batch, adata):
        """Override: token_id - AUX_TOKENS = adata.var row index."""
        import numpy as np

        token_ids = batch["input_ids"].detach().cpu().numpy()  # (B, S)
        emb_np = gene_emb.detach().cpu().numpy()               # (B, S, D)
        B, S, D = emb_np.shape
        n_var = adata.shape[1]
        out = np.full((B, n_var, D), np.nan, dtype=np.float32)
        for b in range(B):
            for s in range(S):
                v = int(token_ids[b, s]) - self._AUX_TOKENS
                if 0 <= v < n_var:
                    out[b, v] = emb_np[b, s]
        return out

    # ------------------------------------------------------------------ #
    #  Paper-faithful head factory (TaskMixin v2)
    # ------------------------------------------------------------------ #
    def default_head(self, task_type: str, emb_dim: int, n_classes: int) -> nn.Module | None:
        """Nicheformer paper-faithful head.

        Ref: Schaar et al., Nature Methods 2025 — ``FineTuningModel``,
        ``_fine_tune_model.py:86``. Both classification and regression use
        the same ``Linear(dim_model, out_dim, bias=False)`` head (output
        dim differs)::

            niche_classification :  Linear(dim_model, n_classes, bias=False)
            density_regression   :  Linear(dim_model, 1, bias=False)
            niche_regression     :  Linear(dim_model, dim_prediction, bias=False)

        ``extractor=False`` is the paper default — no pooler/Tanh stack.
        For regression, yaml ``task_param.n_classes`` carries the output
        dim (= 1 for density / = n_cell_types for niche composition).
        """
        if task_type in ("classification", "regression"):
            return nn.Linear(emb_dim, max(n_classes, 1), bias=False)
        return None
