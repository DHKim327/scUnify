"""Geneformer LoRA trainer — Phase 1 implementation.

Uses HF PEFT for LoRA injection (separate Q/K/V Linears).
MLM loss via BertForMaskedLM built-in head.
"""

import torch
import torch.nn as nn

from .base._basetrainer import BaseTrainer
from .dataset._geneformer_dataset import GeneformerTrainingDataset
from .lora._injection import inject_lora_to_model
from .models._geneformer_wrapper import GeneformerTrainingWrapper


class GeneformerTrainer(BaseTrainer):
    """LoRA trainer for Geneformer (BertForMaskedLM)."""

    def build_dataset(self, adata):
        return GeneformerTrainingDataset(adata, self.cfg)

    def build_model(self):
        return GeneformerTrainingWrapper(self.cfg)

    def inject_lora(self, model: nn.Module) -> nn.Module:
        return inject_lora_to_model(model, "geneformer", self.lora_cfg)

    def compute_pretraining_loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """MLM loss — BertForMaskedLM computes CrossEntropyLoss internally."""
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

    def get_cell_embedding(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """Mean-pooled hidden states (B, D). Ref: Geneformer — no CLS token."""
        m = self._unwrap(model)
        return m.get_cell_embedding(batch["input_ids"], batch["attention_mask"])

    def get_gene_embedding(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """Per-token hidden states (B, S, D)."""
        m = self._unwrap(model)
        return m.get_gene_embedding(batch["input_ids"], batch["attention_mask"])

    def _build_inference_dataset(self, adata):
        """Inference dataset (no MLM masking) for embedding extraction."""
        from ..registry.dataset import GeneformerDataset

        return GeneformerDataset(adata, self.cfg)

    def forward_embed_step(self, model, batch):
        """Single-batch embedding. labels=None → embedding mode."""
        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        emb = model(input_ids, attn_mask)
        return emb, batch["cid"]
