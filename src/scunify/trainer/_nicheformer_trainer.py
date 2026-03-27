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
        return inject_lora_to_model(model, "nicheformer", self.lora_cfg)

    def compute_loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """MLM loss — Nicheformer's classifier_head predicts masked tokens."""
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

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
