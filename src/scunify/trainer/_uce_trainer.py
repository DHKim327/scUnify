"""UCE LoRA trainer — HF PEFT (QKV unfused before injection).

Binary expression prediction loss via TransformerModel's binary_decoder.
"""

import torch
import torch.nn as nn

from .base._basetrainer import BaseTrainer
from .dataset._uce_dataset import UCETrainingDataset
from .lora._injection import inject_lora_to_model
from .models._uce_wrapper import UCETrainingWrapper


class UCETrainer(BaseTrainer):
    """LoRA trainer for UCE (binary expression prediction, fused QKV)."""

    def build_dataset(self, adata):
        return UCETrainingDataset(adata, self.cfg)

    def build_model(self):
        return UCETrainingWrapper(self.cfg)

    def inject_lora(self, model: nn.Module) -> nn.Module:
        return inject_lora_to_model(model, "uce", self.lora_cfg)

    def compute_pretraining_loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """Binary expression prediction loss."""
        return model(
            batch_sentences=batch["batch_sentences"],
            mask=batch["mask"],
            target_genes=batch["target_genes"],
            target_labels=batch["target_labels"],
        )

    def get_cell_embedding(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """CLS token embedding (B, D)."""
        m = self._unwrap(model)
        return m.get_cell_embedding(batch["batch_sentences"], batch["mask"])

    def get_gene_embedding(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """Gene-level embedding (B, S, D)."""
        m = self._unwrap(model)
        return m.get_gene_embedding(batch["batch_sentences"], batch["mask"])

    # ------------------------------------------------------------------ #
    #  Embedding extraction (distributed, via BaseTrainer)
    # ------------------------------------------------------------------ #
    def _build_inference_dataset(self, adata):
        """Inference dataset (no masking) for post-training embedding extraction."""
        from ..registry.dataset import UCEDataset

        return UCEDataset(adata, self.cfg)

    def forward_embed_step(self, model, batch):
        """Single-batch embedding. Inference dataset returns tuple via collator."""
        batch_sentences, mask, cid, _ = batch
        emb = model(batch_sentences, mask)  # target_labels=None → embedding mode
        return emb, cid
