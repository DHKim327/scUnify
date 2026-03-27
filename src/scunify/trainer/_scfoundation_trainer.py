"""scFoundation LoRA trainer — HF PEFT (QKV unfused before injection).

MAE reconstruction loss via MaeAutobin's full encoder + decoder pipeline.
"""

import torch
import torch.nn as nn

from .base._basetrainer import BaseTrainer
from .dataset._scfoundation_dataset import ScFoundationTrainingDataset
from .lora._injection import inject_lora_to_model
from .models._scfoundation_wrapper import ScFoundationTrainingWrapper


class ScFoundationTrainer(BaseTrainer):
    """LoRA trainer for scFoundation (MAE reconstruction, fused QKV)."""

    def build_dataset(self, adata):
        return ScFoundationTrainingDataset(adata, self.cfg)

    def build_model(self):
        return ScFoundationTrainingWrapper(self.cfg)

    def inject_lora(self, model: nn.Module) -> nn.Module:
        return inject_lora_to_model(model, "scfoundation", self.lora_cfg)

    def compute_loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """MAE reconstruction loss — MSE on masked gene positions."""
        return model(
            x=batch["x"],
            padding_label=batch["padding_label"],
            encoder_position_gene_ids=batch["encoder_position_gene_ids"],
            encoder_labels=batch["encoder_labels"],
            decoder_data=batch["decoder_data"],
            decoder_position_gene_ids=batch["decoder_position_gene_ids"],
            decoder_data_padding_labels=batch["decoder_data_padding_labels"],
            mask_labels=batch["mask_labels"],
            targets=batch["targets"],
        )

    # ------------------------------------------------------------------ #
    #  Embedding extraction (distributed, via BaseTrainer)
    # ------------------------------------------------------------------ #
    def _build_inference_dataset(self, adata):
        """Inference dataset (no masking) for post-training embedding extraction."""
        from ..registry.dataset import ScFoundationDataset

        return ScFoundationDataset(adata, self.cfg)

    def forward_embed_step(self, model, batch):
        """Single-batch embedding. Inference dataset returns tuple."""
        values, pad, pos, cid = batch
        emb = model(values, pad, pos)  # targets=None → embedding mode
        return emb, cid
