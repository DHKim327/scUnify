"""scGPT LoRA trainer — HF PEFT (QKV unfused before injection).

GEP + MVC loss via TransformerModel's ExprDecoder and MVCDecoder.
"""

import torch
import torch.nn as nn

from .base._basetrainer import BaseTrainer
from .dataset._scgpt_dataset import ScGPTTrainingDataset
from .lora._injection import inject_lora_to_model
from .models._scgpt_wrapper import ScGPTTrainingWrapper


class ScGPTTrainer(BaseTrainer):
    """LoRA trainer for scGPT (GEP + MVC, fused QKV)."""

    def build_dataset(self, adata):
        return ScGPTTrainingDataset(adata, self.cfg)

    def build_model(self):
        return ScGPTTrainingWrapper(self.cfg)

    def inject_lora(self, model: nn.Module) -> nn.Module:
        return inject_lora_to_model(model, "scgpt", self.lora_cfg)

    def compute_loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """GEP + MVC loss on masked gene expression positions."""
        pad_token_id = batch["pad_token_id"]
        src_key_padding_mask = batch["gene"].eq(pad_token_id)
        return model(
            gene=batch["gene"],
            masked_expr=batch["masked_expr"],
            src_key_padding_mask=src_key_padding_mask,
            target_values=batch["expr"],
        )

    # ------------------------------------------------------------------ #
    #  Embedding extraction (distributed, via BaseTrainer)
    # ------------------------------------------------------------------ #
    def _build_inference_dataset(self, adata):
        """Inference dataset (no masking) for post-training embedding extraction."""
        from ..registry.dataset import ScGPTDataset

        return ScGPTDataset(adata, self.cfg)

    def forward_embed_step(self, model, batch):
        """Single-batch embedding. Inference dataset uses same batch format."""
        gene = batch["gene"]
        expr = batch["expr"]
        pad_token_id = batch["pad_token_id"]
        src_key_padding_mask = gene.eq(pad_token_id)
        emb = model(
            gene=gene,
            masked_expr=expr,
            src_key_padding_mask=src_key_padding_mask,
        )  # target_values=None → embedding mode
        return emb, batch["cid"]
