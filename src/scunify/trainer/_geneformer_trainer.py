"""Geneformer LoRA trainer — Phase 1 implementation.

Uses HF PEFT for LoRA injection (separate Q/K/V Linears).
MLM loss via BertForMaskedLM built-in head.
"""

import numpy as np
import torch
import torch.nn as nn

from .base._basetrainer import BaseTrainer
from .dataset._geneformer_dataset import GeneformerTrainingDataset
from .lora._injection import inject_lora_to_model
from .models._geneformer_wrapper import GeneformerTrainingWrapper


class GeneformerTrainer(BaseTrainer):
    """LoRA trainer for Geneformer (BertForMaskedLM)."""

    _uses_hf_peft = True

    def build_dataset(self, adata):
        return GeneformerTrainingDataset(adata, self.cfg)

    def build_model(self):
        return GeneformerTrainingWrapper(self.cfg)

    def inject_lora(self, model: nn.Module) -> nn.Module:
        return inject_lora_to_model(model, "geneformer", self.lora_cfg)

    def compute_loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """MLM loss — BertForMaskedLM computes CrossEntropyLoss internally."""
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

    def extract_embeddings(self, model: nn.Module, adata) -> np.ndarray:
        """Extract embeddings with LoRA-adapted model (no masking).

        Uses registry's GeneformerDataset (inference version, no MLM masking)
        but the training wrapper with labels=None → embedding mode.
        """
        from torch.utils.data import DataLoader

        from ..registry.dataset import GeneformerDataset

        model.eval()
        ds = GeneformerDataset(adata, self.cfg)
        inf_cfg = self.cfg.get("inference", {})
        bs = int(inf_cfg.get("batch_size", 64))
        dl = DataLoader(
            ds,
            batch_size=bs,
            collate_fn=ds.collator,
            shuffle=False,
            drop_last=False,
        )

        device = next(model.parameters()).device
        emb_chunks = []
        cid_chunks = []

        with torch.no_grad():
            for batch in dl:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                cid = batch["cid"]

                emb = model(input_ids, attn_mask)  # labels=None → embedding
                emb_chunks.append(emb.cpu())
                cid_chunks.append(cid)

        # Restore original cell order (SequentialSampler, but be safe)
        E = torch.cat(emb_chunks, dim=0)
        C = torch.cat(cid_chunks, dim=0).long()
        order = torch.argsort(C, stable=True)
        E = E[order]

        return E.float().numpy()
