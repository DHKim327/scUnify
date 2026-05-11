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
        return inject_lora_to_model(
            model, "uce", self.lora_cfg, self.freeze_cfg, self.is_backbone_param,
        )

    def compute_pretraining_loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """Binary expression prediction loss."""
        return model(
            batch_sentences=batch["batch_sentences"],
            mask=batch["mask"],
            target_genes=batch["target_genes"],
            target_labels=batch["target_labels"],
        )

    def get_cell_embedding(self, model: nn.Module, batch) -> torch.Tensor:
        """CLS token embedding (B, D). Handles both dict (training) and tuple (inference) batches."""
        m = self._unwrap(model)
        if isinstance(batch, dict):
            return m.get_cell_embedding(batch["batch_sentences"], batch["mask"])
        # Inference dataset collator returns tuple: (batch_sentences, mask, cid, ...)
        return m.get_cell_embedding(batch[0], batch[1])

    def get_gene_embedding(self, model: nn.Module, batch) -> torch.Tensor:
        """Gene-level embedding (B, S, D). Handles both dict and tuple batches."""
        m = self._unwrap(model)
        if isinstance(batch, dict):
            return m.get_gene_embedding(batch["batch_sentences"], batch["mask"])
        return m.get_gene_embedding(batch[0], batch[1])

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

    # ------------------------------------------------------------------ #
    #  Gene-level helpers (standard interface)
    # ------------------------------------------------------------------ #
    def get_gene_token_ids(self, batch):
        """UCE batch is a tuple ``(batch_sentences, mask, cid, ...)`` from the
        collator, or a dict during training. ``batch_sentences`` (B, pad_length)
        contains: ``[CLS, [chrom_open, protein_emb_indices..., chrom_close]*, PAD]``.
        Each gene token equals the row index in the species' protein-embedding
        table; chromosome / CLS / PAD tokens use special offsets.
        """
        if isinstance(batch, dict):
            return batch.get("batch_sentences", batch.get("input_ids"))
        return batch[0]

    def gene_vocab(self):
        """``{token_id: gene_symbol}`` from UCE's per-species protein-embedding
        dict (loaded from ``cfg.resources['protein_embeddings'][species]``).

        Token ID == enumeration index of the protein-embedding dict.
        Special tokens (CLS, chrom_open/close, PAD) are not in this map and
        will be NaN-padded by ``align_to_adata_var``.
        """
        if getattr(self, "_gene_vocab_cache", None) is None:
            import torch as _torch

            preproc = self.cfg.get("preprocessing", {}) or {}
            species = preproc.get("species", "human")
            res = self.cfg.get("resources", {})
            pe_path = res["protein_embeddings"][species]
            pe = _torch.load(pe_path)  # {gene_symbol: tensor}
            self._gene_vocab_cache = {i: sym for i, sym in enumerate(pe.keys())}
        return self._gene_vocab_cache

    # ------------------------------------------------------------------ #
    #  Paper-faithful head factory (TaskMixin v2)
    # ------------------------------------------------------------------ #
    def default_head(self, task_type: str, emb_dim: int, n_classes: int) -> nn.Module | None:
        """UCE classification head — scunify default.

        UCE (Rosen et al. 2024) is a zero-shot model; the original paper
        does not fine-tune for classification. We provide a single Linear
        head as the scunify default.
        """
        if task_type == "classification":
            return nn.Linear(emb_dim, max(n_classes, 1))
        return None
