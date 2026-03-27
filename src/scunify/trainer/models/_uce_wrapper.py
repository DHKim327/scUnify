"""Training wrapper for UCE — inherits from registry, overrides forward.

Changes from inference version:
- No ``torch.no_grad()`` on the training path (gradient flow for LoRA).
- ``forward(target_labels=...)`` dispatches between binary expression
  prediction loss and embedding extraction.
- Uses the model's ``binary_decoder`` + ``gene_embedding_layer`` for loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...registry.models import UCEWrapper


class UCETrainingWrapper(UCEWrapper):
    """UCE wrapper for LoRA training.

    Inherits ``__init__`` from :class:`UCEWrapper` — same TransformerModel
    loading (including binary_decoder, gene_embedding_layer, pe_embedding).

    ``forward`` is overridden to support two modes:

    - ``target_labels is not None`` → binary expression prediction + BCE loss
    - ``target_labels is None`` → CLS embedding extraction
    """

    def __init__(self, config):
        super().__init__(config)
        # Alias for LoRA injection compatibility (expects model.model)
        self.model = self.encoder

    def forward(
        self,
        batch_sentences,
        mask,
        target_genes=None,
        target_labels=None,
    ):
        # Common: look up ESM2 embeddings and normalize
        src = batch_sentences.permute(1, 0)  # [seq_len, batch]
        src = self.pe_embedding(src.long())  # [seq_len, batch, 5120]
        src = nn.functional.normalize(src, dim=2)

        if target_labels is not None:
            return self._forward_train(src, mask, target_genes, target_labels)
        return self._forward_embedding(src, mask)

    # ------------------------------------------------------------------ #
    #  Training: binary expression prediction loss
    # ------------------------------------------------------------------ #
    def _forward_train(self, src, mask, target_genes, target_labels):
        """Binary expression prediction: predict if target genes are expressed.

        Uses model's binary_decoder(cell_emb || gene_emb) → BCE loss.
        """
        # Forward through TransformerModel → (gene_output, cell_emb)
        _, cell_emb = self.encoder(src, mask=mask)
        # cell_emb: (B, 1280) — L2-normalized CLS token

        # Look up target gene ESM2 embeddings
        target_pe = self.pe_embedding(target_genes.long())  # (B, N_target, 5120)

        # Project through gene_embedding_layer
        gene_emb = self.encoder.gene_embedding_layer(target_pe)  # (B, N_target, 1280)

        # Expand cell_emb and concatenate with gene_emb
        B, N, D = gene_emb.shape
        cell_expanded = cell_emb.unsqueeze(1).expand(B, N, -1)  # (B, N, 1280)
        combined = torch.cat([cell_expanded, gene_emb], dim=-1)  # (B, N, 2560)

        # Binary prediction
        pred = self.encoder.binary_decoder(combined).squeeze(-1)  # (B, N)

        return F.binary_cross_entropy_with_logits(pred, target_labels)

    # ------------------------------------------------------------------ #
    #  Embedding extraction (post-training, same as inference wrapper)
    # ------------------------------------------------------------------ #
    def _forward_embedding(self, src, mask):
        """CLS embedding extraction. Returns (B, output_dim)."""
        with torch.no_grad():
            _, embedding = self.encoder(src, mask=mask)
            return embedding
