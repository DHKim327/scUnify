"""Training wrapper for UCE — inherits from registry, overrides forward.

Changes from inference version:
- No ``torch.no_grad()`` on the training path (gradient flow for LoRA).
- ``forward(target_labels=...)`` dispatches between binary expression
  prediction loss and embedding extraction.
- Uses the model's ``binary_decoder`` + ``gene_embedding_layer`` for loss.

Note: After PEFT injection, ``self.model`` becomes PeftModel wrapping the
TransformerModel. All forward calls MUST go through ``self.model`` to
ensure LoRA adapters are active. Sub-modules (``gene_embedding_layer``,
``binary_decoder``, etc.) are accessed via PEFT's __getattr__ forwarding.
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
        # After inject_lora(), self.model becomes PeftModel.
        # All forward calls must use self.model, not self.encoder.
        self.model = self.encoder

    def _prepare_src(self, batch_sentences):
        """Common input preparation: ESM2 lookup + normalize."""
        src = batch_sentences.permute(1, 0)  # [seq_len, batch]
        src = self.pe_embedding(src.long())  # [seq_len, batch, 5120]
        return nn.functional.normalize(src, dim=2)

    # ------------------------------------------------------------------ #
    #  Embedding access (gradient flow preserved for downstream tasks)
    # ------------------------------------------------------------------ #
    def get_cell_embedding(self, batch_sentences, mask):
        """CLS token embedding (B, D). Gradient flow preserved.
        Routes through self.model (PeftModel) for LoRA activation."""
        src = self._prepare_src(batch_sentences)
        _, cell_emb = self.model(src, mask=mask)
        return cell_emb

    def get_gene_embedding(self, batch_sentences, mask):
        """Gene-level output (B, S, D). Gradient flow preserved."""
        src = self._prepare_src(batch_sentences)
        gene_output, _ = self.model(src, mask=mask)
        return gene_output.permute(1, 0, 2)  # (S, B, D) → (B, S, D)

    # ------------------------------------------------------------------ #
    #  Forward dispatch
    # ------------------------------------------------------------------ #
    def forward(
        self,
        batch_sentences,
        mask,
        target_genes=None,
        target_labels=None,
    ):
        src = self._prepare_src(batch_sentences)

        if target_labels is not None:
            return self._forward_train(src, mask, target_genes, target_labels)
        return self._forward_embedding(src, mask)

    # ------------------------------------------------------------------ #
    #  Training: binary expression prediction loss
    # ------------------------------------------------------------------ #
    def _forward_train(self, src, mask, target_genes, target_labels):
        """Binary expression prediction: predict if target genes are expressed.

        Uses model's binary_decoder(cell_emb || gene_emb) → BCE loss.
        Routes through self.model (PeftModel) for LoRA activation.
        """
        # Forward through PeftModel → TransformerModel → (gene_output, cell_emb)
        _, cell_emb = self.model(src, mask=mask)

        # Sub-modules accessed via PeftModel's __getattr__ forwarding
        target_pe = self.pe_embedding(target_genes.long())
        gene_emb = self.model.gene_embedding_layer(target_pe)

        B, N, D = gene_emb.shape
        cell_expanded = cell_emb.unsqueeze(1).expand(B, N, -1)
        combined = torch.cat([cell_expanded, gene_emb], dim=-1)

        pred = self.model.binary_decoder(combined).squeeze(-1)

        return F.binary_cross_entropy_with_logits(pred, target_labels)

    # ------------------------------------------------------------------ #
    #  Embedding extraction (post-training, same as inference wrapper)
    # ------------------------------------------------------------------ #
    def _forward_embedding(self, src, mask):
        """CLS embedding extraction. Returns (B, output_dim)."""
        with torch.no_grad():
            _, embedding = self.model(src, mask=mask)
            return embedding
