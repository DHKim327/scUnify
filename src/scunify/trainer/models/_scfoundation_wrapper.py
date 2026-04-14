"""Training wrapper for scFoundation — inherits from registry, overrides forward.

Changes from inference version:
- No ``torch.no_grad()`` on the training path (gradient flow for LoRA).
- ``forward(targets=...)`` dispatches between MAE loss and embedding extraction.
- Embedding extraction uses same encoder-only pooling as inference wrapper.
"""

import torch
import torch.nn.functional as F

from ...registry.models import ScFoundationWrapper


class ScFoundationTrainingWrapper(ScFoundationWrapper):
    """scFoundation wrapper for LoRA training.

    Inherits ``__init__`` from :class:`ScFoundationWrapper` — same full
    MaeAutobin model loading.

    ``forward`` is overridden to support two modes:

    - ``targets is not None`` → full MAE forward (encoder+decoder) + MSE loss
    - ``targets is None`` → encoder-only embedding extraction
    """

    def __init__(self, config):
        super().__init__(config)
        self.pool_type = config.get("model", {}).get("pool_type", "all")

    # ------------------------------------------------------------------ #
    #  Embedding access (gradient flow preserved for downstream tasks)
    # ------------------------------------------------------------------ #
    def get_cell_embedding(self, x, padding_label, position_gene_ids, encoder_labels=None):
        """Encoder output mean-pooled (B, D). Gradient flow preserved.
        Ref: Hao et al., Nature Methods 2024."""
        x_in = torch.unsqueeze(x, 2)
        x_emb = self.model.token_emb(x_in, output_weight=0)
        position_emb = self.model.pos_emb(position_gene_ids)
        x_emb += position_emb
        geneemb = self.model.encoder(x_emb, padding_mask=padding_label)

        # Ref: scFoundation get_embedding.py — pool_type choices: 'all', 'max'
        if self.pool_type == "all":
            g1 = geneemb[:, -1, :]
            g2 = geneemb[:, -2, :]
            g3, _ = torch.max(geneemb[:, :-2, :], dim=1)
            g4 = torch.mean(geneemb[:, :-2, :], dim=1)
            return torch.cat([g1, g2, g3, g4], dim=1)
        elif self.pool_type == "max":
            result, _ = torch.max(geneemb, dim=1)
            return result
        else:
            raise ValueError(f"pool_type must be 'all' or 'max', got {self.pool_type}")

    def get_gene_embedding(self, x, padding_label, position_gene_ids, encoder_labels=None):
        """Encoder per-gene output (B, S, D). Gradient flow preserved."""
        x_in = torch.unsqueeze(x, 2)
        x_emb = self.model.token_emb(x_in, output_weight=0)
        position_emb = self.model.pos_emb(position_gene_ids)
        x_emb += position_emb
        return self.model.encoder(x_emb, padding_mask=padding_label)

    # ------------------------------------------------------------------ #
    #  Forward dispatch
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x,
        padding_label,
        encoder_position_gene_ids,
        encoder_labels=None,
        decoder_data=None,
        decoder_position_gene_ids=None,
        decoder_data_padding_labels=None,
        mask_labels=None,
        targets=None,
    ):
        if targets is not None:
            return self._forward_mae(
                x,
                padding_label,
                encoder_position_gene_ids,
                encoder_labels,
                decoder_data,
                decoder_position_gene_ids,
                decoder_data_padding_labels,
                mask_labels,
                targets,
            )
        return self._forward_embedding(x, padding_label, encoder_position_gene_ids)

    # ------------------------------------------------------------------ #
    #  Training: MAE reconstruction loss
    # ------------------------------------------------------------------ #
    def _forward_mae(
        self,
        x,
        padding_label,
        encoder_position_gene_ids,
        encoder_labels,
        decoder_data,
        decoder_position_gene_ids,
        decoder_data_padding_labels,
        mask_labels,
        targets,
    ):
        """Full MAE encoder+decoder forward → MSE loss on masked positions."""
        pred = self.model(
            x=x,
            padding_label=padding_label,
            encoder_position_gene_ids=encoder_position_gene_ids,
            encoder_labels=encoder_labels,
            decoder_data=decoder_data,
            mask_gene_name=False,
            mask_labels=mask_labels,
            decoder_position_gene_ids=decoder_position_gene_ids,
            decoder_data_padding_labels=decoder_data_padding_labels,
        )  # (B, N_dec)

        # MSE loss on masked positions only
        return F.mse_loss(pred[mask_labels], targets[mask_labels])

    # ------------------------------------------------------------------ #
    #  Embedding extraction (post-training, same as inference wrapper)
    # ------------------------------------------------------------------ #
    def _forward_embedding(self, x, padding_label, position_gene_ids):
        """Encoder-only forward with pooling. Returns (B, D) embeddings."""
        with torch.no_grad():
            x_in = torch.unsqueeze(x, 2)
            x_emb = self.model.token_emb(x_in, output_weight=0)
            position_emb = self.model.pos_emb(position_gene_ids)
            x_emb += position_emb
            geneemb = self.model.encoder(x_emb, padding_mask=padding_label)

            if self.pool_type == "all":
                g1 = geneemb[:, -1, :]
                g2 = geneemb[:, -2, :]
                g3, _ = torch.max(geneemb[:, :-2, :], dim=1)
                g4 = torch.mean(geneemb[:, :-2, :], dim=1)
                return torch.cat([g1, g2, g3, g4], dim=1)
            elif self.pool_type == "max":
                result, _ = torch.max(geneemb, dim=1)
                return result
            else:
                raise ValueError(f"pool_type must be all or max, got {self.pool_type}")
