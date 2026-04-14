"""Training wrapper for scGPT — inherits from registry, overrides forward.

Changes from inference version:
- No ``torch.no_grad()`` on the training path (gradient flow for LoRA).
- ``forward(target_values=...)`` dispatches between GEP+MVC loss and
  embedding extraction.
- Uses ``masked_mse_loss`` (from original scGPT / scPEFT) for
  continuous gene expression prediction.
"""

import torch
import torch.nn.functional as F

from ...registry.models import ScGPTWrapper


def masked_mse_loss(pred, target, mask):
    """MSE loss computed only at masked positions.

    Identical to ``Foundations/scGPT/scgpt/loss.py::masked_mse_loss``.
    """
    mask = mask.float()
    loss = F.mse_loss(pred * mask, target * mask, reduction="sum")
    return loss / mask.sum()


class ScGPTTrainingWrapper(ScGPTWrapper):
    """scGPT wrapper for LoRA training.

    Inherits ``__init__`` from :class:`ScGPTWrapper` — same TransformerModel
    loading (including ExprDecoder, MVCDecoder, etc.).

    ``forward`` is overridden to support two modes:

    - ``target_values is not None`` → GEP + MVC loss (for ``compute_loss``)
    - ``target_values is None`` → CLS embedding extraction
    """

    MASK_VALUE = -1  # scGPT convention: masked expression positions = -1

    def forward(
        self,
        gene,
        masked_expr,
        src_key_padding_mask,
        target_values=None,
    ):
        if target_values is not None:
            return self._forward_train(
                gene, masked_expr, src_key_padding_mask, target_values
            )
        return self._forward_embedding(gene, masked_expr, src_key_padding_mask)

    # ------------------------------------------------------------------ #
    #  Training: GEP + MVC loss
    # ------------------------------------------------------------------ #
    def _forward_train(self, gene, masked_expr, src_key_padding_mask, target_values):
        """Multi-task forward: GEP (ExprDecoder) + MVC (MVCDecoder)."""
        output_dict = self.model(
            gene,
            masked_expr,
            src_key_padding_mask,
            MVC=hasattr(self.model, "mvc_decoder"),
        )

        masked_positions = masked_expr.eq(self.MASK_VALUE)

        # GEP loss (always present)
        loss = masked_mse_loss(
            output_dict["mlm_output"], target_values, masked_positions
        )

        # MVC loss (if MVCDecoder exists)
        if "mvc_output" in output_dict:
            loss = loss + masked_mse_loss(
                output_dict["mvc_output"], target_values, masked_positions
            )

        return loss

    # ------------------------------------------------------------------ #
    #  Embedding access (gradient flow preserved for downstream tasks)
    # ------------------------------------------------------------------ #
    def get_cell_embedding(self, gene, masked_expr, src_key_padding_mask):
        """CLS token embedding (B, D). Ref: scGPT Tutorial_Annotation."""
        hidden = self.model._encode(gene, masked_expr, src_key_padding_mask)
        return hidden[:, 0, :]  # CLS token

    def get_gene_embedding(self, gene, masked_expr, src_key_padding_mask):
        """Per-gene hidden states (B, S, D)."""
        return self.model._encode(gene, masked_expr, src_key_padding_mask)

    # ------------------------------------------------------------------ #
    #  Embedding extraction (post-training, same as inference wrapper)
    # ------------------------------------------------------------------ #
    def _forward_embedding(self, gene, expr, src_key_padding_mask):
        """Encoder-only CLS embedding. Returns (B, d_model)."""
        with torch.no_grad():
            embedding = self.model._encode(gene, expr, src_key_padding_mask)
            embedding = embedding[:, 0, :]  # CLS token
            return F.normalize(embedding, p=2, dim=1)
