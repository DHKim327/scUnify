"""Training wrapper for Geneformer — inherits from registry, overrides forward.

Changes from inference version:
- No ``torch.no_grad()`` (gradient flow needed for LoRA)
- ``forward(labels=...)`` dispatches between MLM loss and embedding extraction
"""

import torch

from ...registry.models import GeneformerWrapper


class GeneformerTrainingWrapper(GeneformerWrapper):
    """Geneformer wrapper for LoRA training.

    Inherits ``__init__`` from :class:`GeneformerWrapper` — same model
    loading, ``emb_layer``, ``_layer_to_quant`` setup.

    ``forward`` is overridden to support two modes:
    - ``labels is not None`` → MLM loss (for ``compute_loss``)
    - ``labels is None`` → embedding extraction (for ``extract_embeddings``)
    """

    # ------------------------------------------------------------------ #
    #  Embedding access (gradient flow preserved for downstream tasks)
    # ------------------------------------------------------------------ #
    def get_cell_embedding(self, input_ids, attention_mask):
        """Mean-pooled hidden states (B, D). Geneformer has no CLS token.
        Ref: Theodoris et al., Nature 2023."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[self._layer_to_quant]
        # Mean pooling excluding CLS and EOS positions
        mask = attention_mask.clone()
        mask[:, 0] = 0  # exclude CLS position
        lengths = attention_mask.sum(dim=1)
        for i in range(len(lengths)):
            eos_pos = lengths[i] - 1
            if eos_pos > 0:
                mask[i, eos_pos] = 0
        mask_expanded = mask.unsqueeze(-1).float()
        sum_hidden = (hidden * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1)
        return sum_hidden / count

    def get_gene_embedding(self, input_ids, attention_mask):
        """Per-token hidden states (B, S, D)."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return outputs.hidden_states[self._layer_to_quant]

    # ------------------------------------------------------------------ #
    #  Forward dispatch
    # ------------------------------------------------------------------ #
    def forward(self, input_ids, attention_mask, labels=None):
        if labels is not None:
            # Training mode: MLM loss via BertForMaskedLM head
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            return outputs.loss
        else:
            # Embedding extraction mode (post-training)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[self._layer_to_quant]

            if self.emb_mode == "cls":
                return hidden[:, 0, :]
            else:
                # Mean pooling (same as inference wrapper)
                mask = attention_mask.clone()
                mask[:, 0] = 0
                lengths = attention_mask.sum(dim=1)
                for i in range(len(lengths)):
                    eos_pos = lengths[i] - 1
                    if eos_pos > 0:
                        mask[i, eos_pos] = 0
                mask_expanded = mask.unsqueeze(-1).float()
                sum_hidden = (hidden * mask_expanded).sum(dim=1)
                count = mask_expanded.sum(dim=1).clamp(min=1)
                return sum_hidden / count
