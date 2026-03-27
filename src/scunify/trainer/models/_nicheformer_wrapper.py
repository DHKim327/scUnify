"""Training wrapper for Nicheformer — inherits from registry, overrides forward.

Changes from inference version:
- No ``torch.no_grad()`` on the training path (gradient flow for LoRA).
- ``forward(labels=...)`` dispatches between MLM loss and embedding extraction.
- Mask convention converted from HF (1=attend) to PyTorch (True=ignore)
  before calling the backbone's forward.
"""

import torch
import torch.nn.functional as F

from ...registry.models import NicheformerWrapper


class NicheformerTrainingWrapper(NicheformerWrapper):
    """Nicheformer wrapper for LoRA training.

    Inherits ``__init__`` from :class:`NicheformerWrapper` — same model
    loading, ``emb_layer`` setup.

    ``forward`` is overridden to support two modes:
    - ``labels is not None`` → MLM loss (for ``compute_loss``)
    - ``labels is None`` → embedding extraction (for ``extract_embeddings``)
    """

    def forward(self, input_ids, attention_mask, labels=None):
        if labels is not None:
            return self._forward_mlm(input_ids, attention_mask, labels)
        else:
            return self._forward_embedding(input_ids, attention_mask)

    # ------------------------------------------------------------------ #
    #  Training: MLM loss
    # ------------------------------------------------------------------ #
    def _forward_mlm(self, input_ids, attention_mask, labels):
        """Forward pass for MLM training. Returns scalar loss."""
        # Convert HF mask (1=attend, 0=pad) → PyTorch mask (True=ignore)
        padding_mask = ~attention_mask.bool()

        # Backbone forward → MaskedLMOutput(logits=..., hidden_states=...)
        outputs = self.model(input_ids, padding_mask)

        # MaskedLMOutput supports both attribute and dict access
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        return loss

    # ------------------------------------------------------------------ #
    #  Embedding extraction (post-training, same as inference)
    # ------------------------------------------------------------------ #
    def _forward_embedding(self, input_ids, attention_mask):
        """Forward pass for embedding extraction. Returns (B, d_model)."""
        with torch.no_grad():
            emb = self.model.get_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer=self.emb_layer,
                with_context=False,
            )
        return emb
