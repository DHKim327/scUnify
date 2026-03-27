import torch

from ...registry.models import NicheformerWrapper


class NicheformerInferenceWrapper(NicheformerWrapper):
    """Inference wrapper — embedding extraction via get_embeddings."""

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            emb = self.model.get_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer=self.emb_layer,
                with_context=False,
            )
        return emb  # (B, 512)
