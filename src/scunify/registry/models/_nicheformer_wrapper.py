import torch
import torch.nn as nn


class NicheformerWrapper(nn.Module):
    """Wrapper for Nicheformer (HuggingFace) embedding extraction.

    Uses ``NicheformerForMaskedLM.get_embeddings()`` which:
      1. Embeds tokens + learnable positional encoding
      2. Passes through transformer layers up to *layer*
      3. Removes first 3 context tokens (species, assay, modality)
      4. Mean-pools over remaining sequence → (B, 512)
    """

    def __init__(self, config):
        super().__init__()
        self.model = load(config)
        inference_cfg = config.get("inference", {})
        self.emb_layer = inference_cfg.get("emb_layer", -1)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            emb = self.model.get_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer=self.emb_layer,
                with_context=False,
            )
        return emb  # (B, 512)


def load(config):
    """Load Nicheformer from HuggingFace (or local directory)."""
    from transformers import AutoModelForMaskedLM

    resources = config.get("resources", {})
    model_dir = resources.get("model_dir", "theislab/nicheformer")

    model = AutoModelForMaskedLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )
    model.eval()
    return model
