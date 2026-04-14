import torch
import torch.nn as nn


class NicheformerWrapper(nn.Module):
    """Base wrapper for Nicheformer — model loading only, no forward.

    Subclassed by inferencer and trainer wrappers which define their own forward.
    """

    def __init__(self, config):
        super().__init__()
        self.model = load(config)
        model_cfg = config.get("model", {})
        self.emb_layer = model_cfg.get("emb_layer", -1)


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
