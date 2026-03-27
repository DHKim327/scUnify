import torch
import torch.nn as nn


class GeneformerWrapper(nn.Module):
    """Base wrapper for Geneformer V2 — model loading only, no forward.

    Subclassed by inferencer and trainer wrappers which define their own forward.
    """

    def __init__(self, config):
        super().__init__()
        self.model = load(config)
        inference_cfg = config.get("inference", {})
        self.emb_layer = inference_cfg.get("emb_layer", -1)
        self.emb_mode = inference_cfg.get("emb_mode", "cls")

        num_layers = self.model.config.num_hidden_layers
        self._layer_to_quant = num_layers + self.emb_layer


def load(config):
    """Load Geneformer model with output_hidden_states=True."""
    from transformers import BertForMaskedLM

    inference_cfg = config.get("inference", {})
    variant = inference_cfg.get("model_variant", "V2-104M")
    resources = config.get("resources", {})
    model_dir = resources["model_dirs"][variant]
    model = BertForMaskedLM.from_pretrained(
        model_dir,
        output_hidden_states=True,
        output_attentions=False,
    )
    model.eval()
    return model
