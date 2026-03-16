import torch
import torch.nn as nn


class GeneformerWrapper(nn.Module):
    """Wrapper for Geneformer V2 (BertForMaskedLM) embedding extraction."""

    def __init__(self, config):
        super().__init__()
        self.model = load(config)
        inference_cfg = config.get("inference", {})
        self.emb_layer = inference_cfg.get("emb_layer", -1)
        self.emb_mode = inference_cfg.get("emb_mode", "cls")

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # hidden_states: tuple of (batch, seq_len, hidden_size) per layer
        # index -1 = last layer, -2 = second-to-last (Geneformer default uses emb_layer=-1 for 2nd-to-last)
        hidden = outputs.hidden_states[self.emb_layer]

        if self.emb_mode == "cls":
            # CLS token is at position 0
            return hidden[:, 0, :]  # (batch, hidden_size)
        else:
            # Mean pooling over gene tokens (exclude CLS at 0 and EOS at end)
            # Use attention_mask to exclude padding
            mask = attention_mask.clone()
            mask[:, 0] = 0  # exclude CLS
            # Find EOS position per sample and exclude
            lengths = attention_mask.sum(dim=1)  # actual sequence lengths
            for i in range(len(lengths)):
                eos_pos = lengths[i] - 1
                if eos_pos > 0:
                    mask[i, eos_pos] = 0

            mask_expanded = mask.unsqueeze(-1).float()
            sum_hidden = (hidden * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            return sum_hidden / count  # (batch, hidden_size)


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
