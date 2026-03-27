import torch

from ...registry.models import GeneformerWrapper


class GeneformerInferenceWrapper(GeneformerWrapper):
    """Inference wrapper — hidden state extraction from BertForMaskedLM."""

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        hidden = outputs.hidden_states[self._layer_to_quant]

        if self.emb_mode == "cls":
            return hidden[:, 0, :]  # (batch, hidden_size)
        else:
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
            return sum_hidden / count  # (batch, hidden_size)
