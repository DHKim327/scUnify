import torch

from ...registry.models import ScGPTWrapper


class ScGPTInferenceWrapper(ScGPTWrapper):
    """Inference wrapper — CLS embedding with L2 normalization."""

    def forward(self, input_gene_ids, expr, src_key_padding_mask):
        embedding = self.model._encode(
            input_gene_ids,
            expr,
            src_key_padding_mask=src_key_padding_mask,
        )
        embedding = embedding[:, 0, :]  # CLS token
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding
