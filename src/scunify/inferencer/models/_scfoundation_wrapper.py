import torch

from ...registry.models import ScFoundationWrapper


class ScFoundationInferenceWrapper(ScFoundationWrapper):
    """Inference wrapper — encoder-only forward with pooling."""

    def __init__(self, config):
        super().__init__(config)
        self.pool_type = config.get("model", {}).get("pool_type", "all")

    def forward(self, x, x_padding, position_gene_ids):
        x = torch.unsqueeze(x, 2)
        x = self.model.token_emb(x, output_weight=0)
        position_emb = self.model.pos_emb(position_gene_ids)
        x += position_emb
        geneemb = self.model.encoder(x, x_padding)
        if self.pool_type == "all":
            geneemb1 = geneemb[:, -1, :]
            geneemb2 = geneemb[:, -2, :]
            geneemb3, _ = torch.max(geneemb[:, :-2, :], dim=1)
            geneemb4 = torch.mean(geneemb[:, :-2, :], dim=1)
            geneembmerge = torch.cat(
                [geneemb1, geneemb2, geneemb3, geneemb4], axis=1
            )
        elif self.pool_type == "max":
            geneembmerge, _ = torch.max(geneemb, dim=1)
        else:
            raise ValueError("pool_type must be all or max")
        return geneembmerge
