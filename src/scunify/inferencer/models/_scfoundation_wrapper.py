import torch

from ...registry.models import ScFoundationWrapper


def _gather_data(data, labels, pad_token_id):
    """Batch-wise gatherData — pads variable-length nonzero genes to max_num.
    Ref: scFoundation load.py gatherData / scPEFT annotation/load.py"""
    value_nums = labels.sum(1)
    max_num = int(value_nums.max().item())

    fake_data = torch.full((data.shape[0], max_num), pad_token_id,
                           dtype=data.dtype, device=data.device)
    data_padded = torch.cat([data, fake_data], dim=1)

    fake_label = torch.ones((labels.shape[0], max_num),
                            dtype=labels.dtype, device=labels.device)
    none_labels = (labels == 0)
    labels_copy = labels.clone().float()
    labels_copy[none_labels] = -float('inf')

    F = labels.shape[1]
    tmp_data = torch.tensor(
        [(i + 1) * 20000 for i in range(F, 0, -1)],
        dtype=labels_copy.dtype, device=labels_copy.device,
    )
    labels_copy = labels_copy + tmp_data
    labels_padded = torch.cat([labels_copy, fake_label], dim=1)

    topk_indices = labels_padded.topk(max_num, dim=1).indices
    gathered_data = torch.gather(data_padded, 1, topk_indices)
    padding_labels = (gathered_data == pad_token_id)

    return gathered_data, padding_labels


class ScFoundationInferenceWrapper(ScFoundationWrapper):
    """Inference wrapper — encoder-only forward with batch-wise gatherData."""

    def __init__(self, config):
        super().__init__(config)
        self.pool_type = config.get("model", {}).get("pool_type", "all")
        model_cfg = config.get("model", {})
        self.pad_token_id = config.model_param[model_cfg.get("version", "cell")]["mae_autobin"]["pad_token_id"]
        self.seq_len = config.model_param[model_cfg.get("version", "cell")]["mae_autobin"]["seq_len"]

    def forward(self, pretrain_gene_x):
        """Batch-wise forward: (B, 19266) → gatherData → encoder → pooling.

        Args:
            pretrain_gene_x: (B, 19266) — fixed-length from dataset
        Returns:
            (B, D) cell embeddings
        """
        # value_labels: nonzero mask
        value_labels = (pretrain_gene_x > 0).float()  # (B, 19266)

        # data_gene_ids: [0, 1, ..., 19265] repeated for batch
        B = pretrain_gene_x.size(0)
        data_gene_ids = torch.arange(
            pretrain_gene_x.size(1), device=pretrain_gene_x.device
        ).unsqueeze(0).expand(B, -1)  # (B, 19266)

        # Batch-wise gatherData — pads to max nonzero in batch
        x, x_padding = _gather_data(pretrain_gene_x, value_labels, self.pad_token_id)
        position_gene_ids, _ = _gather_data(
            data_gene_ids.float(), value_labels, self.pad_token_id
        )
        position_gene_ids = position_gene_ids.long()
        position_gene_ids[x_padding] = self.seq_len

        # Encoder forward
        with torch.no_grad():
            x_in = torch.unsqueeze(x, 2)
            x_emb = self.model.token_emb(x_in, output_weight=0)
            position_emb = self.model.pos_emb(position_gene_ids)
            x_emb += position_emb
            geneemb = self.model.encoder(x_emb, padding_mask=x_padding)

            if self.pool_type == "all":
                g1 = geneemb[:, -1, :]
                g2 = geneemb[:, -2, :]
                g3, _ = torch.max(geneemb[:, :-2, :], dim=1)
                g4 = torch.mean(geneemb[:, :-2, :], dim=1)
                return torch.cat([g1, g2, g3, g4], dim=1)
            elif self.pool_type == "max":
                result, _ = torch.max(geneemb, dim=1)
                return result
            else:
                raise ValueError(f"pool_type must be 'all' or 'max', got {self.pool_type}")
