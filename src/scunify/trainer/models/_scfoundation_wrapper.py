"""Training wrapper for scFoundation — downstream tasks with batch_size > 1.

All methods accept fixed-length (B, 19266) input and perform batch-wise
gatherData internally. No pretraining MAE support (removed).
"""

import torch
import torch.nn.functional as F

from ...registry.models import ScFoundationWrapper


def _gather_data(data, labels, pad_token_id):
    """Batch-wise gatherData — pads variable-length nonzero genes to max_num.
    Ref: scFoundation load.py / scPEFT annotation/load.py"""
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

    F_dim = labels.shape[1]
    tmp_data = torch.tensor(
        [(i + 1) * 20000 for i in range(F_dim, 0, -1)],
        dtype=labels_copy.dtype, device=labels_copy.device,
    )
    labels_copy = labels_copy + tmp_data
    labels_padded = torch.cat([labels_copy, fake_label], dim=1)

    topk_indices = labels_padded.topk(max_num, dim=1).indices
    gathered_data = torch.gather(data_padded, 1, topk_indices)
    padding_labels = (gathered_data == pad_token_id)

    return gathered_data, padding_labels


class ScFoundationTrainingWrapper(ScFoundationWrapper):
    """scFoundation wrapper for downstream LoRA training.

    All methods accept batch["pretrain_gene_x"] (B, 19266) and perform
    batch-wise gatherData internally. Supports batch_size > 1.
    """

    def __init__(self, config):
        super().__init__(config)
        self.pool_type = config.get("model", {}).get("pool_type", "all")
        model_cfg = config.get("model", {})
        version = model_cfg.get("version", "cell")
        mae_cfg = config.model_param[version]["mae_autobin"]
        self.pad_token_id = mae_cfg["pad_token_id"]
        self.seq_len = mae_cfg["seq_len"]

    def _prepare_batch(self, pretrain_gene_x):
        """Convert fixed-length (B, 19266) → encoder inputs via batch-wise gatherData."""
        value_labels = (pretrain_gene_x > 0).float()
        B = pretrain_gene_x.size(0)
        data_gene_ids = torch.arange(
            pretrain_gene_x.size(1), device=pretrain_gene_x.device
        ).unsqueeze(0).expand(B, -1).float()

        x, x_padding = _gather_data(pretrain_gene_x, value_labels, self.pad_token_id)
        pos_ids, _ = _gather_data(data_gene_ids, value_labels, self.pad_token_id)
        pos_ids = pos_ids.long()
        pos_ids[x_padding] = self.seq_len

        return x, x_padding, pos_ids

    # ------------------------------------------------------------------ #
    #  Embedding access (gradient flow preserved, batch_size > 1)
    # ------------------------------------------------------------------ #
    def get_cell_embedding(self, pretrain_gene_x):
        """Encoder output pooled (B, D). Gradient flow preserved."""
        x, x_padding, pos_ids = self._prepare_batch(pretrain_gene_x)
        x_in = torch.unsqueeze(x, 2)
        x_emb = self.model.token_emb(x_in, output_weight=0)
        position_emb = self.model.pos_emb(pos_ids)
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

    def get_gene_embedding(self, pretrain_gene_x):
        """Encoder per-gene output (B, S, D). Gradient flow preserved."""
        x, x_padding, pos_ids = self._prepare_batch(pretrain_gene_x)
        x_in = torch.unsqueeze(x, 2)
        x_emb = self.model.token_emb(x_in, output_weight=0)
        position_emb = self.model.pos_emb(pos_ids)
        x_emb += position_emb
        return self.model.encoder(x_emb, padding_mask=x_padding)

    def full_forward(self, pretrain_gene_x):
        """Full encoder+decoder forward (B, 19266) → (B, S, decoder_hidden).
        Used by perturbation task. Requires model.to_final = None."""
        value_labels = (pretrain_gene_x > 0).float()
        B = pretrain_gene_x.size(0)
        data_gene_ids = torch.arange(
            pretrain_gene_x.size(1), device=pretrain_gene_x.device
        ).unsqueeze(0).expand(B, -1).float()

        x_enc, enc_padding = _gather_data(pretrain_gene_x, value_labels, self.pad_token_id)
        enc_pos, _ = _gather_data(data_gene_ids, value_labels, self.pad_token_id)
        enc_pos = enc_pos.long()
        enc_pos[enc_padding] = self.seq_len
        encoder_labels = (pretrain_gene_x > 0)

        dec_data = pretrain_gene_x.clone()
        dec_padding = torch.zeros_like(dec_data, dtype=torch.bool)
        dec_pos = torch.arange(
            pretrain_gene_x.size(1), device=pretrain_gene_x.device
        ).unsqueeze(0).expand(B, -1).long()

        return self.model(
            x=x_enc, padding_label=enc_padding,
            encoder_position_gene_ids=enc_pos,
            encoder_labels=encoder_labels,
            decoder_data=dec_data, mask_gene_name=False,
            mask_labels=None,
            decoder_position_gene_ids=dec_pos,
            decoder_data_padding_labels=dec_padding,
        )

    # ------------------------------------------------------------------ #
    #  Forward dispatch
    # ------------------------------------------------------------------ #
    def forward(self, pretrain_gene_x=None, **kwargs):
        """Forward: embedding extraction from raw (B, 19266)."""
        if pretrain_gene_x is not None:
            with torch.no_grad():
                return self.get_cell_embedding(pretrain_gene_x)
        raise ValueError("pretrain_gene_x is required")
