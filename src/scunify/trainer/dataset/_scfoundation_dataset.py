"""Training dataset for scFoundation — inherits from registry, adds MAE masking.

Changes from inference version:
- Randomly masks a fraction of non-zero genes (MAE-style)
- Returns all inputs needed by MaeAutobin.forward() plus targets for loss
- batch_size must be 1 (variable sequence lengths per cell)
"""

import logging

import numpy as np
import scipy.sparse as sp
import torch

from ...registry.dataset._scfoundation_dataset import ScFoundationDataset

logger = logging.getLogger(__name__)


class ScFoundationTrainingDataset(ScFoundationDataset):
    """scFoundation training dataset with MAE masking.

    Inherits ``__init__`` from :class:`ScFoundationDataset` — same gene
    alignment, normalization, resolution token.

    Overrides ``__getitem__`` to split non-zero genes into visible
    (encoder) and masked (decoder) sets, producing all inputs for
    ``MaeAutobin.forward()`` plus ground-truth targets for MSE loss.
    """

    def __init__(self, adata, config):
        super().__init__(adata, config)
        training_cfg = config.get("training", {})
        mae_cfg = training_cfg.get("mae", {})
        self.mask_ratio = float(mae_cfg.get("mask_ratio", 0.4))
        self.mask_token_id = config.model_param[config.inference["version"]][
            "mae_autobin"
        ]["mask_token_id"]

    def __getitem__(self, idx):
        # ── Per-cell normalization (same logic as base class) ──
        row = self.X[idx]
        if sp.issparse(row):
            row = row.toarray().ravel()
        else:
            row = np.asarray(row, dtype=np.float64).ravel()

        gene_vec = np.zeros(19264, dtype=np.float64)
        gene_vec[self.gene_map_valid] = row[self.gene_map[self.gene_map_valid]]

        if self.pre_normalized == "F":
            cell_sum = gene_vec.sum()
            if cell_sum > 0:
                tmpdata = np.log1p(gene_vec / cell_sum * 1e4).tolist()
            else:
                tmpdata = gene_vec.tolist()
        elif self.pre_normalized == "T":
            tmpdata = gene_vec.tolist()
        elif self.pre_normalized == "A":
            tmpdata = gene_vec[:-1].tolist()
        else:
            raise ValueError(
                f"pre_normalized must be T, F or A, got {self.pre_normalized}"
            )

        if self.pre_normalized == "A":
            totalcount = gene_vec[-1]
        else:
            totalcount = gene_vec.sum()

        if self.tg_mode == "f":
            resolution = np.log10(totalcount * self.tg_val)
        elif self.tg_mode == "a":
            resolution = np.log10(totalcount) + self.tg_val
        elif self.tg_mode == "t":
            resolution = self.tg_val
        else:
            raise ValueError(
                f"tgthighres must start with f, a or t, got {self.tg_mode}"
            )

        logtc = np.log10(totalcount) if totalcount > 0 else -np.inf

        pretrain_gene_x = torch.tensor(
            tmpdata + [resolution, logtc], dtype=torch.float32
        ).unsqueeze(0)  # (1, 19266)

        data_gene_ids = torch.arange(19266, dtype=torch.long).unsqueeze(
            0
        )  # (1, 19266)

        # ── Non-zero gene mask ──
        value_labels = (pretrain_gene_x > 0).float()  # (1, 19266)

        # ── Random MAE masking ──
        nonzero_pos = value_labels[0].nonzero(as_tuple=True)[0]
        n_nonzero = len(nonzero_pos)
        n_mask = max(1, int(n_nonzero * self.mask_ratio))
        mask_indices = nonzero_pos[torch.randperm(n_nonzero)[:n_mask]]

        # ── Encoder: only visible (non-masked, non-zero) genes ──
        enc_value_labels = value_labels.clone()
        enc_value_labels[0, mask_indices] = 0
        x_enc, enc_padding = self._gatherData(
            pretrain_gene_x, enc_value_labels, self.pad_token_id
        )
        enc_pos_ids, _ = self._gatherData(
            data_gene_ids, enc_value_labels, self.pad_token_id
        )

        # ── Decoder: all non-zero genes (masked positions → mask_token_id) ──
        dec_data = pretrain_gene_x.clone()
        dec_data[0, mask_indices] = self.mask_token_id
        x_dec, dec_padding = self._gatherData(
            dec_data, value_labels, self.pad_token_id
        )
        dec_pos_ids, _ = self._gatherData(
            data_gene_ids, value_labels, self.pad_token_id
        )

        # ── Labels in decoder space ──
        encoder_labels = (~dec_padding) & (x_dec != self.mask_token_id)
        mask_labels = (~dec_padding) & (x_dec == self.mask_token_id)

        # ── Ground truth (original values at all decoder positions) ──
        x_targets, _ = self._gatherData(
            pretrain_gene_x, value_labels, self.pad_token_id
        )

        return {
            # Encoder inputs
            "x": x_enc.squeeze(0),
            "padding_label": enc_padding.squeeze(0),
            "encoder_position_gene_ids": enc_pos_ids.squeeze(0).long(),
            # Decoder inputs
            "encoder_labels": encoder_labels.squeeze(0),
            "decoder_data": x_dec.squeeze(0),
            "decoder_position_gene_ids": dec_pos_ids.squeeze(0).long(),
            "decoder_data_padding_labels": dec_padding.squeeze(0),
            "mask_labels": mask_labels.squeeze(0),
            # Ground truth + meta
            "targets": x_targets.squeeze(0),
            "cid": torch.tensor(idx, dtype=torch.long),
        }
