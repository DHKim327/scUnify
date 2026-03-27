"""
ScFoundation Dataset - faithfully following the original get_embedding.py
Reference: Foundations/scFoundation/model/get_embedding.py lines 140-190
"""

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ScFoundationDataset(Dataset):
    """
    Faithfully reproduces the cell embedding generation logic from Foundations get_embedding.py.

    Key features:
    - Only batch_size=1 is supported (same as the original paper)
    - Lazy loading: sparse X kept in __init__, per-cell processing in __getitem__
    - Gene selection via precomputed index mapping (not per-cell DataFrame ops)
    """

    def __init__(self, adata, config):
        self.config = config
        self.pad_token_id = config.model_param[config.inference["version"]]["mae_autobin"]["pad_token_id"]
        self.collator = None
        self.sampler = None
        # Force batch_size=1 (same as original paper)
        if config.inference.get("batch_size", 1) != 1:
            raise ValueError("ScFoundationDataset only supports batch_size=1 (same as original paper)")

        # Load gene list (19264 target genes)
        gene_list_df = pd.read_csv(config.resources["gene_list"], sep='\t', header=0)
        self.gene_list = list(gene_list_df['gene_name'])

        # Build gene column names from adata
        try:
            col = adata.var.gene_name.tolist()
        except Exception:
            col = adata.var_names.tolist()

        # Precompute gene_map: gene_list[i] → column index in adata (-1 if missing)
        col_to_idx = {c: i for i, c in enumerate(col)}
        gene_map = np.array([col_to_idx.get(g, -1) for g in self.gene_list], dtype=np.intp)
        self.gene_map_valid = gene_map >= 0  # (19264,) bool
        self.gene_map = gene_map  # (19264,) index (-1 = zero-pad)

        # Lazy: keep sparse reference
        self.X = adata.X
        self.N = adata.n_obs

        n_matched = int(self.gene_map_valid.sum())
        logger.info(f"ScFoundationDataset: {self.N} cells, {n_matched}/19264 genes matched")

        # Preprocessing option
        self.pre_normalized = config.preprocessing.get("option", "F")  # 'F', 'T', 'A'

        # tgthighres
        tg = config.inference["tgthighres"]
        self.tg_mode = tg[0]  # 'f', 'a', or 't'
        self.tg_val = float(tg[1:])  # numeric value

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # ── Per-cell: sparse row → 19264 gene vector via precomputed mapping ──
        row = self.X[idx]
        if sp.issparse(row):
            row = row.toarray().ravel()
        else:
            row = np.asarray(row, dtype=np.float64).ravel()

        # Map to 19264 genes (zero-pad missing)
        gene_vec = np.zeros(19264, dtype=np.float64)
        gene_vec[self.gene_map_valid] = row[self.gene_map[self.gene_map_valid]]

        # Pre-normalization (original paper lines 159-165)
        if self.pre_normalized == 'F':
            cell_sum = gene_vec.sum()
            if cell_sum > 0:
                tmpdata = np.log1p(gene_vec / cell_sum * 1e4).tolist()
            else:
                tmpdata = gene_vec.tolist()
        elif self.pre_normalized == 'T':
            tmpdata = gene_vec.tolist()
        elif self.pre_normalized == 'A':
            tmpdata = gene_vec[:-1].tolist()
        else:
            raise ValueError(f'pre_normalized must be T, F or A, got {self.pre_normalized}')

        # Total count (original paper lines 167-170)
        if self.pre_normalized == 'A':
            totalcount = gene_vec[-1]
        else:
            totalcount = gene_vec.sum()

        # Resolution token (original paper lines 172-179)
        if self.tg_mode == 'f':
            resolution = np.log10(totalcount * self.tg_val)
        elif self.tg_mode == 'a':
            resolution = np.log10(totalcount) + self.tg_val
        elif self.tg_mode == 't':
            resolution = self.tg_val
        else:
            raise ValueError(f'tgthighres must start with f, a or t, got {self.tg_mode}')

        logtc = np.log10(totalcount) if totalcount > 0 else -np.inf

        # pretrain_gene_x: [19264 genes, resolution, logtc] (original paper lines 173-179)
        pretrain_gene_x = torch.tensor(
            tmpdata + [resolution, logtc],
            dtype=torch.float32
        ).unsqueeze(0)  # (1, 19266)

        # data_gene_ids: [0, 1, 2, ..., 19265] (original paper line 180)
        data_gene_ids = torch.arange(19266, dtype=torch.long).unsqueeze(0)  # (1, 19266)

        # value_labels: mask for non-zero values (original paper line 182)
        value_labels = (pretrain_gene_x > 0).float()  # (1, 19266)

        # Apply gatherData (original paper line 183)
        x, x_padding = self._gatherData(pretrain_gene_x, value_labels, self.pad_token_id)
        position_gene_ids, _ = self._gatherData(data_gene_ids, value_labels, self.pad_token_id)

        # Return: (values, padding_mask, position_ids, cell_id)
        return (
            x.squeeze(0),  # (K,)
            x_padding.squeeze(0),  # (K,) bool
            position_gene_ids.squeeze(0),  # (K,) long
            torch.tensor(idx, dtype=torch.long),  # cell id
        )

    def _gatherData(self, data, labels, pad_token_id):
        """
        Reproduces the gatherData function from the original load.py.
        """
        max_num = int(labels.sum(1).max().item())

        fake_data = torch.full((data.shape[0], max_num), pad_token_id, dtype=data.dtype)
        data_padded = torch.cat([data, fake_data], dim=1)

        fake_label = torch.ones((labels.shape[0], max_num), dtype=labels.dtype)
        none_labels = (labels == 0)
        labels_copy = labels.clone().float()
        labels_copy[none_labels] = -float('inf')

        F = labels.shape[1]
        tmp_data = torch.tensor(
            [(i + 1) * 20000 for i in range(F, 0, -1)],
            dtype=labels_copy.dtype
        )
        labels_copy = labels_copy + tmp_data
        labels_padded = torch.cat([labels_copy, fake_label], dim=1)

        topk_indices = labels_padded.topk(max_num, dim=1).indices

        gathered_data = torch.gather(data_padded, 1, topk_indices)
        padding_labels = (gathered_data == pad_token_id)

        return gathered_data, padding_labels
