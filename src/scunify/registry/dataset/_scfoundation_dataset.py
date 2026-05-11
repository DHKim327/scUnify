"""
ScFoundation Dataset — supports batch_size > 1.

Key change from original: __getitem__ returns fixed-length 19266 vector
(19264 genes + resolution + logtc). gatherData is performed in the
model forward (batch-wise), not in __getitem__ (cell-wise).
This allows standard PyTorch batching with batch_size > 1.

Reference: Foundations/scFoundation/model/get_embedding.py
"""

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ScFoundationDataset(Dataset):
    """scFoundation dataset — returns fixed-length 19266 vector per cell.

    gatherData (variable-length encoder/decoder split) is deferred to
    the model wrapper's forward method, enabling batch_size > 1.
    """

    N_GENES = 19264
    N_FEATURES = 19266  # 19264 genes + resolution + logtc

    def __init__(self, adata, config):
        self.config = config
        model_cfg = config.get("model", {})
        self.pad_token_id = config.model_param[model_cfg.get("version", "cell")]["mae_autobin"]["pad_token_id"]

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
        self.pre_normalized = config.preprocessing.get("option", "F")

        # tgthighres
        tg = model_cfg.get("tgthighres", "t4")
        self.tg_mode = tg[0]
        self.tg_val = float(tg[1:])

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Per-cell: sparse row → 19264 gene vector
        row = self.X[idx]
        if sp.issparse(row):
            row = row.toarray().ravel()
        else:
            row = np.asarray(row, dtype=np.float64).ravel()

        gene_vec = np.zeros(self.N_GENES, dtype=np.float64)
        gene_vec[self.gene_map_valid] = row[self.gene_map[self.gene_map_valid]]

        # Pre-normalization
        if self.pre_normalized == 'F':
            cell_sum = gene_vec.sum()
            if cell_sum > 0:
                tmpdata = np.log1p(gene_vec / cell_sum * 1e4)
            else:
                tmpdata = gene_vec.copy()
        elif self.pre_normalized == 'T':
            tmpdata = gene_vec.copy()
        elif self.pre_normalized == 'A':
            tmpdata = gene_vec[:-1].copy()
        else:
            raise ValueError(f'pre_normalized must be T, F or A, got {self.pre_normalized}')

        # Total count
        if self.pre_normalized == 'A':
            totalcount = gene_vec[-1]
        else:
            totalcount = gene_vec.sum()

        # Resolution token
        if self.tg_mode == 'f':
            resolution = np.log10(totalcount * self.tg_val) if totalcount > 0 else 0.0
        elif self.tg_mode == 'a':
            resolution = np.log10(totalcount) + self.tg_val if totalcount > 0 else 0.0
        elif self.tg_mode == 't':
            resolution = self.tg_val
        else:
            raise ValueError(f'tgthighres must start with f, a or t, got {self.tg_mode}')

        logtc = np.log10(totalcount) if totalcount > 0 else 0.0

        # Fixed-length output: [19264 genes, resolution, logtc] = 19266
        features = np.concatenate([tmpdata, [resolution, logtc]]).astype(np.float32)

        return {
            "pretrain_gene_x": torch.from_numpy(features),  # (19266,)
            "cid": torch.tensor(idx, dtype=torch.long),
        }

    @staticmethod
    def collator(batch):
        """Stack fixed-length features into batch."""
        return {
            "pretrain_gene_x": torch.stack([b["pretrain_gene_x"] for b in batch]),  # (B, 19266)
            "cid": torch.tensor([b["cid"] for b in batch], dtype=torch.long),
        }
