"""
ScFoundation Dataset - faithfully following the original get_embedding.py
Reference: Foundations/scFoundation/model/get_embedding.py lines 140-190
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


class ScFoundationDataset(Dataset):
    """
    Faithfully reproduces the cell embedding generation logic from Foundations get_embedding.py.
    
    Key features:
    - Only batch_size=1 is supported (same as the original paper)
    - Implements lines 140-190 from the original paper exactly
    - Includes main_gene_selection logic (mapping to 19264 genes)
    """
    
    def __init__(self, adata, config):
        self.config = config
        self.pad_token_id = config.model_param[config.inference["version"]]["mae_autobin"]["pad_token_id"]
        self.collator = None
        self.sampler = None
        # Force batch_size=1 (same as original paper)
        if config.inference.get("batch_size", 1) != 1:
            raise ValueError("ScFoundationDataset only supports batch_size=1 (same as original paper)")
        
        # Load gene list
        gene_list_df = pd.read_csv(config.resources["gene_list"], sep='\t', header=0)
        self.gene_list = list(gene_list_df['gene_name'])
        
        # Convert original data to pandas DataFrame
        idx = adata.obs_names.tolist()
        try:
            col = adata.var.gene_name.tolist()
        except:
            col = adata.var_names.tolist()
        
        if sp.issparse(adata.X):
            gexpr_feature = adata.X.toarray()
        else:
            gexpr_feature = adata.X
        
        self.gexpr_feature = pd.DataFrame(gexpr_feature, index=idx, columns=col)
        self.N = self.gexpr_feature.shape[0]
        
        # Lazy loading: gene selection performed per cell in __getitem__ (memory-efficient)
        print(f'Lazy loading enabled: gene selection will be done per cell')
        print(f'Original gene count: {self.gexpr_feature.shape[1]} → will convert to 19264 in __getitem__')
        
        # Preprocessing option
        self.pre_normalized = config.preprocessing.get("option", "F")  # 'F', 'T', 'A'
        
        # tgthighres
        tg = config.inference["tgthighres"]
        self.tg_mode = tg[0]  # 'f', 'a', or 't'
        self.tg_val = float(tg[1:])  # numeric value
        
        print(f"ScFoundationDataset initialized: {self.N} cells, 19264 genes")
        print(f"  pre_normalized={self.pre_normalized}, tgthighres={tg}")
    
    def _main_gene_selection(self, X_df, gene_list):
        """
        Reproduces the main_gene_selection function from the original paper.
        Maps to 19264 genes; missing genes are zero-padded.
        """
        to_fill_columns = list(set(gene_list) - set(X_df.columns))
        
        if to_fill_columns:
            padding_df = pd.DataFrame(
                np.zeros((X_df.shape[0], len(to_fill_columns))),
                columns=to_fill_columns,
                index=X_df.index
            )
            X_df = pd.DataFrame(
                np.concatenate([X_df.values, padding_df.values], axis=1),
                index=X_df.index,
                columns=list(X_df.columns) + list(padding_df.columns)
            )
        
        # Reorder columns to gene_list order
        X_df = X_df[gene_list]
        return X_df
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        # Lazy gene selection: map only this cell to 19264 genes
        cell_data = self.gexpr_feature.iloc[idx:idx+1, :]  # (1, M)
        
        if cell_data.shape[1] < 19264:
            cell_data = self._main_gene_selection(cell_data, self.gene_list)
        
        # Convert to 1D Series for downstream logic
        cell_series = cell_data.iloc[0, :]
        
        # Pre-normalization (original paper lines 159-165)
        if self.pre_normalized == 'F':
            # normalize_total=10000 + log1p
            cell_sum = cell_series.sum()
            if cell_sum > 0:
                tmpdata = np.log1p(cell_series / cell_sum * 1e4)
            else:
                tmpdata = cell_series
            tmpdata = tmpdata.tolist()
        elif self.pre_normalized == 'T':
            tmpdata = cell_series.tolist()
        elif self.pre_normalized == 'A':
            tmpdata = cell_series[:-1].tolist()
        else:
            raise ValueError(f'pre_normalized must be T, F or A, got {self.pre_normalized}')
        
        # Total count (original paper lines 167-170)
        if self.pre_normalized == 'A':
            totalcount = cell_series.iloc[-1]
        else:
            totalcount = cell_series.sum()
        
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
        
        data: (1, F) tensor
        labels: (1, F) binary mask
        pad_token_id: int
        
        Returns:
        - gathered_data: (1, K) where K = max(labels.sum())
        - padding_labels: (1, K) bool mask
        """
        # Max of labels.sum(1)
        max_num = int(labels.sum(1).max().item())
        
        # Add padding
        fake_data = torch.full((data.shape[0], max_num), pad_token_id, dtype=data.dtype)
        data_padded = torch.cat([data, fake_data], dim=1)
        
        # Assign priority to labels (original paper logic)
        fake_label = torch.ones((labels.shape[0], max_num), dtype=labels.dtype)
        none_labels = (labels == 0)
        labels_copy = labels.clone().float()
        labels_copy[none_labels] = -float('inf')
        
        # Add position-based priority
        F = labels.shape[1]
        tmp_data = torch.tensor(
            [(i + 1) * 20000 for i in range(F, 0, -1)],
            dtype=labels_copy.dtype
        )
        labels_copy = labels_copy + tmp_data
        labels_padded = torch.cat([labels_copy, fake_label], dim=1)
        
        # Top-K selection
        topk_indices = labels_padded.topk(max_num, dim=1).indices
        
        # Gather
        gathered_data = torch.gather(data_padded, 1, topk_indices)
        padding_labels = (gathered_data == pad_token_id)
        
        return gathered_data, padding_labels
