"""scGPT Perturbation dataset — paper Tutorial recipe.

Cell graph format (Cui et al. 2024, ``Tutorial_Perturbation.ipynb``)::

    Data(
        x         = (n_genes, 2)   # col 0 = gene_value, col 1 = pert_flag (0/1)
        y         = (1, n_genes)   # post-perturbation target
        de_idx    = (20,)          # top-DE gene indices
        pert      = condition str
        pert_idx  = list[int] | [-1]   (carried for downstream metrics)
    )

No total_count append, no 19264 expand — operates on the raw 5045-gene
NORMAN.h5ad. Cache: ``<adata_path>.cell_graph_scgpt.pkl``.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from .pertdata import BasePertData


class ScGPTPerturbationDataset(BasePertData):
    """scGPT paper Tutorial recipe — feature_mat = (n_genes, 2)."""

    format = "scgpt"

    # ------------------------------------------------------------------ #
    #  cell graph build — Tutorial cell-7 layout
    # ------------------------------------------------------------------ #
    def create_cell_graph_dataset(self, split_adata, pert_category: str, num_samples: int = 1):
        num_de_genes = 20
        adata_ = split_adata[split_adata.obs["condition"] == pert_category]

        de_genes = adata_.uns.get("rank_genes_groups_cov_all")
        de = de_genes is not None
        if not de:
            num_de_genes = 1

        Xs, ys, pert_idx = [], [], None
        if pert_category != "ctrl":
            pert_idx = self.get_pert_idx(pert_category)
            pert_de_category = adata_.obs["condition_name"][0]
            if de:
                de_idx = np.where(
                    adata_.var_names.isin(np.array(de_genes[pert_de_category][:num_de_genes]))
                )[0]
            else:
                de_idx = [-1] * num_de_genes
            for cell_z in adata_.X:
                ctrl_samples = self.ctrl_adata[
                    np.random.randint(0, len(self.ctrl_adata), num_samples), :
                ]
                for c in ctrl_samples.X:
                    Xs.append(c.toarray())   # (1, n_genes)
                    ys.append(cell_z)
        else:
            de_idx = [-1] * num_de_genes
            for cell_z in adata_.X:
                Xs.append(cell_z.toarray())
                ys.append(cell_z)

        cell_graphs = []
        for X, y in zip(Xs, ys):
            cell_graphs.append(self._create_cell_graph(X, y.toarray(), de_idx, pert_category, pert_idx))
        return cell_graphs

    @staticmethod
    def _create_cell_graph(X, y, de_idx, pert: str, pert_idx):
        """Tutorial recipe::
            pert_feats[g] = 1 if gene g is in pert_idx else 0
            feature_mat   = concat([X, pert_feats], axis=0).T   → (n_genes, 2)
        """
        n_genes = X.shape[1]
        pert_feats = np.zeros((1, n_genes), dtype=np.float32)
        if pert_idx is not None:
            for p in pert_idx:
                pert_feats[0, int(np.abs(p))] = 1.0
        feature_mat = torch.Tensor(np.concatenate([X, pert_feats], axis=0)).T  # (n_genes, 2)
        return Data(
            x=feature_mat,
            pert_idx=pert_idx if pert_idx is not None else [-1],
            y=torch.Tensor(y),
            de_idx=de_idx,
            pert=pert,
        )
