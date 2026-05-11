"""scFoundation Perturbation dataset — newer GEARS recipe with 19264 expand.

scFoundation backbone is pretrained on a fixed 19264-gene vocab
(``OS_scRNA_gene_index.19264.tsv``). The raw NORMAN dataset has only
5045 HVGs, so we need ``main_gene_selection`` (paper-faithful from
scPEFT/scfoundation/perturbation/gears/pertdata.py) to zero-pad missing
genes and reorder to the 19264 vocab.

Cell graph format (newer GEARS — ``RelatedWorks/Foundations/scFoundation/
GEARS/gears/pertdata.py``)::

    Data(
        x         = (n_genes+1, 1)  # last position = total_count
        y         = (1, n_genes)
        de_idx    = (20,)
        pert      = condition str
        pert_idx  = list[int] | [-1]
    )

Cache: ``<adata_path>.cell_graph_scfoundation.pkl``.

The expanded 19264-gene adata is also persisted next to the source so
re-runs skip the expensive expand step:

    <adata_path>.expanded_19264.h5ad
"""
from __future__ import annotations

import os

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from torch_geometric.data import Data

from .pertdata import BasePertData

# scFoundation pretrained gene vocab (paper-fixed)
_GENE_LIST_19264 = "/home1/irteam/dhkim/scUnify/resources/scFoundation/OS_scRNA_gene_index.19264.tsv"


def main_gene_selection(adata, gene_list):
    """Verbatim from ``scPEFT/scfoundation/perturbation/gears/pertdata.py:22``.

    Reorder/expand adata to the target gene_list, zero-padding missing
    genes; ``var['mask']`` flags padded positions.
    """
    X_df = pd.DataFrame(
        adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
        index=adata.obs_names,
        columns=adata.var["gene_name"],
    )
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    print(f"[scfoundation expand] mapping gene num: {len(gene_list) - len(to_fill_columns)} / {len(gene_list)}")

    padding_df = pd.DataFrame(
        np.zeros((X_df.shape[0], len(to_fill_columns))),
        columns=to_fill_columns,
        index=X_df.index,
    )
    X_df = pd.concat([X_df, padding_df], axis=1)
    X_df = X_df[gene_list]

    var = pd.DataFrame(index=X_df.columns)
    var["mask"] = [1 if g in to_fill_columns else 0 for g in var.index]

    adata_new = ad.AnnData(X=X_df.values, obs=adata.obs.copy(), var=var)
    adata_new.obs_names = adata.obs_names
    adata_new.var_names = X_df.columns
    adata_new.uns = adata.uns
    adata_new.var["gene_name"] = gene_list
    adata_new.X = sparse.csr_matrix(adata_new.X)
    return adata_new, to_fill_columns, var


class ScFoundationPerturbationDataset(BasePertData):
    """Newer GEARS recipe + scFoundation 19264-gene vocab expand."""

    format = "scfoundation"
    gene_list_path: str = _GENE_LIST_19264

    # ------------------------------------------------------------------ #
    #  Adata expand — first-run cache as sister h5ad
    # ------------------------------------------------------------------ #
    def _expand_adata(self, adata):
        cache_path = f"{self.adata_path}.expanded_19264.h5ad"
        if os.path.isfile(cache_path):
            print(f"[scfoundation expand] loading expanded adata: {cache_path}")
            return sc.read_h5ad(cache_path)

        print(f"[scfoundation expand] reading vocab: {self.gene_list_path}")
        gene_list_df = pd.read_csv(self.gene_list_path, header=0, delimiter="\t")
        gene_list = list(gene_list_df["gene_name"])

        if "gene_name" not in adata.var.columns:
            adata.var["gene_name"] = adata.var.index
        adata_exp, _, _ = main_gene_selection(adata, gene_list)
        adata_exp.obs_names_make_unique()
        sc.pp.calculate_qc_metrics(adata_exp, inplace=True)

        print(f"[scfoundation expand] writing {cache_path}")
        adata_exp.write_h5ad(cache_path)
        return adata_exp

    # ------------------------------------------------------------------ #
    #  Cell graph build — newer GEARS recipe (total_count append)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_ensg2sym():
        """Load ENSG → symbol mapping (built once via mygene). Used to convert
        DE genes (Ensembl IDs in ``rank_genes_groups_cov_all``) to symbols
        that match the 19264-expanded ``var_names`` (which are gene symbols
        from ``OS_scRNA_gene_index.19264.tsv``)."""
        import pickle, os
        path = "/home/irteam/dhkim/scUnify/resources/ensg_to_symbol.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return {}

    def create_cell_graph_dataset(self, split_adata, pert_category: str, num_samples: int = 1):
        num_de_genes = 20
        adata_ = split_adata[split_adata.obs["condition"] == pert_category]

        de_genes = adata_.uns.get("rank_genes_groups_cov_all")
        de = de_genes is not None
        if not de:
            num_de_genes = 1

        # ENSG → symbol (resolves the 19264-vocab mismatch with paper DE keys)
        if not hasattr(self, "_ensg2sym"):
            self._ensg2sym = self._load_ensg2sym()

        Xs, ys, pert_idx = [], [], None
        if pert_category != "ctrl":
            pert_idx = self.get_pert_idx(pert_category)
            pert_de_category = adata_.obs["condition_name"][0]
            if de:
                de_ensg = list(de_genes[pert_de_category][:num_de_genes])
                # Convert ENSG → symbol (skip if no mapping)
                de_syms = [self._ensg2sym[e] for e in de_ensg if e in self._ensg2sym]
                de_idx = np.where(adata_.var_names.isin(np.array(de_syms)))[0]
                if len(de_idx) == 0:
                    de_idx = [-1] * num_de_genes
            else:
                de_idx = [-1] * num_de_genes
            for cell_z in adata_.X:
                ctrl_samples = self.ctrl_adata[
                    np.random.randint(0, len(self.ctrl_adata), num_samples), :
                ]
                ctrl_obs_counts = ctrl_samples.obs["total_count"]
                for ic, c in enumerate(ctrl_samples.X):
                    ipert_total_count = np.array([[float(ctrl_obs_counts[ic])]])
                    comb = np.append(c.toarray(), ipert_total_count, axis=1)  # (1, n_genes+1)
                    Xs.append(comb)
                    ys.append(cell_z)
        else:
            de_idx = [-1] * num_de_genes
            ctrl_obs_counts = adata_.obs["total_count"]
            for ic, cell_z in enumerate(adata_.X):
                ipert_total_count = np.array([[float(ctrl_obs_counts[ic])]])
                comb = np.append(cell_z.toarray(), ipert_total_count, axis=1)
                Xs.append(comb)
                ys.append(cell_z)

        cell_graphs = []
        for X, y in zip(Xs, ys):
            cell_graphs.append(self._create_cell_graph(X, y.toarray(), de_idx, pert_category, pert_idx))
        return cell_graphs

    @staticmethod
    def _create_cell_graph(X, y, de_idx, pert: str, pert_idx):
        # X is already (1, n_genes+1) with total_count appended
        feature_mat = torch.Tensor(X).T  # (n_genes+1, 1)
        return Data(
            x=feature_mat,
            pert_idx=pert_idx if pert_idx is not None else [-1],
            y=torch.Tensor(y),
            de_idx=de_idx,
            pert=pert,
        )
