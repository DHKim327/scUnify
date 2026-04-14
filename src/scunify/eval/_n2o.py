"""
N2O (Nearest Neighbor Overlap) — model-pair local agreement.

Measures how much two models agree on each cell's local neighborhood.
Label-free: only uses embeddings, no celltype annotation required.

Reference: Lin et al., arXiv 1909.10724
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


class N2OWrapper:
    """N2O evaluation across multiple model embeddings.

    Parameters
    ----------
    adata
        AnnData with embeddings in obsm.
    embedding_keys
        List of obsm keys to compare pair-wise.
    label_key
        obs column for per-celltype breakdown (optional).
    k
        Number of nearest neighbors.
    """

    def __init__(
        self,
        adata: "AnnData",
        embedding_keys: list[str],
        label_key: str | None = None,
        k: int = 50,
    ):
        self.adata = adata
        self.embedding_keys = embedding_keys
        self.label_key = label_key
        self.k = k

        self._knn_cache: dict[str, np.ndarray] = {}
        self._matrix_result: pd.DataFrame | None = None
        self._per_cell_result: np.ndarray | None = None
        self._per_ct_result: pd.DataFrame | None = None

    def _get_knn(self, key: str) -> np.ndarray:
        """Compute and cache kNN indices for an embedding.

        Returns
        -------
        (n_cells, k) array of neighbor indices.
        """
        if key not in self._knn_cache:
            emb = np.array(self.adata.obsm[key])
            nn = NearestNeighbors(
                n_neighbors=self.k + 1, metric="cosine", n_jobs=-1,
            )
            nn.fit(emb)
            _, indices = nn.kneighbors(emb)
            self._knn_cache[key] = indices[:, 1:]  # exclude self
            logger.info(f"kNN computed: {key}")
        return self._knn_cache[key]

    def n2o_pairwise(self, key_a: str, key_b: str) -> np.ndarray:
        """Per-cell N2O between two embeddings.

        N2O_i = |kNN_A(i) ∩ kNN_B(i)| / k

        Returns
        -------
        (n_cells,) array, values in [0, 1].
        """
        nn_a = self._get_knn(key_a)
        nn_b = self._get_knn(key_b)
        n_cells = len(nn_a)

        overlap = np.array([
            len(set(nn_a[i]) & set(nn_b[i])) / self.k
            for i in range(n_cells)
        ])
        return overlap

    def run_matrix(self) -> pd.DataFrame:
        """Pair-wise mean N2O matrix.

        Returns
        -------
        Symmetric DataFrame (n_models x n_models), values in [0, 1].
        """
        keys = self.embedding_keys
        n = len(keys)
        matrix = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                overlap = self.n2o_pairwise(keys[i], keys[j])
                mean_n2o = overlap.mean()
                matrix[i, j] = mean_n2o
                matrix[j, i] = mean_n2o
                logger.info(f"N2O({keys[i]}, {keys[j]}) = {mean_n2o:.4f}")

        self._matrix_result = pd.DataFrame(matrix, index=keys, columns=keys)
        return self._matrix_result

    def run_per_cell(self) -> np.ndarray:
        """Mean N2O across all model pairs, per cell.

        Returns
        -------
        (n_cells,) array — multi-model agreement score per cell.
        """
        keys = self.embedding_keys
        n_cells = len(self.adata)
        all_scores = np.zeros(n_cells)
        n_pairs = 0

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                all_scores += self.n2o_pairwise(keys[i], keys[j])
                n_pairs += 1

        self._per_cell_result = all_scores / max(n_pairs, 1)
        return self._per_cell_result

    def run_per_celltype(self) -> pd.DataFrame:
        """Per-celltype N2O statistics.

        Returns
        -------
        DataFrame (celltypes x [mean, std, median]) for the multi-model
        agreement score.
        """
        if self.label_key is None:
            raise ValueError("label_key is required for per-celltype N2O.")

        if self._per_cell_result is None:
            self.run_per_cell()

        labels = self.adata.obs[self.label_key].values
        unique_ct = sorted(set(labels))

        rows = {}
        for ct in unique_ct:
            mask = labels == ct
            scores = self._per_cell_result[mask]
            rows[ct] = {
                "mean": scores.mean(),
                "std": scores.std(),
                "median": np.median(scores),
                "n_cells": mask.sum(),
            }

        self._per_ct_result = pd.DataFrame(rows).T
        self._per_ct_result.index.name = "celltype"
        return self._per_ct_result

    def run(self) -> dict:
        """Run all N2O analyses.

        Returns
        -------
        {"matrix": DataFrame, "per_cell": ndarray, "per_celltype": DataFrame}
        """
        result = {
            "matrix": self.run_matrix(),
            "per_cell": self.run_per_cell(),
        }
        if self.label_key is not None:
            result["per_celltype"] = self.run_per_celltype()
        return result
