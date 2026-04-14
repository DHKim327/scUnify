"""
CKA (Centered Kernel Alignment) — model-pair representation similarity.

Measures how similar two models' embedding structures are,
independent of embedding dimension.

Reference: Kornblith et al., ICML 2019
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Linear CKA between two embedding matrices.

    Parameters
    ----------
    X : (n_cells, d1)
    Y : (n_cells, d2)

    Returns
    -------
    CKA score in [0, 1]. Higher = more similar structure.
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    YtX = Y.T @ X
    XtX = X.T @ X
    YtY = Y.T @ Y

    num = np.linalg.norm(YtX, "fro") ** 2
    denom = np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro")

    if denom == 0:
        return 0.0
    return float(num / denom)


class CKAWrapper:
    """CKA evaluation across multiple model embeddings.

    Parameters
    ----------
    adata
        AnnData with embeddings in obsm.
    embedding_keys
        List of obsm keys to compare pair-wise.
    label_key
        obs column for per-celltype CKA (optional).
    n_sample
        Max cells to sample for speed. None = use all.
    seed
        Random seed for sampling.
    """

    def __init__(
        self,
        adata: "AnnData",
        embedding_keys: list[str],
        label_key: str | None = None,
        n_sample: int | None = 10000,
        seed: int = 42,
    ):
        self.adata = adata
        self.embedding_keys = embedding_keys
        self.label_key = label_key
        self.n_sample = n_sample
        self.seed = seed

        self._global_result: pd.DataFrame | None = None
        self._per_ct_result: pd.DataFrame | None = None

    def _sample_idx(self, mask: np.ndarray | None = None) -> np.ndarray:
        """Return sampled indices (or all if n_sample is None)."""
        if mask is not None:
            pool = np.where(mask)[0]
        else:
            pool = np.arange(len(self.adata))

        if self.n_sample is not None and len(pool) > self.n_sample:
            rng = np.random.RandomState(self.seed)
            return rng.choice(pool, self.n_sample, replace=False)
        return pool

    def run_global(self) -> pd.DataFrame:
        """Compute pair-wise CKA across all embeddings.

        Returns
        -------
        Symmetric DataFrame (n_models x n_models), values in [0, 1].
        """
        keys = self.embedding_keys
        n = len(keys)
        idx = self._sample_idx()

        matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                X = np.array(self.adata.obsm[keys[i]])[idx]
                Y = np.array(self.adata.obsm[keys[j]])[idx]
                cka = linear_cka(X, Y)
                matrix[i, j] = cka
                matrix[j, i] = cka

        self._global_result = pd.DataFrame(matrix, index=keys, columns=keys)
        return self._global_result

    def run_per_celltype(self, min_cells: int = 50) -> pd.DataFrame:
        """Compute CKA per celltype for each model pair.

        Parameters
        ----------
        min_cells
            Skip celltypes with fewer cells.

        Returns
        -------
        DataFrame (celltypes x model_pairs).
        """
        if self.label_key is None:
            raise ValueError("label_key is required for per-celltype CKA.")

        keys = self.embedding_keys
        labels = self.adata.obs[self.label_key].values
        unique_ct = sorted(set(labels))

        # Model pair names
        pair_names = []
        pair_indices = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ki = keys[i].replace("X_", "")
                kj = keys[j].replace("X_", "")
                pair_names.append(f"{ki} vs {kj}")
                pair_indices.append((i, j))

        rng = np.random.RandomState(self.seed)
        results = {}

        for ct in unique_ct:
            ct_mask = labels == ct
            n_ct = ct_mask.sum()
            if n_ct < min_cells:
                continue

            ct_idx = np.where(ct_mask)[0]
            if self.n_sample and n_ct > self.n_sample:
                ct_idx = rng.choice(ct_idx, self.n_sample, replace=False)

            row = {}
            for (i, j), pname in zip(pair_indices, pair_names):
                X = np.array(self.adata.obsm[keys[i]])[ct_idx]
                Y = np.array(self.adata.obsm[keys[j]])[ct_idx]
                row[pname] = linear_cka(X, Y)
            results[ct] = row

        self._per_ct_result = pd.DataFrame(results).T
        self._per_ct_result.index.name = "celltype"
        return self._per_ct_result

    def run(self) -> dict[str, pd.DataFrame]:
        """Run both global and per-celltype CKA.

        Returns
        -------
        {"global": DataFrame, "per_celltype": DataFrame}
        """
        result = {"global": self.run_global()}
        if self.label_key is not None:
            result["per_celltype"] = self.run_per_celltype()
        return result
