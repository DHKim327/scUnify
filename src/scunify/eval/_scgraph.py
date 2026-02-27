"""
scGraph wrapper (Islander-based)
Consensus distance-based embedding quality evaluation

Reference: Islander/src/scGraph.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import trim_mean
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from anndata import AnnData


class ScGraphWrapper:
    """scGraph: Consensus distance-based embedding evaluation.
    
    Computes cell-type centroids per batch,
    compares the embedding distance structure against the
    consensus distance matrix to evaluate embedding quality.
    
    Parameters
    ----------
    adata
        AnnData object (embeddings must be stored in obsm)
    embedding_keys
        List of embedding obsm keys to evaluate
    batch_key
        obs column name containing batch information
    label_key
        obs column name containing cell-type labels
    trim_rate
        Fraction to trim from each side for trimmed mean (default: 0.05)
    thres_batch
        Minimum batch size (batches smaller than this are excluded)
    thres_celltype
        Minimum cell-type size (cell types smaller than this are excluded)
    
    Examples
    --------
    >>> wrapper = ScGraphWrapper(
    ...     adata,
    ...     embedding_keys=["X_scgpt", "X_uce"],
    ...     batch_key="batch",
    ...     label_key="cell_type",
    ... )
    >>> results = wrapper.run()
    """
    
    def __init__(
        self,
        adata: "AnnData",
        embedding_keys: list[str],
        batch_key: str,
        label_key: str,
        trim_rate: float = 0.05,
        thres_batch: int = 100,
        thres_celltype: int = 10,
    ):
        self.adata = adata
        self.embedding_keys = embedding_keys
        self.batch_key = batch_key
        self.label_key = label_key
        self.trim_rate = trim_rate
        self.thres_batch = thres_batch
        self.thres_celltype = thres_celltype
        
        # Cell types to exclude
        self._ignore_celltypes: list[str] = []
        # Per-batch consensus distances
        self._collect_pca: dict[str, pd.DataFrame] = {}
        # consensus distance matrix
        self._consensus_df: pd.DataFrame | None = None
        
        self._prepared = False
        self._results: pd.DataFrame | None = None
    
    def _preprocess(self) -> None:
        """Filter out cell types that are too small."""
        celltype_counts = self.adata.obs[self.label_key].value_counts()
        for celltype, count in celltype_counts.items():
            if count < self.thres_celltype:
                print(f"Skipped cell type '{celltype}': < {self.thres_celltype} cells")
                self._ignore_celltypes.append(celltype)
    
    def _calculate_trimmed_means(
        self,
        X: np.ndarray,
        labels: pd.Series,
        trim_proportion: float = 0.05,
    ) -> pd.DataFrame:
        """Calculate trimmed mean centroid for each cell type.
        
        Parameters
        ----------
        X
            embedding matrix (n_cells, n_dims)
        labels
            cell type labels
        trim_proportion
            Fraction to trim from each side
        
        Returns
        -------
        DataFrame with centroids (n_celltypes, n_dims)
        """
        unique_labels = [l for l in labels.unique() if l not in self._ignore_celltypes]
        centroids = {}
        
        for label in unique_labels:
            mask = labels == label
            X_subset = X[mask]
            if len(X_subset) > 0:
                centroid = np.array([
                    trim_mean(X_subset[:, i], proportiontocut=trim_proportion)
                    for i in range(X_subset.shape[1])
                ])
                centroids[label] = centroid
        
        return pd.DataFrame(centroids).T
    
    def _compute_pairwise_distances(self, centroids: pd.DataFrame) -> pd.DataFrame:
        """Calculate pairwise distances between cell types.
        
        Parameters
        ----------
        centroids
            centroid DataFrame (n_celltypes, n_dims)
        
        Returns
        -------
        Distance matrix DataFrame (n_celltypes, n_celltypes)
        """
        dist_matrix = cdist(centroids.values, centroids.values, metric='euclidean')
        return pd.DataFrame(
            dist_matrix,
            index=centroids.index,
            columns=centroids.index,
        )
    
    def _process_batches(self) -> None:
        """Calculate centroids and pairwise distances for each batch."""
        print("Processing batches: calculating centroids and pairwise distances...")
        
        # PCA-based consensus calculation
        import scanpy as sc
        
        for batch in tqdm(self.adata.obs[self.batch_key].unique()):
            adata_batch = self.adata[self.adata.obs[self.batch_key] == batch].copy()
            
            if len(adata_batch) < self.thres_batch:
                print(f"Skipped batch '{batch}': < {self.thres_batch} cells")
                continue
            
            # HVG + PCA computation
            try:
                sc.pp.highly_variable_genes(adata_batch, n_top_genes=min(1000, adata_batch.n_vars))
                sc.pp.pca(adata_batch, n_comps=min(10, adata_batch.n_obs - 1), use_highly_variable=True)
            except Exception as e:
                print(f"Skipped batch '{batch}': PCA failed ({e})")
                continue
            
            # Centroid and distance calculation
            centroids = self._calculate_trimmed_means(
                adata_batch.obsm["X_pca"],
                adata_batch.obs[self.label_key],
                trim_proportion=self.trim_rate,
            )
            
            if len(centroids) < 2:
                continue
            
            pairwise_dist = self._compute_pairwise_distances(centroids)
            # Normalize by max
            normalized = pairwise_dist.div(pairwise_dist.max(axis=0), axis=1)
            self._collect_pca[batch] = normalized
    
    def _calculate_consensus(self) -> None:
        """Average per-batch distances to create consensus distance matrix."""
        if not self._collect_pca:
            raise ValueError("No batches processed. Run _process_batches first.")
        
        # Merge distance matrices from all batches
        df_combined = pd.concat(self._collect_pca.values(), axis=0, sort=False)
        # Average over identical indices
        consensus = df_combined.groupby(df_combined.index).mean()
        # Ensure symmetric matrix
        common_labels = consensus.index.intersection(consensus.columns)
        consensus = consensus.loc[common_labels, common_labels]
        # Normalize by max
        self._consensus_df = consensus / consensus.max(axis=0)
    
    def prepare(self) -> "ScGraphWrapper":
        """Preprocess and compute consensus distances.
        
        Returns
        -------
        self
        """
        self._preprocess()
        self._process_batches()
        self._calculate_consensus()
        self._prepared = True
        return self
    
    def _evaluate_embedding(self, obsm_key: str) -> dict[str, float]:
        """Evaluate a single embedding.
        
        Parameters
        ----------
        obsm_key
            Embedding obsm key
        
        Returns
        -------
        Metric dict: {'Rank-PCA': ..., 'Corr-PCA': ..., 'Corr-Weighted': ...}
        """
        if self._consensus_df is None:
            raise ValueError("Consensus not calculated. Run prepare() first.")
        
        # Calculate embedding centroid and distances
        centroids = self._calculate_trimmed_means(
            np.array(self.adata.obsm[obsm_key]),
            self.adata.obs[self.label_key],
            trim_proportion=self.trim_rate,
        )
        
        # Use only cell types common to consensus
        common_labels = centroids.index.intersection(self._consensus_df.index)
        if len(common_labels) < 2:
            return {'Rank-PCA': np.nan, 'Corr-PCA': np.nan, 'Corr-Weighted': np.nan}
        
        centroids = centroids.loc[common_labels]
        consensus = self._consensus_df.loc[common_labels, common_labels]
        
        pairwise_dist = self._compute_pairwise_distances(centroids)
        pairwise_dist = pairwise_dist.loc[common_labels, common_labels]
        normalized = pairwise_dist.div(pairwise_dist.max(axis=0), axis=1)
        
        # Calculate metrics
        rank_corr = self._rank_diff(normalized, consensus)
        corr_pca = self._corr_diff(normalized, consensus)
        corr_weighted = self._corrw_diff(normalized, consensus)
        
        return {
            'Rank-PCA': rank_corr,
            'Corr-PCA': corr_pca,
            'Corr-Weighted': corr_weighted,
        }
    
    @staticmethod
    def _rank_diff(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """Spearman correlation (rank-based)."""
        correlations = []
        for col in df1.columns:
            if col in df2.columns:
                paired = pd.concat([df1[col], df2[col]], axis=1).dropna()
                if len(paired) > 1:
                    corr = paired.iloc[:, 0].corr(paired.iloc[:, 1], method='spearman')
                    correlations.append(corr)
        return np.nanmean(correlations) if correlations else np.nan
    
    @staticmethod
    def _corr_diff(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """Pearson correlation"""
        correlations = []
        for col in df1.columns:
            if col in df2.columns:
                paired = pd.concat([df1[col], df2[col]], axis=1).dropna()
                if len(paired) > 1:
                    corr = paired.iloc[:, 0].corr(paired.iloc[:, 1], method='pearson')
                    correlations.append(corr)
        return np.nanmean(correlations) if correlations else np.nan
    
    @staticmethod
    def _weighted_pearson(x: np.ndarray, y: np.ndarray, distances: np.ndarray) -> float:
        """Weighted Pearson correlation."""
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = 1 / distances
            weights[distances == 0] = 0
        
        if np.sum(weights) == 0:
            return np.nan
        
        weights = weights / np.sum(weights)
        
        weighted_mean_x = np.average(x, weights=weights)
        weighted_mean_y = np.average(y, weights=weights)
        
        covariance = np.sum(weights * (x - weighted_mean_x) * (y - weighted_mean_y))
        variance_x = np.sum(weights * (x - weighted_mean_x) ** 2)
        variance_y = np.sum(weights * (y - weighted_mean_y) ** 2)
        
        if variance_x * variance_y == 0:
            return np.nan
        
        return covariance / np.sqrt(variance_x * variance_y)
    
    def _corrw_diff(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """Weighted Pearson correlation."""
        correlations = []
        for col in df1.columns:
            if col in df2.columns:
                paired = pd.concat([df1[col], df2[col]], axis=1).dropna()
                if len(paired) > 1:
                    corr = self._weighted_pearson(
                        paired.iloc[:, 0].values,
                        paired.iloc[:, 1].values,
                        paired.iloc[:, 1].values,
                    )
                    correlations.append(corr)
        return np.nanmean(correlations) if correlations else np.nan
    
    def run(self) -> pd.DataFrame:
        """Run evaluation on all embeddings.
        
        Returns
        -------
        Results DataFrame (embedding x metrics)
        """
        if not self._prepared:
            self.prepare()
        
        results = {}
        for emb_key in tqdm(self.embedding_keys, desc="Evaluating embeddings"):
            results[emb_key] = self._evaluate_embedding(emb_key)
        
        self._results = pd.DataFrame(results).T
        self._results.index.name = 'Embedding'
        return self._results
    
    def get_results(self) -> pd.DataFrame:
        """Return results DataFrame."""
        if self._results is None:
            return self.run()
        return self._results
