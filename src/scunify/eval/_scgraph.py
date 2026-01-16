"""
scGraph 래퍼 (Islander 기반)
Consensus distance 기반 embedding 품질 평가

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
    """scGraph: Consensus distance 기반 embedding 평가
    
    각 batch별로 cell type centroid를 계산하고,
    consensus distance matrix와 embedding의 거리를 비교하여
    embedding 품질을 평가합니다.
    
    Parameters
    ----------
    adata
        AnnData 객체 (embedding이 obsm에 저장되어 있어야 함)
    embedding_keys
        평가할 embedding obsm 키 리스트
    batch_key
        batch 정보가 있는 obs 컬럼명
    label_key
        cell type 라벨이 있는 obs 컬럼명
    trim_rate
        trimmed mean 계산시 양쪽에서 제거할 비율 (기본값: 0.05)
    thres_batch
        최소 batch 크기 (이보다 작은 batch는 제외)
    thres_celltype
        최소 cell type 크기 (이보다 작은 cell type은 제외)
    
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
        
        # 제외할 cell types
        self._ignore_celltypes: list[str] = []
        # batch별 consensus distance
        self._collect_pca: dict[str, pd.DataFrame] = {}
        # consensus distance matrix
        self._consensus_df: pd.DataFrame | None = None
        
        self._prepared = False
        self._results: pd.DataFrame | None = None
    
    def _preprocess(self) -> None:
        """cell type 필터링 (너무 작은 cell type 제외)"""
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
        """각 cell type별 trimmed mean centroid 계산
        
        Parameters
        ----------
        X
            embedding matrix (n_cells, n_dims)
        labels
            cell type labels
        trim_proportion
            양쪽에서 제거할 비율
        
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
        """cell type 간 pairwise distance 계산
        
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
        """각 batch별로 centroid 및 pairwise distance 계산"""
        print("Processing batches: calculating centroids and pairwise distances...")
        
        # PCA 기반 consensus 계산
        import scanpy as sc
        
        for batch in tqdm(self.adata.obs[self.batch_key].unique()):
            adata_batch = self.adata[self.adata.obs[self.batch_key] == batch].copy()
            
            if len(adata_batch) < self.thres_batch:
                print(f"Skipped batch '{batch}': < {self.thres_batch} cells")
                continue
            
            # HVG + PCA 계산
            try:
                sc.pp.highly_variable_genes(adata_batch, n_top_genes=min(1000, adata_batch.n_vars))
                sc.pp.pca(adata_batch, n_comps=min(10, adata_batch.n_obs - 1), use_highly_variable=True)
            except Exception as e:
                print(f"Skipped batch '{batch}': PCA failed ({e})")
                continue
            
            # Centroid 및 distance 계산
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
        """batch별 distance를 평균하여 consensus distance matrix 생성"""
        if not self._collect_pca:
            raise ValueError("No batches processed. Run _process_batches first.")
        
        # 모든 batch의 distance matrix 병합
        df_combined = pd.concat(self._collect_pca.values(), axis=0, sort=False)
        # 동일 index끼리 평균
        consensus = df_combined.groupby(df_combined.index).mean()
        # 대칭 행렬로 정리
        common_labels = consensus.index.intersection(consensus.columns)
        consensus = consensus.loc[common_labels, common_labels]
        # Normalize by max
        self._consensus_df = consensus / consensus.max(axis=0)
    
    def prepare(self) -> "ScGraphWrapper":
        """전처리 및 consensus distance 계산
        
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
        """단일 embedding 평가
        
        Parameters
        ----------
        obsm_key
            embedding obsm 키
        
        Returns
        -------
        메트릭 dict: {'Rank-PCA': ..., 'Corr-PCA': ..., 'Corr-Weighted': ...}
        """
        if self._consensus_df is None:
            raise ValueError("Consensus not calculated. Run prepare() first.")
        
        # Embedding centroid 및 distance 계산
        centroids = self._calculate_trimmed_means(
            np.array(self.adata.obsm[obsm_key]),
            self.adata.obs[self.label_key],
            trim_proportion=self.trim_rate,
        )
        
        # Consensus와 공통 cell type만 사용
        common_labels = centroids.index.intersection(self._consensus_df.index)
        if len(common_labels) < 2:
            return {'Rank-PCA': np.nan, 'Corr-PCA': np.nan, 'Corr-Weighted': np.nan}
        
        centroids = centroids.loc[common_labels]
        consensus = self._consensus_df.loc[common_labels, common_labels]
        
        pairwise_dist = self._compute_pairwise_distances(centroids)
        pairwise_dist = pairwise_dist.loc[common_labels, common_labels]
        normalized = pairwise_dist.div(pairwise_dist.max(axis=0), axis=1)
        
        # 메트릭 계산
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
        """Spearman correlation (순위 기반)"""
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
        """가중 Pearson correlation"""
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
        """가중 Pearson correlation"""
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
        """모든 embedding 평가 실행
        
        Returns
        -------
        결과 DataFrame (embedding x metrics)
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
        """결과 DataFrame 반환"""
        if self._results is None:
            return self.run()
        return self._results
