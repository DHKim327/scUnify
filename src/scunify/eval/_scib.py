"""
scib-metrics wrapper
https://scib-metrics.readthedocs.io/en/stable/
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pandas as pd

# JAX memory settings (prevent GPU OOM)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

if TYPE_CHECKING:
    from anndata import AnnData

# Lazy imports to avoid loading dependencies at import time
_SCIB_METRICS_IMPORTED = False
_Benchmarker = None
_BioConservation = None
_BatchCorrection = None


def _ensure_scib_metrics():
    global _SCIB_METRICS_IMPORTED, _Benchmarker, _BioConservation, _BatchCorrection
    if not _SCIB_METRICS_IMPORTED:
        from scib_metrics.benchmark import BatchCorrection, BioConservation, Benchmarker
        _Benchmarker = Benchmarker
        _BioConservation = BioConservation
        _BatchCorrection = BatchCorrection
        _SCIB_METRICS_IMPORTED = True


def _validate_batch_info(adata: "AnnData", batch_key: str, label_key: str) -> bool:
    """Validate whether batch information is valid for metric computation.
    
    Returns
    -------
    bool
        True: batch info is valid (metrics can be computed)
        False: batch info is invalid (metrics cannot be computed)
    """
    # Return False if batch_key is missing
    if batch_key not in adata.obs.columns:
        print(f"[WARNING] batch_key '{batch_key}' not found in adata.obs")
        return False
    
    # Batch correction metrics cannot be computed if batch == label
    if batch_key == label_key:
        print(f"[WARNING] batch_key and label_key are identical. Disabling batch correction metrics.")
        return False
    
    # Cannot compute if batch and label values are identical
    batch_values = set(adata.obs[batch_key].unique())
    label_values = set(adata.obs[label_key].unique())
    
    if batch_values == label_values:
        # Additional check: verify per-cell identity
        if (adata.obs[batch_key] == adata.obs[label_key]).all():
            print(f"[WARNING] batch and label values are identical. Disabling batch correction metrics.")
            return False
    
    # Need at least 2 batches per label for BRAS computation
    min_batches_per_label = float('inf')
    for label in label_values:
        mask = adata.obs[label_key] == label
        n_batches = adata.obs.loc[mask, batch_key].nunique()
        min_batches_per_label = min(min_batches_per_label, n_batches)
    
    if min_batches_per_label < 2:
        print(f"[WARNING] Some cell types have only 1 batch. BRAS metrics may fail.")
        return False
    
    return True


class ScibWrapper:
    """scib-metrics Benchmarker wrapper.
    
    Parameters
    ----------
    adata
        AnnData object (cell x gene)
    embedding_keys
        List of embedding obsm keys to evaluate (e.g., ["X_scgpt", "X_uce"])
    batch_key
        obs column name containing batch information
    label_key
        obs column name containing cell-type labels
    bio_metrics
        Bio conservation metric settings (None for defaults)
    batch_metrics
        Batch correction metric settings (None for defaults, "auto" for auto-detection)
    n_jobs
        Number of parallel jobs for neighbor computation
    
    Examples
    --------
    >>> wrapper = ScibWrapper(
    ...     adata,
    ...     embedding_keys=["X_scgpt", "X_uce"],
    ...     batch_key="batch",
    ...     label_key="cell_type",
    ... )
    >>> results = wrapper.run()
    >>> wrapper.plot(save_dir="./results")
    """
    
    def __init__(
        self,
        adata: "AnnData",
        embedding_keys: list[str],
        batch_key: str,
        label_key: str,
        bio_metrics: "BioConservation | None" = None,
        batch_metrics: "BatchCorrection | str | None" = "auto",
        n_jobs: int = -1,
    ):
        _ensure_scib_metrics()
        
        self.adata = adata
        self.embedding_keys = embedding_keys
        self.batch_key = batch_key
        self.label_key = label_key
        self.n_jobs = n_jobs
        
        # Validate batch information
        self._batch_valid = _validate_batch_info(adata, batch_key, label_key)
        
        # Default bio conservation metric settings
        if bio_metrics is None:
            bio_metrics = _BioConservation(
                isolated_labels=True,
                nmi_ari_cluster_labels_leiden=True,
                nmi_ari_cluster_labels_kmeans=True,
                silhouette_label=True,
                clisi_knn=True,
            )
        
        # Batch correction metric settings (auto-detect or manual)
        if batch_metrics == "auto":
            if self._batch_valid:
                print("[INFO] Batch info valid. Enabling all batch correction metrics.")
                batch_metrics = _BatchCorrection(
                    bras=True,
                    ilisi_knn=True,
                    kbet_per_label=True,
                    graph_connectivity=True,
                    pcr_comparison=True,
                )
            else:
                print("[INFO] Batch info invalid. Disabling batch correction metrics (bio conservation only).")
                batch_metrics = _BatchCorrection(
                    bras=False,
                    ilisi_knn=False,
                    kbet_per_label=False,
                    graph_connectivity=False,
                    pcr_comparison=False,
                )
        elif batch_metrics is None:
            batch_metrics = _BatchCorrection(
                bras=True,
                ilisi_knn=True,
                kbet_per_label=True,
                graph_connectivity=True,
                pcr_comparison=True,
            )
        
        self.bio_metrics = bio_metrics
        self.batch_metrics = batch_metrics
        
        # Initialize Benchmarker
        self.benchmarker = _Benchmarker(
            adata,
            batch_key=batch_key,
            label_key=label_key,
            embedding_obsm_keys=embedding_keys,
            bio_conservation_metrics=bio_metrics,
            batch_correction_metrics=batch_metrics,
            n_jobs=n_jobs,
        )
        
        self._prepared = False
        self._benchmarked = False
        self._results = None
    
    def prepare(self, neighbor_computer=None) -> "ScibWrapper":
        """Compute neighbors (prepare step).
        
        Parameters
        ----------
        neighbor_computer
            Custom neighbor computation function (None uses pynndescent)
        
        Returns
        -------
        self
        """
        self.benchmarker.prepare(neighbor_computer=neighbor_computer)
        self._prepared = True
        return self
    
    def benchmark(self) -> "ScibWrapper":
        """Run metric computation.
        
        Returns
        -------
        self
        """
        if not self._prepared:
            self.prepare()
        self.benchmarker.benchmark()
        self._benchmarked = True
        return self
    
    def run(self, min_max_scale: bool = False) -> pd.DataFrame:
        """Run prepare + benchmark + get_results in one call.
        
        Parameters
        ----------
        min_max_scale
            Whether to scale results to 0-1
        
        Returns
        -------
        Results DataFrame
        """
        if not self._benchmarked:
            self.benchmark()
        self._results = self.benchmarker.get_results(min_max_scale=min_max_scale)
        return self._results
    
    def get_results(self, min_max_scale: bool = False) -> pd.DataFrame:
        """Return results DataFrame.
        
        Parameters
        ----------
        min_max_scale
            Whether to scale results to 0-1
        
        Returns
        -------
        Results DataFrame
        """
        if self._results is None:
            return self.run(min_max_scale=min_max_scale)
        return self._results
    
    def plot(
        self,
        min_max_scale: bool = False,
        show: bool = True,
        save_dir: str | None = None,
    ):
        """Plot scIB-style results table.
        
        Parameters
        ----------
        min_max_scale
            Whether to scale results to 0-1
        show
            Whether to display the plot on screen
        save_dir
            Save directory (None to skip saving)
        
        Returns
        -------
        plottable.Table object
        """
        if not self._benchmarked:
            self.benchmark()
        return self.benchmarker.plot_results_table(
            min_max_scale=min_max_scale,
            show=show,
            save_dir=save_dir,
        )
