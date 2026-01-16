"""
scib-metrics 래퍼
https://scib-metrics.readthedocs.io/en/stable/
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pandas as pd

# JAX 메모리 설정 (GPU 메모리 부족 방지)
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
    """batch 정보가 유효한지 검증
    
    Returns
    -------
    bool
        True: batch 정보 유효 (메트릭 계산 가능)
        False: batch 정보 무효 (메트릭 계산 불가)
    """
    # batch_key가 없으면 False
    if batch_key not in adata.obs.columns:
        print(f"[WARNING] batch_key '{batch_key}' not found in adata.obs")
        return False
    
    # batch와 label이 동일하면 batch correction 메트릭 계산 불가
    if batch_key == label_key:
        print(f"[WARNING] batch_key와 label_key가 동일합니다. Batch correction 메트릭을 비활성화합니다.")
        return False
    
    # batch 값과 label 값이 완전히 동일하면 계산 불가
    batch_values = set(adata.obs[batch_key].unique())
    label_values = set(adata.obs[label_key].unique())
    
    if batch_values == label_values:
        # 추가 검증: 각 cell에 대해 실제로 동일한지 확인
        if (adata.obs[batch_key] == adata.obs[label_key]).all():
            print(f"[WARNING] batch와 label 값이 동일합니다. Batch correction 메트릭을 비활성화합니다.")
            return False
    
    # 각 label 내에 최소 2개 이상의 batch가 있어야 BRAS 계산 가능
    min_batches_per_label = float('inf')
    for label in label_values:
        mask = adata.obs[label_key] == label
        n_batches = adata.obs.loc[mask, batch_key].nunique()
        min_batches_per_label = min(min_batches_per_label, n_batches)
    
    if min_batches_per_label < 2:
        print(f"[WARNING] 일부 cell type 내에 batch가 1개뿐입니다. BRAS 메트릭이 실패할 수 있습니다.")
        return False
    
    return True


class ScibWrapper:
    """scib-metrics Benchmarker 래퍼
    
    Parameters
    ----------
    adata
        AnnData 객체 (cell x gene)
    embedding_keys
        평가할 embedding obsm 키 리스트 (예: ["X_scgpt", "X_uce"])
    batch_key
        batch 정보가 있는 obs 컬럼명
    label_key
        cell type 라벨이 있는 obs 컬럼명
    bio_metrics
        Bio conservation 메트릭 설정 (None이면 기본값)
    batch_metrics
        Batch correction 메트릭 설정 (None이면 기본값, "auto"면 자동 감지)
    n_jobs
        neighbor 계산 병렬화 수
    
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
        
        # batch 정보 유효성 검증
        self._batch_valid = _validate_batch_info(adata, batch_key, label_key)
        
        # 기본 Bio conservation 메트릭 설정
        if bio_metrics is None:
            bio_metrics = _BioConservation(
                isolated_labels=True,
                nmi_ari_cluster_labels_leiden=True,
                nmi_ari_cluster_labels_kmeans=True,
                silhouette_label=True,
                clisi_knn=True,
            )
        
        # Batch correction 메트릭 설정 (자동 감지 또는 수동 설정)
        if batch_metrics == "auto":
            if self._batch_valid:
                print("[INFO] Batch 정보 유효. 모든 batch correction 메트릭 활성화.")
                batch_metrics = _BatchCorrection(
                    bras=True,
                    ilisi_knn=True,
                    kbet_per_label=True,
                    graph_connectivity=True,
                    pcr_comparison=True,
                )
            else:
                print("[INFO] Batch 정보 무효. Batch correction 메트릭 비활성화 (Bio conservation만 실행).")
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
        
        # Benchmarker 초기화
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
        """neighbors 계산 (prepare 단계)
        
        Parameters
        ----------
        neighbor_computer
            커스텀 neighbor 계산 함수 (None이면 pynndescent 사용)
        
        Returns
        -------
        self
        """
        self.benchmarker.prepare(neighbor_computer=neighbor_computer)
        self._prepared = True
        return self
    
    def benchmark(self) -> "ScibWrapper":
        """메트릭 계산 실행
        
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
        """prepare + benchmark + get_results 한번에 실행
        
        Parameters
        ----------
        min_max_scale
            결과를 0-1로 스케일링할지 여부
        
        Returns
        -------
        결과 DataFrame
        """
        if not self._benchmarked:
            self.benchmark()
        self._results = self.benchmarker.get_results(min_max_scale=min_max_scale)
        return self._results
    
    def get_results(self, min_max_scale: bool = False) -> pd.DataFrame:
        """결과 DataFrame 반환
        
        Parameters
        ----------
        min_max_scale
            결과를 0-1로 스케일링할지 여부
        
        Returns
        -------
        결과 DataFrame
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
        """scib 스타일 테이블 플로팅
        
        Parameters
        ----------
        min_max_scale
            결과를 0-1로 스케일링할지 여부
        show
            플롯을 화면에 표시할지 여부
        save_dir
            저장 디렉토리 (None이면 저장 안함)
        
        Returns
        -------
        plottable.Table 객체
        """
        if not self._benchmarked:
            self.benchmark()
        return self.benchmarker.plot_results_table(
            min_max_scale=min_max_scale,
            show=show,
            save_dir=save_dir,
        )
