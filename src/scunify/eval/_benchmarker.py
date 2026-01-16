"""
scUnify 통합 평가 프레임워크
scib-metrics + scGraph 메트릭 통합
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._scib import ScibWrapper
from ._scgraph import ScGraphWrapper

if TYPE_CHECKING:
    from anndata import AnnData
    from ..config import ScUnifyConfig


class Evaluator:
    """scib + scgraph 통합 평가
    
    Foundation Model embedding의 품질을 scib-metrics와 scgraph 메트릭으로
    종합적으로 평가합니다.
    
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
    n_jobs
        scib neighbor 계산 병렬화 수 (기본값: -1, 모든 코어 사용)
    
    Examples
    --------
    >>> import scunify as scu
    >>> 
    >>> evaluator = scu.eval.Evaluator(
    ...     adata,
    ...     embedding_keys=["X_scgpt", "X_uce", "X_scfoundation"],
    ...     batch_key="batch",
    ...     label_key="cell_type",
    ... )
    >>> 
    >>> # scib 메트릭만 실행
    >>> scib_results = evaluator.run_scib()
    >>> 
    >>> # scgraph 메트릭만 실행
    >>> scgraph_results = evaluator.run_scgraph()
    >>> 
    >>> # 모든 메트릭 실행
    >>> all_results = evaluator.run_all()
    >>> 
    >>> # 이쁜 플로팅
    >>> evaluator.plot_results(save_dir="./results")
    """
    
    def __init__(
        self,
        adata: "AnnData",
        embedding_keys: list[str],
        batch_key: str,
        label_key: str,
        n_jobs: int = -1,
    ):
        self.adata = adata
        self.embedding_keys = embedding_keys
        self.batch_key = batch_key
        self.label_key = label_key
        self.n_jobs = n_jobs
        
        # Lazy 초기화
        self._scib: ScibWrapper | None = None
        self._scgraph: ScGraphWrapper | None = None
        
        self._scib_results: pd.DataFrame | None = None
        self._scgraph_results: pd.DataFrame | None = None
        self._combined_results: pd.DataFrame | None = None
    
    @property
    def scib(self) -> ScibWrapper:
        """ScibWrapper 인스턴스 (lazy 생성)"""
        if self._scib is None:
            self._scib = ScibWrapper(
                self.adata,
                embedding_keys=self.embedding_keys,
                batch_key=self.batch_key,
                label_key=self.label_key,
                n_jobs=self.n_jobs,
            )
        return self._scib
    
    @property
    def scgraph(self) -> ScGraphWrapper:
        """ScGraphWrapper 인스턴스 (lazy 생성)"""
        if self._scgraph is None:
            self._scgraph = ScGraphWrapper(
                self.adata,
                embedding_keys=self.embedding_keys,
                batch_key=self.batch_key,
                label_key=self.label_key,
            )
        return self._scgraph
    
    def run_scib(self, min_max_scale: bool = False) -> pd.DataFrame:
        """scib-metrics 평가 실행
        
        Parameters
        ----------
        min_max_scale
            결과를 0-1로 스케일링할지 여부
        
        Returns
        -------
        scib 결과 DataFrame
        """
        self._scib_results = self.scib.run(min_max_scale=min_max_scale)
        return self._scib_results
    
    def run_scgraph(self) -> pd.DataFrame:
        """scGraph 평가 실행
        
        Returns
        -------
        scgraph 결과 DataFrame
        """
        self._scgraph_results = self.scgraph.run()
        return self._scgraph_results
    
    def run_all(self, min_max_scale: bool = False) -> pd.DataFrame:
        """scib + scgraph 메트릭 모두 계산
        
        Parameters
        ----------
        min_max_scale
            결과를 0-1로 스케일링할지 여부
        
        Returns
        -------
        통합 결과 DataFrame
        """
        print("=" * 50)
        print("Running scib-metrics evaluation...")
        print("=" * 50)
        scib_df = self.run_scib(min_max_scale=min_max_scale)
        
        print("\n" + "=" * 50)
        print("Running scGraph evaluation...")
        print("=" * 50)
        scgraph_df = self.run_scgraph()
        
        self._combined_results = self._merge_results(scib_df, scgraph_df)
        return self._combined_results
    
    def _merge_results(
        self,
        scib_df: pd.DataFrame,
        scgraph_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """scib와 scgraph 결과 병합
        
        최종 결과 형태: (embeddings x metrics)
        - 행: embedding 이름 (X_scgpt, X_uce, ...)
        - 열: 모든 메트릭 (scib + scgraph)
        
        Parameters
        ----------
        scib_df
            scib 결과 DataFrame (embeddings x metrics)
        scgraph_df
            scgraph 결과 DataFrame (embeddings x metrics)
        
        Returns
        -------
        병합된 DataFrame (embeddings x all_metrics)
        """
        # scib 결과에서 Metric Type 행 제거 (있는 경우)
        if 'Metric Type' in scib_df.index:
            scib_df = scib_df.drop('Metric Type')
        
        # scib-metrics의 get_results()는 이미 (embeddings x metrics) 형태
        # scgraph도 (embeddings x metrics) 형태
        
        # 공통 embedding 확인
        common_embeddings = scib_df.index.intersection(scgraph_df.index)
        
        if len(common_embeddings) > 0:
            scib_subset = scib_df.loc[common_embeddings]
            scgraph_subset = scgraph_df.loc[common_embeddings]
            # 열 방향으로 병합 (같은 embedding에 대해 metric 열을 합침)
            combined = pd.concat([scib_subset, scgraph_subset], axis=1)
        else:
            # 디버깅: 인덱스 출력
            print(f"[DEBUG] scib_df.index: {list(scib_df.index)}")
            print(f"[DEBUG] scgraph_df.index: {list(scgraph_df.index)}")
            # 그냥 합침 (인덱스가 다른 경우)
            combined = pd.concat([scib_df, scgraph_df], axis=1)
        
        # 중복 열 제거
        combined = combined.loc[:, ~combined.columns.duplicated()]
        
        combined.index.name = 'Embedding'
        
        return combined
    
    def get_results(self, min_max_scale: bool = False) -> pd.DataFrame:
        """결과 DataFrame 반환
        
        Parameters
        ----------
        min_max_scale
            결과를 0-1로 스케일링할지 여부
        
        Returns
        -------
        통합 결과 DataFrame
        """
        if self._combined_results is None:
            return self.run_all(min_max_scale=min_max_scale)
        return self._combined_results
    
    def plot_scib(
        self,
        min_max_scale: bool = False,
        show: bool = True,
        save_dir: str | None = None,
    ):
        """scib 결과만 플로팅
        
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
        return self.scib.plot(
            min_max_scale=min_max_scale,
            show=show,
            save_dir=save_dir,
        )
    
    def plot_results(
        self,
        min_max_scale: bool = False,
        show: bool = True,
        save_dir: str | None = None,
    ):
        """scib + scgraph 통합 결과 플로팅
        
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
        from ._plotting import plot_combined_table
        
        if self._combined_results is None:
            self.run_all(min_max_scale=min_max_scale)
        
        return plot_combined_table(
            self._combined_results,
            save_dir=save_dir,
            min_max_scale=min_max_scale,
            show=show,
        )
    
    def save_results(self, path: str) -> None:
        """결과를 CSV로 저장
        
        Parameters
        ----------
        path
            저장 경로 (.csv)
        """
        if self._combined_results is None:
            self.run_all()
        self._combined_results.to_csv(path)
        print(f"Results saved to {path}")
    
    @classmethod
    def from_tasks(
        cls,
        tasks: list["ScUnifyConfig"],
        batch_key: str,
        label_key: str,
        result_dir: str | Path | None = None,
        n_jobs: int = -1,
    ) -> dict[str, "Evaluator"]:
        """ScUnifyRunner의 tasks로부터 Evaluator 생성
        
        동일한 adata_path를 가진 task들을 그룹화하고,
        각 모델의 결과를 adata.obsm['X_{model_name}']에서 로드합니다.
        
        Parameters
        ----------
        tasks
            ScUnifyConfig 리스트
        batch_key
            batch 정보가 있는 obs 컬럼명
        label_key
            cell type 라벨이 있는 obs 컬럼명
        result_dir
            결과가 저장된 디렉토리 (None이면 각 task의 save_dir 사용)
        n_jobs
            scib neighbor 계산 병렬화 수
        
        Returns
        -------
        dict[str, Evaluator]
            adata_path -> Evaluator 매핑
        
        Examples
        --------
        >>> tasks = [
        ...     ScUnifyConfig("PBMC3k.h5ad", "scgpt.yaml"),
        ...     ScUnifyConfig("PBMC3k.h5ad", "uce.yaml"),
        ...     ScUnifyConfig("PBMC3k.h5ad", "scfoundation.yaml"),
        ... ]
        >>> 
        >>> evaluators = Evaluator.from_tasks(
        ...     tasks,
        ...     batch_key="batch",
        ...     label_key="cell_type",
        ... )
        >>> 
        >>> for adata_name, evaluator in evaluators.items():
        ...     print(f"Evaluating {adata_name}...")
        ...     results = evaluator.run_all()
        ...     evaluator.plot_results(save_dir=f"./results/{adata_name}")
        """
        import scanpy as sc
        
        # 1. 동일 adata_path끼리 그룹화
        groups: dict[str, list] = defaultdict(list)
        for task in tasks:
            adata_key = str(task.adata_dir)
            groups[adata_key].append(task)
        
        print(f"Found {len(groups)} unique adata files:")
        for k, v in groups.items():
            print(f"  - {k}: {len(v)} models")
        
        # 2. 각 그룹에 대해 Evaluator 생성
        evaluators = {}
        
        for adata_path, task_list in groups.items():
            print(f"\n{'='*50}")
            print(f"Loading {adata_path}...")
            print(f"{'='*50}")
            
            # AnnData 로드
            adata = sc.read_h5ad(adata_path)
            
            # 각 task의 결과 (.npy) 로드하여 obsm에 추가
            embedding_keys = []
            for task in task_list:
                model_name = task.model_name.lower()
                obsm_key = f"X_{model_name}"
                # 결과 파일 경로 결정
                if result_dir:
                    npy_path = Path(result_dir) / f"{task.task_name}.npy"
                else:
                    npy_path = None
                
                if npy_path is not None:
                    print(f"  Loading {obsm_key} from {npy_path}")
                    embedding = np.load(npy_path)
                    adata.obsm[obsm_key] = embedding
                    embedding_keys.append(obsm_key)
                elif obsm_key in adata.obsm:
                    print(f"  Found {obsm_key} in adata.obsm")
                    embedding_keys.append(obsm_key)
                else:
                    print(f"  [WARNING] {obsm_key} not found!")
            
            if not embedding_keys:
                print(f"  [ERROR] No embeddings found for {adata_path}, skipping...")
                continue
            
            print(f"  Embedding keys: {embedding_keys}")
            
            # Evaluator 생성
            adata_name = Path(adata_path).stem
            evaluators[adata_name] = cls(
                adata=adata,
                embedding_keys=embedding_keys,
                batch_key=batch_key,
                label_key=label_key,
                n_jobs=n_jobs,
            )
        
        return evaluators
    
    @classmethod
    def from_adata(
        cls,
        adata: "AnnData",
        batch_key: str,
        label_key: str,
        embedding_prefix: str = "X_",
        n_jobs: int = -1,
    ) -> "Evaluator":
        """AnnData에서 자동으로 embedding 키 감지하여 Evaluator 생성
        
        Parameters
        ----------
        adata
            AnnData 객체
        batch_key
            batch 정보가 있는 obs 컬럼명
        label_key
            cell type 라벨이 있는 obs 컬럼명
        embedding_prefix
            embedding obsm 키 prefix (기본값: "X_")
        n_jobs
            scib neighbor 계산 병렬화 수
        
        Returns
        -------
        Evaluator 인스턴스
        
        Examples
        --------
        >>> # adata.obsm에 X_scgpt, X_uce, X_scfoundation이 있다고 가정
        >>> evaluator = Evaluator.from_adata(
        ...     adata,
        ...     batch_key="batch",
        ...     label_key="cell_type",
        ... )
        >>> results = evaluator.run_all()
        """
        # embedding_prefix로 시작하는 키 중 FM 관련 키만 선택
        fm_keywords = ["scgpt", "uce", "scfoundation", "geneformer", "cellbert"]
        
        embedding_keys = []
        for key in adata.obsm.keys():
            if key.startswith(embedding_prefix):
                model_name = key[len(embedding_prefix):].lower()
                if any(kw in model_name for kw in fm_keywords):
                    embedding_keys.append(key)
        
        if not embedding_keys:
            # FM 키워드가 없으면 prefix로 시작하는 모든 키 사용
            embedding_keys = [k for k in adata.obsm.keys() 
                            if k.startswith(embedding_prefix) 
                            and k not in ["X_pca", "X_umap", "X_tsne"]]
        
        print(f"Auto-detected embedding keys: {embedding_keys}")
        
        return cls(
            adata=adata,
            embedding_keys=embedding_keys,
            batch_key=batch_key,
            label_key=label_key,
            n_jobs=n_jobs,
        )
