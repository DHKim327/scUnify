# core/data_actor.py
"""
모델별 DataLoader Actor

각 모델 환경(scunify_scgpt, scunify_uce, scunify_scfoundation)에서 실행되는
Stateful Actor로, 같은 환경의 Worker들이 데이터를 zero-copy로 공유할 수 있게 함.

주요 특징:
  - 같은 경로의 데이터는 한 번만 로드
  - 같은 환경 내에서 직렬화 호환 보장
  - Object Store를 통한 zero-copy 공유
"""

import ray


@ray.remote
class DataLoaderActor:
    """
    특정 모델 환경에서 실행되는 데이터 로더 Actor
    
    Usage:
        # 모델별 Actor 생성 (특정 conda 환경에서 실행)
        actor = DataLoaderActor.options(
            runtime_env={"conda": "scunify_scgpt"}
        ).remote()
        
        # 데이터 로드 요청
        adata_ref = ray.get(actor.get_or_load.remote("/path/to/data.h5ad"))
        
        # Worker에서 데이터 사용
        adata = ray.get(adata_ref)
    """
    
    def __init__(self):
        import os
        # matplotlib backend 설정 (Jupyter inline backend 충돌 방지)
        # Actor 시작 시 즉시 설정해야 함
        os.environ["MPLBACKEND"] = "Agg"
        
        self.cache = {}  # {path: ObjectRef}
        self._loaded_paths = []  # 로딩 순서 기록
    
    def get_or_load(self, path: str):
        """
        경로에서 AnnData를 로드하고 Object Store에 저장
        
        Args:
            path: h5ad 파일 경로
            
        Returns:
            ray.ObjectRef: Object Store에 저장된 AnnData의 참조
        """
        import os
        # matplotlib backend 설정 (Jupyter inline backend 충돌 방지)
        os.environ["MPLBACKEND"] = "Agg"
        import matplotlib
        matplotlib.use("Agg")
        
        import numpy as np
        import scanpy as sc
        import scipy.sparse as sp
        
        if path not in self.cache:
            # 새로운 데이터 로드
            adata = sc.read_h5ad(path)
            
            # float32 변환 (메모리 효율 + 모델 호환성)
            if sp.issparse(adata.X):
                adata.X = adata.X.astype(np.float32)
            else:
                adata.X = np.asarray(adata.X, dtype=np.float32, order="C")
            
            # Object Store에 저장
            self.cache[path] = ray.put(adata)
            self._loaded_paths.append(path)
        
        return self.cache[path]
    
    def preload(self, paths: list[str]):
        """
        여러 경로를 미리 로드
        
        Args:
            paths: h5ad 파일 경로 리스트
            
        Returns:
            dict: {path: ObjectRef} 매핑
        """
        result = {}
        for path in paths:
            result[path] = self.get_or_load(path)
        return result
    
    def clear_cache(self, path: str = None):
        """
        캐시 정리
        
        Args:
            path: 특정 경로만 정리 (None이면 전체)
        """
        if path:
            self.cache.pop(path, None)
            if path in self._loaded_paths:
                self._loaded_paths.remove(path)
        else:
            self.cache.clear()
            self._loaded_paths.clear()
    
    def get_cache_info(self) -> dict:
        """
        캐시 상태 정보 반환
        """
        return {
            "n_cached": len(self.cache),
            "paths": list(self.cache.keys()),
            "load_order": self._loaded_paths.copy(),
        }
    
    def is_cached(self, path: str) -> bool:
        """
        특정 경로가 캐시되어 있는지 확인
        """
        return path in self.cache


def create_model_actors(model_names: list[str]) -> dict:
    """
    모델별 DataLoader Actor 생성
    
    Args:
        model_names: 모델 이름 리스트 (예: ["scGPT", "UCE", "scFoundation"])
        
    Returns:
        dict: {model_name: DataLoaderActor} 매핑
    """
    actors = {}
    
    for model_name in model_names:
        env_name = f"scunify_{model_name.lower()}"
        
        # 모델 환경에서 실행되는 Actor 생성
        actor = DataLoaderActor.options(
            runtime_env={"conda": env_name},
            name=f"data_loader_{model_name.lower()}",
            lifetime="detached",  # Runner 종료 후에도 유지 가능
        ).remote()
        
        actors[model_name] = actor
    
    return actors
