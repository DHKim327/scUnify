# core/Runner.py
from __future__ import annotations

import shutil
import time
import warnings
from pathlib import Path
from typing import Any
import ray
from ray.train import FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from ..utils import load_yaml
from .logger import ProgressActor, ProgressUI
from .data_actor import DataLoaderActor


warnings.filterwarnings("ignore")


_SYSTEM_PARAM_PATH = Path(__file__).resolve().parent.parent
_SYSTEM_PARAM_PATH = _SYSTEM_PARAM_PATH / "config/defaults/system.yaml"


class ScUnifyRunner:
    def __init__(
        self,
        tasks,
        total_cpus: int,
        total_gpus: int,
        per_task_gpus: int | list[int],
        per_task_cpus: int | list[int],
        gpu_indices: list[int] | None = None,
        *,
        verbose=True,
        mixed_precision: str | None = None,  # no | fp16 | bf16 | fp8
        ray_storage_dir: str | None = None,
        ray_temp_dir: str | None = None,
        auto_cleanup: bool = True,
    ):
        syscfg = load_yaml(_SYSTEM_PARAM_PATH)

        self.total_cpus = int(total_cpus)
        self.tasks = tasks

        self.gpu_indices = list(gpu_indices) if gpu_indices is not None else None
        visible_gpu_count = len(self.gpu_indices) if self.gpu_indices else (int(total_gpus) if total_gpus else 0)
        self.total_gpus = visible_gpu_count
        # GPU & CPU handling
        if isinstance(per_task_gpus, int):
            self.per_worker_gpus = [per_task_gpus] * len(self.tasks)
        elif isinstance(per_task_gpus, list):
            assert len(per_task_gpus) == len(self.tasks), "per_task_gpus 길이는 tasks 개수와 같아야 합니다."
            self.per_worker_gpus = [int(v) for v in per_task_gpus]
        else:
            self.per_worker_gpus = [int(syscfg["ray"]["per_worker"].get("GPU"))] * len(self.tasks)

        if isinstance(per_task_cpus, int):
            self.per_worker_cpus = [per_task_cpus] * len(self.tasks)
        elif isinstance(per_task_cpus, list):
            assert len(per_task_cpus) == len(self.tasks), "per_task_cpus 길이는 tasks 개수와 같아야 합니다."
            self.per_worker_cpus = [int(v) for v in per_task_cpus]
        else:
            self.per_worker_cpus = [int(syscfg["ray"]["per_worker"].get("CPU"))] * len(self.tasks)

        # accelerate
        self.accel_cfg = {
            "mixed_precision": mixed_precision or syscfg["accelerate"].get("mixed_precision", "fp16"),
            "cpu": bool(syscfg["accelerate"].get("cpu", False)),
        }

        storage_root = Path(ray_storage_dir).expanduser().resolve() if ray_storage_dir else None
        temp_root = Path(ray_temp_dir or "~/ray_temps").expanduser().resolve()
        self._storage_dir = storage_root
        self._temp_dir = temp_root

        # Other Configuration
        self.placement_strategy = syscfg["ray"].get("placement_strategy", "PACK")
        self.verbose = verbose

        # Ray 초기화
        self._auto_cleanup = bool(auto_cleanup)
        self._resource_check()
        self._initialize_ray()

    def _initialize_ray(self):
        import os
        # Worker Startup Timeout 증가 (24시간) - 대규모 데이터셋 처리 시 필요
        os.environ.setdefault("RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S", "86400")
        
        if ray.is_initialized():
            return
        init_kwargs = dict(ignore_reinit_error=True, local_mode=False)
        
        init_kwargs["_temp_dir"] = str(self._temp_dir)
        init_kwargs["num_cpus"] = self.total_cpus
        if self.gpu_indices:
            init_kwargs["num_gpus"] = len(self.gpu_indices)
        else:
            init_kwargs["num_gpus"] = self.total_gpus 
        
        # working_dir와 excludes는 ray.init에서 설정
        __PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
        
        env_vars = {
            "PYTHONPATH": __PROJECT_ROOT,
            "RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS": "0",
            "MPLBACKEND": "Agg",  # Headless backend for workers
        }
        
        if self.gpu_indices:
            env_vars["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_indices))
        
        excludes = [
            "resources/",
            "test/",
            ".git/",
            ".vscode/",
            ".ruff_cache/",
            "__pycache__/",
            "*.pyc",
            "*.pyo",
            "*.h5ad",
            "Foundations/",
        ]
        
        init_kwargs["runtime_env"] = {
            "working_dir": __PROJECT_ROOT,
            "excludes": excludes,
            "env_vars": env_vars,
        }
        
        ray.init(**init_kwargs)

    def _resource_check(self):
        per_job = []
        for g, c in zip(self.per_worker_gpus, self.per_worker_cpus):
            g = int(g)
            c = int(c)
            per_job.append({"gpus": g, "cpus": g * c, "cpu_per_worker": c})

            # [1] 잡 단위 초과 → 에러
            if self.total_gpus and g > self.total_gpus:
                raise ValueError(f"Job requires {g} GPUs but total_gpus is {self.total_gpus}.")
            if self.total_cpus and g * c > self.total_cpus:
                raise ValueError(f"Job requires {g * c} CPUs but total_cpus is {self.total_cpus}.")

        # [2] 총합 초과는 큐잉 → 워닝만
        need_gpus = sum(j["gpus"] for j in per_job)
        need_cpus = sum(j["cpus"] for j in per_job)

        if self.verbose:
            print("[RayRunner] Resource plan")
            for i, j in enumerate(per_job):
                print(
                    f"  - task[{i}]: num_workers={self.per_worker_gpus[i]} → needs GPU={j['gpus']}, CPU={j['cpus']} (per-worker CPU={j['cpu_per_worker']})"
                )
            print(
                f"  = TOTAL requested: GPU={need_gpus}, CPU={need_cpus} (limits: GPU={self.total_gpus}, CPU={self.total_cpus})"
            )

            if (self.total_gpus and need_gpus > self.total_gpus) or (self.total_cpus and need_cpus > self.total_cpus):
                print(
                    "[RayRunner] Note: Total requested > limits. Jobs will queue and run sequentially as resources free up."
                )

    def shutdown(self):
        # Actor 정리
        if hasattr(self, 'data_actors') and self.data_actors:
            if self.verbose:
                print("[Runner] Cleaning up DataLoader Actors...")
            for model_name, actor in self.data_actors.items():
                try:
                    ray.get(actor.clear_cache.remote())
                    ray.kill(actor)
                except Exception as e:
                    if self.verbose:
                        print(f"  - Failed to cleanup {model_name} actor: {e}")
            self.data_actors = {}
        
        if ray.is_initialized():
            ray.shutdown()

        if self._auto_cleanup:
            for p in [self._storage_dir, self._temp_dir]:
                try:
                    if p.exists():
                        shutil.rmtree(p, ignore_errors=True)
                except Exception as e:
                    print(f"[Runner.cleanup] Failed to remove {p}: {e}")

    def _create_data_actors(self) -> dict:
        """
        모델별 DataLoader Actor 생성
        
        각 모델 환경(scunify_scgpt, scunify_uce, scunify_scfoundation)에서
        별도의 Actor를 생성하여 같은 환경의 Worker들이 데이터를 공유할 수 있게 함.
        
        Returns:
            dict: {model_name: DataLoaderActor} 매핑
        """
        # 사용되는 모델 목록 추출
        model_names = set(t.get("model_name") for t in self.tasks)
        
        actors = {}
        for model_name in model_names:
            env_name = f"scunify_{model_name.lower()}"
            
            if self.verbose:
                print(f"[Runner] Creating DataLoader Actor for {model_name} (env: {env_name})...")
            
            # 모델 환경에서 실행되는 Actor 생성
            actor = DataLoaderActor.options(
                runtime_env={
                    "conda": env_name,
                    "env_vars": {"MPLBACKEND": "Agg"},  # matplotlib headless
                },
                name=f"data_loader_{model_name.lower()}",
            ).remote()
            
            actors[model_name] = actor
        
        return actors

    def _prepare_adata_payloads(self) -> float:
        """
        모델별 Actor를 통해 데이터 로드
        
        각 모델 환경의 Actor가 데이터를 로드하여 Object Store에 저장.
        같은 모델의 Worker들은 zero-copy로 데이터 공유.
        
        Returns:
            float: 데이터 로드 시간 (초)
        """
        t0 = time.time()
        
        # 모델별 Actor 생성
        self.data_actors = self._create_data_actors()
        
        # 모델별로 필요한 데이터 경로 그룹핑
        model_paths: dict[str, set[str]] = {}
        for t in self.tasks:
            model_name = t.get("model_name")
            path = str(t.adata_dir)
            if model_name not in model_paths:
                model_paths[model_name] = set()
            model_paths[model_name].add(path)
        
        # 각 Actor에서 데이터 프리로드
        preload_futures = {}
        for model_name, paths in model_paths.items():
            actor = self.data_actors[model_name]
            if self.verbose:
                print(f"[Runner] Preloading {len(paths)} dataset(s) for {model_name}...")
            preload_futures[model_name] = actor.preload.remote(list(paths))
        
        # 프리로드 완료 대기 및 결과 저장
        self.adata_refs: dict[str, dict[str, Any]] = {}
        for model_name, future in preload_futures.items():
            self.adata_refs[model_name] = ray.get(future)
            if self.verbose:
                print(f"[Runner] {model_name} data loaded!")
        
        # 각 task에 adata_ref 할당
        for t in self.tasks:
            model_name = t.get("model_name")
            path = str(t.adata_dir)
            t.adata_ref = self.adata_refs[model_name][path]
        
        return time.time() - t0

    # ------------------ 실행 ------------------
    def run(self) -> dict[str, list[dict]]:
        # Data Load (모델별 Actor를 통해 로드)
        t_load_data = self._prepare_adata_payloads()

        # Progress bar
        progress = ProgressActor.remote()
        for t, n_gpus in zip(self.tasks, self.per_worker_gpus):
            task_name = t.get("task_name")
            bs = int(t.get("inference").get("batch_size"))
            total_batches = None
            for rank in range(int(n_gpus)):
                ray.get(progress.register.remote(task_name, rank, total_batches, bs))

        def _launch_wrapper(task_cfg, scaling_cfg_kwargs, run_cfg_kwargs, progress_actor):
            """Wrapper function that will be executed with runtime_env"""
            # Worker 환경에서 inference_loop import (accelerate 의존)
            from scunify.core.loops.inference_loop import inference_loop_per_worker
            
            trainer = TorchTrainer(
                inference_loop_per_worker,
                train_loop_config={"cfg": task_cfg, "progress_actor": progress_actor},
                scaling_config=ScalingConfig(**scaling_cfg_kwargs),
                run_config=RunConfig(**run_cfg_kwargs),
            )
            result = trainer.fit()
            return result.metrics

        jobs = []
        for t, n_gpus, n_cpus in zip(self.tasks, self.per_worker_gpus, self.per_worker_cpus):
            scaling_cfg = {
                "num_workers": int(n_gpus),
                "use_gpu": True,
                "resources_per_worker": {"CPU": int(n_cpus), "GPU": 1},
                "placement_strategy": self.placement_strategy,
            }
            run_cfg = {
                "failure_config": FailureConfig(max_failures=0),
                "storage_path": str(self._storage_dir) if self._storage_dir else None,
                #"verbose": self.verbose,
            }
            
            # task의 model_name에 따라 conda 환경 결정
            runtime_env = self._get_runtime_env(t)
            
            # ray.remote에 runtime_env 전달
            _launch_remote = ray.remote(runtime_env=runtime_env)(_launch_wrapper)
            
            t.accelerate = self.accel_cfg
            t.t_load_d = t_load_data
            jobs.append(_launch_remote.remote(t, scaling_cfg, run_cfg, progress))

        ui = ProgressUI(progress, refresh_hz=4, poll_interval=0.25)
        ui.run_until_complete()

        results = ray.get(jobs)
    
    def _get_runtime_env(self, task_cfg) -> dict:
        """
        task 설정에서 model_name을 읽어 적절한 conda 환경을 반환
        
        Args:
            task_cfg: task 설정 (model_name 포함)
            
        Returns:
            runtime_env 딕셔너리
        """
        import subprocess
        
        model_name = task_cfg.get("model_name")
        if not model_name:
            raise ValueError(f"Task configuration missing 'model_name': {task_cfg}")
        
        env_name = f"scunify_{model_name.lower()}"
        
        # conda 환경 존재 확인
        try:
            result = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if env_name not in result.stdout:
                raise RuntimeError(
                    f"Conda environment '{env_name}' not found.\n"
                    f"\n"
                    f"Please run setup first:\n"
                    f"  >>> import scunify as scu\n"
                    f"  >>> scu.setup(\n"
                    f"  ...     resource_dir='./resources',\n"
                    f"  ...     config_dir='./configs',\n"
                    f"  ...     create_conda_envs=True\n"
                    f"  ... )\n"
                    f"\n"
                    f"This will create the required conda environment: {env_name}"
                )
        except FileNotFoundError:
            raise RuntimeError(
                "conda command not found. Please install conda first."
            )
        
        # task 레벨 runtime_env: conda 환경 이름만
        return {
            "conda": env_name,
        }
