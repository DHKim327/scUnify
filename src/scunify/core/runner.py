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
from .loops.inference_loop import inference_loop_per_worker

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
        if ray.is_initialized():
            return
        init_kwargs = {"ignore_reinit_error": True, "local_mode": False}

        init_kwargs["_temp_dir"] = str(self._temp_dir)
        init_kwargs["num_cpus"] = self.total_cpus
        if self.gpu_indices:
            init_kwargs["num_gpus"] = len(self.gpu_indices)
        else:
            init_kwargs["num_gpus"] = self.total_gpus

        __PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
        env_vars = {
            "PYTHONPATH": __PROJECT_ROOT,
            "RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS": "0",
        }
        if self.gpu_indices:
            env_vars["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_indices))
        init_kwargs["runtime_env"] = {
            "working_dir": __PROJECT_ROOT,
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
        if ray.is_initialized():
            ray.shutdown()

        if self._auto_cleanup:
            for p in [self._storage_dir, self._temp_dir]:
                try:
                    if p.exists():
                        shutil.rmtree(p, ignore_errors=True)
                except Exception as e:
                    print(f"[Runner.cleanup] Failed to remove {p}: {e}")

    def _prepare_adata_payloads(self) -> list[dict[str, Any]]:
        """
        같은 adata 객체/경로는 공유하도록 캐싱해서 payload 생성.
        - adata 있으면 → ray.put(adata)
        - adata_dir 있으면 → scanpy.read_h5ad()로 로드 후 ray.put(adata)
        """
        import numpy as np
        import scanpy as sc
        import scipy.sparse as sp

        cache: dict[str, Any] = {}

        for t in self.tasks:
            if t.save_key not in cache:
                ad = sc.read_h5ad(t.adata_dir)
                if sp.issparse(ad.X):
                    ad.X = ad.X.astype(np.float32)
                else:
                    ad.X = np.asarray(ad.X, dtype=np.float32, order="C")
                cache[t.save_key] = ray.put(ad)
            t.adata_ref = cache[t.save_key]
        self.adata_cache = cache

    # ------------------ 실행 ------------------
    def run(self) -> dict[str, list[dict]]:
        # Data Load Time logging
        t0 = time.time()
        self._prepare_adata_payloads()
        t_load_data = time.time() - t0

        # Progress bar
        progress = ProgressActor.remote()
        for t, n_gpus in zip(self.tasks, self.per_worker_gpus):
            task_name = t.get("task_name")
            bs = int(t.get("inference").get("batch_size"))
            total_batches = None
            for rank in range(int(n_gpus)):
                ray.get(progress.register.remote(task_name, rank, total_batches, bs))

        @ray.remote
        def _launch(task_cfg, scaling_cfg_kwargs, run_cfg_kwargs, progress_actor):
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
                "verbose": self.verbose,
            }
            t.accelerate = self.accel_cfg
            t.t_load_d = t_load_data
            jobs.append(_launch.remote(t, scaling_cfg, run_cfg, progress))

        ui = ProgressUI(progress, refresh_hz=4, poll_interval=0.25)
        ui.run_until_complete()

        results = ray.get(jobs)
