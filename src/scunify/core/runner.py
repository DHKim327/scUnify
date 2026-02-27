# core/runner.py
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
        dev_mode: bool = False,
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
            assert len(per_task_gpus) == len(self.tasks), "per_task_gpus length must match the number of tasks."
            self.per_worker_gpus = [int(v) for v in per_task_gpus]
        else:
            self.per_worker_gpus = [int(syscfg["ray"]["per_worker"].get("GPU"))] * len(self.tasks)

        if isinstance(per_task_cpus, int):
            self.per_worker_cpus = [per_task_cpus] * len(self.tasks)
        elif isinstance(per_task_cpus, list):
            assert len(per_task_cpus) == len(self.tasks), "per_task_cpus length must match the number of tasks."
            self.per_worker_cpus = [int(v) for v in per_task_cpus]
        else:
            self.per_worker_cpus = [int(syscfg["ray"]["per_worker"].get("CPU"))] * len(self.tasks)

        # Accelerate configuration
        self.accel_cfg = {
            "mixed_precision": mixed_precision or syscfg["accelerate"].get("mixed_precision", "fp16"),
            "cpu": bool(syscfg["accelerate"].get("cpu", False)),
        }

        storage_root = Path(ray_storage_dir).expanduser().resolve() if ray_storage_dir else None
        temp_root = Path(ray_temp_dir or "~/ray_temps").expanduser().resolve()
        self._storage_dir = storage_root
        self._temp_dir = temp_root

        # Other configuration
        self.placement_strategy = syscfg["ray"].get("placement_strategy", "PACK")
        self.verbose = verbose
        self._dev_mode = dev_mode

        # Initialize Ray
        self._auto_cleanup = bool(auto_cleanup)
        self._resource_check()
        self._initialize_ray()

    def _initialize_ray(self):
        import os
        # Increase worker startup timeout (24h) for large-scale datasets
        os.environ.setdefault("RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S", "86400")
        
        if ray.is_initialized():
            return
        init_kwargs = dict(ignore_reinit_error=True, local_mode=False)
        
        # Disable dashboard to avoid PlacementGroupCleaner State API warnings
        init_kwargs["include_dashboard"] = False
        
        init_kwargs["_temp_dir"] = str(self._temp_dir)
        init_kwargs["num_cpus"] = self.total_cpus
        if self.gpu_indices:
            init_kwargs["num_gpus"] = len(self.gpu_indices)
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_indices))
        else:
            init_kwargs["num_gpus"] = self.total_gpus
        
        # Default environment variables
        env_vars = {
            "RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS": "0",
            "MPLBACKEND": "Agg",  # Headless backend for workers
        }
        
        # Production mode: no working_dir needed (scunify is installed in each conda env)
        runtime_env = {"env_vars": env_vars}
        
        # Dev mode: reflect source changes immediately
        if self._dev_mode:
            import scunify
            src_dir = str(Path(scunify.__file__).resolve().parent.parent)
            
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
            
            runtime_env["working_dir"] = src_dir
            runtime_env["excludes"] = excludes
            
            if self.verbose:
                print(f"[Runner] Dev mode: working_dir={src_dir}")
        
        init_kwargs["runtime_env"] = runtime_env
        ray.init(**init_kwargs)

    def _resource_check(self):
        per_job = []
        for g, c in zip(self.per_worker_gpus, self.per_worker_cpus):
            g = int(g)
            c = int(c)
            per_job.append({"gpus": g, "cpus": g * c, "cpu_per_worker": c})

            # [1] Per-job resource exceeds total -> error
            if self.total_gpus and g > self.total_gpus:
                raise ValueError(f"Job requires {g} GPUs but total_gpus is {self.total_gpus}.")
            if self.total_cpus and g * c > self.total_cpus:
                raise ValueError(f"Job requires {g * c} CPUs but total_cpus is {self.total_cpus}.")

        # [2] Aggregate exceeds total -> warning only (jobs will be queued)
        need_gpus = sum(j["gpus"] for j in per_job)
        need_cpus = sum(j["cpus"] for j in per_job)

        if self.verbose:
            print("[RayRunner] Resource plan")
            for i, j in enumerate(per_job):
                print(
                    f"  - task[{i}]: num_workers={self.per_worker_gpus[i]} -> needs GPU={j['gpus']}, CPU={j['cpus']} (per-worker CPU={j['cpu_per_worker']})"
                )
            print(
                f"  = TOTAL requested: GPU={need_gpus}, CPU={need_cpus} (limits: GPU={self.total_gpus}, CPU={self.total_cpus})"
            )

            if (self.total_gpus and need_gpus > self.total_gpus) or (self.total_cpus and need_cpus > self.total_cpus):
                print(
                    "[RayRunner] Note: Total requested > limits. Jobs will queue and run sequentially as resources free up."
                )

    def shutdown(self):
        # Clean up actors
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
        """Create per-model DataLoader actors.

        Each model environment (scunify_scgpt, scunify_uce, scunify_scfoundation)
        gets a dedicated actor so that workers within the same environment can
        share data via zero-copy reads from the Ray Object Store.

        Returns:
            dict: {model_name: DataLoaderActor} mapping.
        """
        model_names = set(t.get("model_name") for t in self.tasks)
        
        actors = {}
        for model_name in model_names:
            env_name = f"scunify_{model_name.lower()}"
            
            if self.verbose:
                print(f"[Runner] Creating DataLoader Actor for {model_name} (env: {env_name})...")
            
            actor = DataLoaderActor.options(
                runtime_env={
                    "conda": env_name,
                    "env_vars": {"MPLBACKEND": "Agg"},
                },
                name=f"data_loader_{model_name.lower()}",
            ).remote()
            
            actors[model_name] = actor
        
        return actors

    def _prepare_adata_payloads(self) -> float:
        """Load data via per-model actors.

        Each model's actor loads data into the Ray Object Store once.
        Workers within the same model environment share it via zero-copy reads.

        Returns:
            float: Data loading time in seconds.
        """
        t0 = time.time()
        
        self.data_actors = self._create_data_actors()
        
        # Group required data paths by model
        model_paths: dict[str, set[str]] = {}
        for t in self.tasks:
            model_name = t.get("model_name")
            path = str(t.adata_dir)
            if model_name not in model_paths:
                model_paths[model_name] = set()
            model_paths[model_name].add(path)
        
        # Preload data in each actor
        preload_futures = {}
        for model_name, paths in model_paths.items():
            actor = self.data_actors[model_name]
            if self.verbose:
                print(f"[Runner] Preloading {len(paths)} dataset(s) for {model_name}...")
            preload_futures[model_name] = actor.preload.remote(list(paths))
        
        # Wait for preload completion and store refs
        self.adata_refs: dict[str, dict[str, Any]] = {}
        for model_name, future in preload_futures.items():
            self.adata_refs[model_name] = ray.get(future)
            if self.verbose:
                print(f"[Runner] {model_name} data loaded!")
        
        # Assign adata_ref to each task
        for t in self.tasks:
            model_name = t.get("model_name")
            path = str(t.adata_dir)
            t.adata_ref = self.adata_refs[model_name][path]
        
        return time.time() - t0

    # -------------------- Execution --------------------
    def run(self) -> dict[str, list[dict]]:
        # Load data via per-model actors
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
            """Wrapper executed within the model's runtime_env."""
            # Import inference loop inside the worker env (requires accelerate)
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
            }
            
            # Determine conda env from the task's model_name
            runtime_env = self._get_runtime_env(t)
            _launch_remote = ray.remote(runtime_env=runtime_env)(_launch_wrapper)
            
            t.accelerate = self.accel_cfg
            t.t_load_d = t_load_data
            jobs.append(_launch_remote.remote(t, scaling_cfg, run_cfg, progress))

        ui = ProgressUI(progress, refresh_hz=4, poll_interval=0.25)
        ui.run_until_complete()

        results = ray.get(jobs)

        # Post-processing: merge .npy embeddings into AnnData and save as .h5ad
        self._postprocess_results()

        return results

    def _postprocess_results(self) -> None:
        """Merge per-task .npy embeddings into AnnData objects and save as .h5ad.

        Groups tasks by their source adata path, loads the original AnnData once
        per dataset, inserts each model's embedding into ``adata.obsm["X_{model}"]``,
        writes the consolidated ``.h5ad`` to the task's ``save_dir``, and removes
        the intermediate ``.npy`` files.
        """
        import numpy as np
        import scanpy as sc
        from collections import defaultdict

        groups: dict[str, list] = defaultdict(list)
        for t in self.tasks:
            groups[str(t.adata_dir)].append(t)

        for adata_path, task_list in groups.items():
            if self.verbose:
                print(f"[Runner] Post-processing: {Path(adata_path).name}")

            adata = sc.read_h5ad(adata_path)

            for t in task_list:
                model_name = t.get("model_name").lower()
                obsm_key = f"X_{model_name}"
                npy_path = t.save_dir / f"{t.task_name}.npy"

                if npy_path.exists():
                    embedding = np.load(str(npy_path))
                    adata.obsm[obsm_key] = embedding
                    if self.verbose:
                        print(f"  - Loaded {obsm_key}: {embedding.shape}")
                else:
                    if self.verbose:
                        print(f"  - [WARNING] {npy_path} not found, skipping {obsm_key}")

            out_dir = task_list[0].save_dir
            out_path = out_dir / Path(adata_path).name
            adata.write_h5ad(out_path)
            if self.verbose:
                print(f"  - Saved: {out_path}")

            for t in task_list:
                npy_path = t.save_dir / f"{t.task_name}.npy"
                if npy_path.exists():
                    npy_path.unlink()
                    if self.verbose:
                        print(f"  - Removed: {npy_path}")

            del adata

        if self.verbose:
            print("[Runner] Post-processing complete.")

    def _get_runtime_env(self, task_cfg) -> dict:
        """Return the conda runtime_env for a given task.

        Note:
            Conda env existence is already verified during ``scu.setup()``.
            No subprocess calls are made here; Ray handles env activation.

        Args:
            task_cfg: Task configuration containing ``model_name``.

        Returns:
            Runtime env dict for Ray.
        """
        model_name = task_cfg.get("model_name")
        if not model_name:
            raise ValueError(f"Task configuration missing 'model_name': {task_cfg}")
        
        env_name = f"scunify_{model_name.lower()}"
        
        # Task-level runtime_env: conda env name only.
        # Env existence is verified at setup(); delegated to Ray here.
        return {
            "conda": env_name,
        }
