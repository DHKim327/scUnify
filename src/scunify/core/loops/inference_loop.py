# core/loops/inference_loop.py
import os

import ray
import torch
from accelerate import Accelerator
from ray import train as ray_train

from ...inferencer import resolve_inferencer
from ..logger import GPUMonitor, TimeLogger


def inference_loop_per_worker(inference_loop_config):
    """
    Ray TorchTrainer에서 각 워커(=GPU 1장)마다 실행되는 추론 루프.

    기대 입력(inference_loop_config):
      - "config": Dict (name, save_path, inference{batch_size,num_workers}, accelerate{...} 등)
      - "adata_ref": ray.ObjectRef (필수)
      - "save_key": str
      - "inferencer_class": picklable 클래스 (BaseInferencer 구현체)

    출력/동작:
      - DDP+Accelerate 표준에 맞춰 분산 추론
      - main process만 .npy(+ .json) 저장
      - 모든 워커에서 ray_train.report(...) 1회 호출
    """
    # Set Worker GPU for DDP
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
    # --------- Config / Inputs ---------
    cfg = inference_loop_config.get("cfg")
    progress = inference_loop_config.get("progress_actor")
    adata = ray.get(cfg.get("adata_ref"))
    save_key = cfg.save_key

    # --------- Timer Initialization -----------
    timer = TimeLogger()
    timer.start("total")
    t_load_d = float(getattr(cfg, "t_load_d", 0.0))
    # --------- Get Inferencer from model name -------
    InferCls = resolve_inferencer(cfg)

    # --------- Accelerator / Device ---------
    acc_kwargs = cfg.get("accelerate", {}) or {}
    accelerator = Accelerator(**acc_kwargs)

    # --------- Build pipeline via Inferencer ---------
    infer = InferCls(cfg)
    ds = infer.build_dataset(adata)
    dl = infer.build_dataloader(ds)

    timer.start("load(m)")
    model = infer.build_model()

    model.eval()
    # Accelerate prepare (DDP wrapping, device placement)
    model, dl = accelerator.prepare(model, dl)
    t_load_m = timer.stop("load(m)")

    # --------- Progress Bar ------------
    ray.get(progress.set_status.remote(cfg.get("task_name"), local_rank, "STARTING"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_map = [int(x) for x in visible.split(",")] if visible else list(range(torch.cuda.device_count()))
    actual_gpu_id = gpu_map[local_rank] if local_rank < len(gpu_map) else local_rank
    monitor = GPUMonitor(actual_gpu_id, interval=0.5)

    total_batches = len(dl)
    bs = int(cfg.get("inference").get("batch_size"))
    task_name = cfg.get("task_name")

    # --------- Inference ---------
    monitor.start()
    timer.start("infer")
    emb_chunks = []
    cid_chunks = []
    with torch.no_grad(), accelerator.autocast():
        for idx, batch in enumerate(dl):
            emb, cid = infer.forward_step(model, batch)  # emb:(B,D), cid:(B,)
            gemb = accelerator.gather_for_metrics(emb)  # (B*, D)
            gcid = accelerator.gather_for_metrics(cid)  # (B*,)
            emb_chunks.append(gemb.detach().cpu())
            cid_chunks.append(gcid.detach().cpu())
            progress.update.remote(task_name, accelerator.process_index, actual_gpu_id, idx + 1, total_batches, bs)

    accelerator.wait_for_everyone()
    ray.get(progress.finish.remote(cfg.get("task_name"), local_rank))
    # Logger Finish
    t_infer = timer.stop("infer")
    util_mean, util_max, mem_max = monitor.stop()
    if util_mean is None:
        util_mean = util_max = mem_max = 0

    # Save on main process
    n_obs = n_dim = 0
    t_total = timer.stop("total")
    if accelerator.is_main_process:
        E = torch.cat(emb_chunks, dim=0)  # (N, D)
        I = torch.cat(cid_chunks, dim=0).long()  # (N,)
        order = torch.argsort(I, stable=True)  # 전역 원순서 복원
        E = E[order]
        arr = E.float().numpy()
        n_obs, n_dim = int(arr.shape[0]), int(arr.shape[1])

        cfg.get("save_dir").mkdir(parents=True, exist_ok=True)
        out_path = cfg.get("save_dir") / f"{cfg.get('task_name')}.npy"

        meta = {
            "name": cfg.get("task_name"),
            "save_key": save_key,
            "n_obs": n_obs,
            "n_dim": n_dim,
            "world_size": accelerator.num_processes,
            "batch_size": int(cfg.get("inference", {}).get("batch_size", 32)),
            "dataset_n": int(len(ds)),
            "t_infer": round(t_infer, 3),
            "t_load_model": round(t_load_m, 3),
            "t_load_data": round(t_load_d, 3),
            "t_total": round(t_total, 3),
            "gpu_util_mean": round(util_mean, 1),
            "gpu_util_max": round(util_max, 1),
            "gpu_mem_peak": int(mem_max),
        }
        infer.save_outputs(arr, out_path, meta)

    ray_train.report(
        {
            "name": cfg.get("task_name"),
            "save_key": save_key,
            "path": out_path if accelerator.is_main_process else "",
            "world_size": accelerator.num_processes,
            "rank": accelerator.process_index,
            "batch_size": int(cfg.get("inference", {}).get("batch_size", 32)),
            "dataset_n": int(len(ds)),
            "n_obs": n_obs,
            "n_dim": n_dim,
            "t_infer": round(t_infer, 3),
            "t_load_model": round(t_load_m, 3),
            "t_load_data": round(t_load_d, 3),
            "t_total": round(t_total, 3),
            "gpu_util_mean": round(util_mean, 1),
            "gpu_util_max": round(util_max, 1),
            "gpu_mem_peak": int(mem_max),
        }
    )
