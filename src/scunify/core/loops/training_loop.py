# core/loops/training_loop.py
"""Training loop executed per worker (= 1 GPU) by Ray TorchTrainer.

Parallel structure to ``inference_loop.py`` — same config/seed/GPU setup,
but runs epoch-based training with LoRA + gradient accumulation.

Flow (2026-03-20):
  1. Split adata → train / valid / test
  2. Train on train split
  3. Validate each epoch → early stopping
  4. Checkpoint + merge
  5. Extract embeddings per split → concat → save
"""

import logging
import os

import ray
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from ray import train as ray_train

from ...trainer import resolve_trainer
from ...trainer.base import DataSplitter, EarlyStopping
from ..logger import GPUMonitor, TimeLogger

logger = logging.getLogger(__name__)


def _compute_val_loss(model, trainer, valid_dl, accelerator):
    """Run one pass over validation dataloader and return mean loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in valid_dl:
            loss = trainer.compute_loss(model, batch)
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def training_loop_per_worker(training_loop_config):
    """Training loop per worker. Mirrors ``inference_loop_per_worker``.

    Expected inputs (training_loop_config):
      - "cfg": ScUnifyConfig with ``training`` section
      - "progress_actor": TrainingProgressActor
    """
    # --------- Config / Seed ---------
    cfg = training_loop_config.get("cfg")

    import random

    import numpy as np

    training_cfg = cfg.get("training", {})
    seed = training_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)

    progress = training_loop_config.get("progress_actor")

    # --------- Timer ---------
    timer = TimeLogger()
    timer.start("total")
    t_load_d = float(getattr(cfg, "t_load_d", 0.0))

    # --------- Resolve Trainer ---------
    TrainerCls = resolve_trainer(cfg)

    # --------- Accelerator (DDP + LoRA: find_unused_parameters) ---------
    acc_kwargs = cfg.get("accelerate", {}) or {}
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **acc_kwargs)

    # --------- Load adata + Split ---------
    adata = ray.get(cfg.get("adata_ref"))
    trainer = TrainerCls(cfg)

    split_cfg = training_cfg.get("split", {})
    splitter = DataSplitter(split_cfg)
    train_adata, valid_adata, test_adata = splitter.split(adata)

    # --------- Build datasets + dataloaders ---------
    timer.start("load(d)")
    train_ds = trainer.build_dataset(train_adata)
    train_dl = trainer.build_dataloader(train_ds)

    valid_ds = trainer.build_dataset(valid_adata)
    valid_dl = trainer.build_dataloader(valid_ds, shuffle=False, drop_last=False)
    t_load_ds = timer.stop("load(d)")

    # --------- Build model + LoRA ---------
    timer.start("load(m)")
    model = trainer.build_model()
    model = trainer.inject_lora(model)
    t_load_m = timer.stop("load(m)")

    # --------- Optimizer & Scheduler ---------
    epochs = int(training_cfg.get("epochs", 50))
    grad_accum = int(training_cfg.get("gradient_accumulation_steps", 1))
    total_steps = (len(train_dl) // grad_accum) * epochs

    optimizer = trainer.build_optimizer(model)
    scheduler = trainer.build_scheduler(optimizer, total_steps)

    # --------- Accelerate Prepare ---------
    model, optimizer, train_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, scheduler
    )
    valid_dl = accelerator.prepare(valid_dl)

    # --------- Early Stopping ---------
    es_cfg = training_cfg.get("early_stopping", {})
    early_stopper = EarlyStopping(
        patience=int(es_cfg.get("patience", 5)),
        min_delta=float(es_cfg.get("min_delta", 0.0)),
    )

    # --------- GPU / Progress setup ---------
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_map = (
        [int(x) for x in visible.split(",")]
        if visible
        else list(range(torch.cuda.device_count()))
    )
    actual_gpu_id = (
        gpu_map[local_rank] if local_rank < len(gpu_map) else local_rank
    )
    monitor = GPUMonitor(actual_gpu_id, interval=0.5)

    task_name = cfg.get("task_name")
    bs = int(training_cfg.get("batch_size", 32))
    total_batches = len(train_dl)

    ray.get(
        progress.set_status.remote(task_name, local_rank, "STARTING")
    )

    # --------- Training ---------
    monitor.start()
    timer.start("train")
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    avg_loss = 0.0
    val_loss = float("inf")
    actual_epochs = 0

    for epoch in range(epochs):
        actual_epochs = epoch + 1
        epoch_loss = 0.0
        for step, batch in enumerate(train_dl):
            loss = trainer.compute_loss(model, batch)
            loss = loss / grad_accum
            accelerator.backward(loss)

            if (step + 1) % grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * grad_accum

            progress.update.remote(
                task_name,
                accelerator.process_index,
                actual_gpu_id,
                step + 1,
                total_batches,
                bs,
                epoch=epoch,
                total_epochs=epochs,
                loss=loss.item() * grad_accum,
            )

        avg_loss = epoch_loss / max(total_batches, 1)

        # --------- Validation ---------
        val_loss = _compute_val_loss(
            model, trainer, valid_dl, accelerator
        )

        # Report val_loss to progress UI
        progress.update.remote(
            task_name,
            accelerator.process_index,
            actual_gpu_id,
            total_batches,
            total_batches,
            bs,
            epoch=epoch,
            total_epochs=epochs,
            loss=avg_loss,
            val_loss=val_loss,
        )

        logger.info(
            f"[{task_name}] Epoch {epoch + 1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}"
        )
        ray_train.report(
            {"epoch": epoch, "loss": avg_loss, "val_loss": val_loss}
        )

        # --------- Early Stopping Check ---------
        if early_stopper.step(val_loss, epoch):
            logger.info(
                f"[{task_name}] Early stopping at epoch {epoch + 1}"
            )
            break

    t_train = timer.stop("train")
    accelerator.wait_for_everyone()

    # --------- Checkpoint + Merge ---------
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        ckpt_dir = trainer.save_checkpoint(unwrapped, cfg.save_dir)
        merged_dir = trainer.merge_and_save(unwrapped, cfg.save_dir)
        logger.info(f"[{task_name}] Checkpoint: {ckpt_dir}")
        logger.info(f"[{task_name}] Merged model: {merged_dir}")
        logger.info(
            f"[{task_name}] Inference config: "
            f"{cfg.save_dir / f'{task_name}_inference.yaml'}"
        )

    # --------- Embedding extraction (per split → concat) ---------
    n_obs = n_dim = 0
    t_embed = 0.0
    if training_cfg.get("extract_embeddings", True):
        ray.get(
            progress.set_status.remote(task_name, local_rank, "EVAL")
        )
        model.eval()
        timer.start("embed")

        unwrapped_model = accelerator.unwrap_model(model)
        split_embeddings = {}

        for split_name, split_adata in [
            ("train", train_adata),
            ("valid", valid_adata),
            ("test", test_adata),
        ]:
            emb = trainer.extract_embeddings(unwrapped_model, split_adata)
            if emb is not None:
                split_embeddings[split_name] = emb
                logger.info(
                    f"[{task_name}] {split_name} embeddings: {emb.shape}"
                )

        t_embed = timer.stop("embed")

        # Concat and restore original cell order via _scunify_orig_idx
        if accelerator.is_main_process and split_embeddings:
            # Collect embeddings + orig indices + labels per split
            emb_parts = []
            idx_parts = []
            label_parts = []
            for s_name, s_adata in [
                ("train", train_adata),
                ("valid", valid_adata),
                ("test", test_adata),
            ]:
                if s_name in split_embeddings:
                    emb_parts.append(split_embeddings[s_name])
                    idx_parts.append(
                        s_adata.obs["_scunify_orig_idx"].values
                    )
                    label_parts.append(
                        np.full(split_embeddings[s_name].shape[0], s_name)
                    )

            concat_emb = np.concatenate(emb_parts, axis=0)
            concat_idx = np.concatenate(idx_parts)
            concat_labels = np.concatenate(label_parts)

            # Sort by original index to restore adata cell order
            sort_order = np.argsort(concat_idx)
            all_emb = concat_emb[sort_order]
            split_labels = concat_labels[sort_order]

            n_obs, n_dim = all_emb.shape

            out_path = cfg.save_dir / f"{cfg.task_name}.npy"
            meta = {
                "name": task_name,
                "n_obs": n_obs,
                "n_dim": n_dim,
                "epochs": actual_epochs,
                "final_loss": round(avg_loss, 6),
                "final_val_loss": round(val_loss, 6),
                "early_stopped": actual_epochs < epochs,
                "best_val_loss": round(early_stopper.best_loss, 6),
                "best_epoch": early_stopper.best_epoch + 1,
                "trainable_params": train_params,
                "total_params": total_params,
                "split_counts": {
                    "train": len(train_adata),
                    "valid": len(valid_adata),
                    "test": len(test_adata),
                },
                "t_train": round(t_train, 3),
                "t_embed": round(t_embed, 3),
                "t_load_data": round(t_load_d, 3),
                "t_load_model": round(t_load_m, 3),
            }
            trainer.save_outputs(all_emb, out_path, meta)

            # Save split labels
            np.save(
                str(out_path.with_name(f"{cfg.task_name}_splits.npy")),
                split_labels,
                allow_pickle=False,
            )

    # --------- Finish ---------
    ray.get(progress.finish.remote(task_name, local_rank))
    util_mean, util_max, mem_max = monitor.stop()
    if util_mean is None:
        util_mean = util_max = mem_max = 0
    t_total = timer.stop("total")

    ray_train.report(
        {
            "name": task_name,
            "mode": "training",
            "epochs": actual_epochs,
            "final_loss": round(avg_loss, 6),
            "final_val_loss": round(val_loss, 6),
            "early_stopped": actual_epochs < epochs,
            "trainable_params": train_params,
            "total_params": total_params,
            "n_obs": n_obs,
            "n_dim": n_dim,
            "t_train": round(t_train, 3),
            "t_load_data": round(t_load_d, 3),
            "t_load_model": round(t_load_m, 3),
            "t_total": round(t_total, 3),
            "gpu_util_mean": round(util_mean, 1),
            "gpu_util_max": round(util_max, 1),
            "gpu_mem_peak": int(mem_max),
        }
    )
