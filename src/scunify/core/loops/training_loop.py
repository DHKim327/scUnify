# core/loops/training_loop.py
"""Training loop executed per worker (= 1 GPU) by Ray TorchTrainer.

Parallel structure to ``inference_loop.py`` — same config/seed/GPU setup,
but runs epoch-based training with LoRA + gradient accumulation.

Supports two split modes:
  - single split: one train/valid/test cycle
  - kfold: held-out test + N folds, fresh model per fold

Flow:
  1. Split adata → test + folds (or train/valid/test)
  2. Per fold: build model → LoRA → train → validate → checkpoint → embeddings
  3. Save per-fold outputs (fold_N/ style) + summary.json
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
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


def _train_one_fold(
    *,
    trainer,
    train_adata,
    valid_adata,
    accelerator,
    training_cfg,
    cfg,
    progress,
    task_name,
    local_rank,
    actual_gpu_id,
    fold_label,
    timer,
):
    """Train a single fold (or the single split). Returns per-fold results dict.

    This is the inner training loop extracted so it can be called once (single
    split) or N times (kfold) with a fresh model each time.
    """
    # --------- Build datasets + dataloaders ---------
    train_ds = trainer.build_dataset(train_adata)
    train_dl = trainer.build_dataloader(train_ds)

    has_valid = len(valid_adata) > 0
    if has_valid:
        valid_ds = trainer.build_dataset(valid_adata)
        valid_dl = trainer.build_dataloader(valid_ds, shuffle=False, drop_last=False)
    else:
        valid_dl = None

    # --------- Fresh model + LoRA per fold ---------
    model = trainer.build_model()
    model = trainer.inject_lora(model)

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
    if valid_dl is not None:
        valid_dl = accelerator.prepare(valid_dl)

    # --------- Early Stopping ---------
    es_cfg = training_cfg.get("early_stopping", {})
    early_stopper = EarlyStopping(
        patience=int(es_cfg.get("patience", 5)),
        min_delta=float(es_cfg.get("min_delta", 0.0)),
    )

    # --------- Training ---------
    dl_cfg = cfg.get("dataloader", {})
    bs = int(dl_cfg.get("batch_size", 32))
    total_batches = len(train_dl)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    avg_loss = 0.0
    val_loss = float("inf")
    actual_epochs = 0
    epoch_history = []

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
                fold=fold_label or None,
            )

        avg_loss = epoch_loss / max(total_batches, 1)

        # --------- Validation (DDP-safe: all-reduce average) ---------
        if valid_dl is not None:
            progress.set_status.remote(task_name, accelerator.process_index, "VALIDATING")
            val_loss = _compute_val_loss(
                model, trainer, valid_dl, accelerator
            )

            # Sync val_loss across all workers so early stopping decision is identical
            if accelerator.num_processes > 1:
                _vl = torch.tensor([val_loss], device=accelerator.device)
                torch.distributed.all_reduce(_vl, op=torch.distributed.ReduceOp.AVG)
                val_loss = _vl.item()
        else:
            # No validation set — use train loss for logging
            val_loss = avg_loss

        # Record epoch history
        lr_current = optimizer.param_groups[0].get("lr", 0.0)
        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": round(lr_current, 8),
        })

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
            fold=fold_label or None,
        )

        logger.info(
            f"[{task_name}] {fold_label} Epoch {epoch + 1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}"
        )
        ray_train.report(
            {
                "fold": fold_label,
                "epoch": epoch,
                "loss": avg_loss,
                "val_loss": val_loss,
            }
        )

        # Early stopping only when validation set exists
        if valid_dl is not None and early_stopper.step(val_loss, epoch):
            logger.info(
                f"[{task_name}] {fold_label} Early stopping at epoch {epoch + 1}"
            )
            break

    accelerator.wait_for_everyone()

    # --------- Checkpoint + Merge ---------
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        if fold_label:
            fold_save_dir = cfg.save_dir / fold_label
            fold_save_dir.mkdir(parents=True, exist_ok=True)
        else:
            fold_save_dir = cfg.save_dir
        save_cfg = cfg.get("save", {})
        if save_cfg.get("checkpoint", True):
            ckpt_dir = trainer.save_checkpoint(unwrapped, fold_save_dir)
            logger.info(f"[{task_name}] {fold_label or 'single'} Checkpoint: {ckpt_dir}")
        if save_cfg.get("merged_model", True):
            merged_dir = trainer.merge_and_save(unwrapped, fold_save_dir)
            logger.info(f"[{task_name}] {fold_label or 'single'} Merged: {merged_dir}")

    return {
        "model": model,
        "accelerator": accelerator,
        "actual_epochs": actual_epochs,
        "avg_loss": avg_loss,
        "val_loss": val_loss,
        "early_stopper": early_stopper,
        "train_params": train_params,
        "total_params": total_params,
        "epoch_history": epoch_history,
    }


def _extract_and_save_embeddings(
    *,
    trainer,
    model,
    accelerator,
    train_adata,
    valid_adata,
    test_adata,
    full_adata,
    cfg,
    task_name,
    fold_label,
    fold_results,
    timer,
):
    """Extract embeddings and save.

    When ``full_adata`` is provided and test is empty (self-supervised mode),
    embeddings are extracted for the **entire** dataset in one pass.
    Otherwise falls back to per-split extraction.
    """
    model.eval()
    timer.start(f"embed_{fold_label}")

    unwrapped_model = accelerator.unwrap_model(model)
    has_test = len(test_adata) > 0

    if not has_test and full_adata is not None:
        # Self-supervised mode: extract on full dataset at once
        splits = [("all", full_adata)]
    else:
        splits = [
            ("train", train_adata),
            ("valid", valid_adata),
            ("test", test_adata),
        ]

    split_embeddings = {}
    for split_name, split_adata in splits:
        if len(split_adata) == 0:
            continue
        emb = trainer.extract_embeddings(
            unwrapped_model, split_adata, accelerator
        )
        if emb is not None:
            split_embeddings[split_name] = emb
            logger.info(
                f"[{task_name}] {fold_label} {split_name} embeddings: {emb.shape}"
            )

    t_embed = timer.stop(f"embed_{fold_label}")

    if not accelerator.is_main_process or not split_embeddings:
        return 0, 0, t_embed

    # --- Reassemble in original cell order ---
    if "all" in split_embeddings:
        # Full-dataset extraction: already in order
        all_emb = split_embeddings["all"]
        # Split labels: mark train/valid based on orig_idx
        train_idx_set = set(train_adata.obs["_scunify_orig_idx"].values)
        split_labels = np.array([
            "train" if i in train_idx_set else "valid"
            for i in full_adata.obs["_scunify_orig_idx"].values
        ])
    else:
        # Per-split extraction: collect and sort
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

    if fold_label:
        fold_save_dir = cfg.save_dir / fold_label
        fold_save_dir.mkdir(parents=True, exist_ok=True)
    else:
        fold_save_dir = cfg.save_dir
    out_path = fold_save_dir / f"{cfg.task_name}.npy"

    meta = {
        "name": task_name,
        "fold": fold_label or "single",
        "n_obs": n_obs,
        "n_dim": n_dim,
        "epochs": fold_results["actual_epochs"],
        "final_loss": round(fold_results["avg_loss"], 6),
        "final_val_loss": round(fold_results["val_loss"], 6),
        "early_stopped": fold_results["actual_epochs"] < int(
            cfg.get("training", {}).get("epochs", 50)
        ),
        "best_val_loss": round(fold_results["early_stopper"].best_loss, 6),
        "best_epoch": fold_results["early_stopper"].best_epoch + 1,
        "trainable_params": fold_results["train_params"],
        "total_params": fold_results["total_params"],
        "split_counts": {
            "train": len(train_adata),
            "valid": len(valid_adata),
            "test": len(test_adata),
        },
    }
    trainer.save_outputs(all_emb, out_path, meta)

    # Save split labels
    np.save(
        str(out_path.with_name(f"{cfg.task_name}_splits.npy")),
        split_labels,
        allow_pickle=False,
    )

    return n_obs, n_dim, t_embed


def _save_summary(cfg, all_fold_results, *, t_total, t_load_d,
                   gpu_util_mean, gpu_util_max, gpu_mem_peak):
    """Save summary.json with epoch-level history and aggregate metrics."""
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    training_cfg = cfg.get("training", {})
    split_cfg = training_cfg.get("split", {})

    folds_summary = []
    for r in all_fold_results:
        folds_summary.append({
            "fold": r.get("fold_label", "single"),
            "actual_epochs": r["actual_epochs"],
            "final_train_loss": round(r["avg_loss"], 6),
            "final_val_loss": round(r["val_loss"], 6),
            "best_val_loss": round(r["early_stopper"].best_loss, 6),
            "best_epoch": r["early_stopper"].best_epoch + 1,
            "early_stopped": r["actual_epochs"] < int(training_cfg.get("epochs", 50)),
            "trainable_params": r["train_params"],
            "total_params": r["total_params"],
            "epoch_history": r["epoch_history"],
        })

    val_losses = [r["val_loss"] for r in all_fold_results]
    best_val_losses = [r["early_stopper"].best_loss for r in all_fold_results]

    summary = {
        "task_name": cfg.get("task_name"),
        "model_name": cfg.get("model_name"),
        "config": {
            "lora": training_cfg.get("lora", {}),
            "optimizer": training_cfg.get("optimizer", {}),
            "scheduler": training_cfg.get("scheduler", {}),
            "early_stopping": training_cfg.get("early_stopping", {}),
            "split": split_cfg,
            "epochs": int(training_cfg.get("epochs", 50)),
            "batch_size": int(cfg.get("dataloader", {}).get("batch_size", 32)),
        },
        "results": {
            "n_folds": len(all_fold_results),
            "mean_val_loss": round(float(np.mean(val_losses)), 6),
            "std_val_loss": round(float(np.std(val_losses)), 6),
            "mean_best_val_loss": round(float(np.mean(best_val_losses)), 6),
            "per_fold": folds_summary,
        },
        "timing": {
            "t_total": round(t_total, 3),
            "t_load_data": round(t_load_d, 3),
        },
        "gpu": {
            "util_mean": round(gpu_util_mean, 1),
            "util_max": round(gpu_util_max, 1),
            "mem_peak_bytes": int(gpu_mem_peak),
        },
    }

    out_path = save_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"[{cfg.get('task_name')}] Summary saved: {out_path}")


# ------------------------------------------------------------------ #
#  Main entry point
# ------------------------------------------------------------------ #

def training_loop_per_worker(training_loop_config):
    """Training loop per worker. Mirrors ``inference_loop_per_worker``.

    Expected inputs (training_loop_config):
      - "cfg": ScUnifyConfig with ``training`` section
      - "progress_actor": TrainingProgressActor
    """
    # --------- Config / Seed ---------
    cfg = training_loop_config.get("cfg")

    import random

    training_cfg = cfg.get("training", {})
    seed = cfg.get("dataloader", {}).get("seed", 42)
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

    # --------- Load adata ---------
    adata = ray.get(cfg.get("adata_ref"))
    trainer = TrainerCls(cfg)

    # --------- GPU / Progress setup ---------
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_map = (
        [int(x) for x in visible.split(",")]
        if visible
        else list(range(torch.cuda.device_count()))
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    actual_gpu_id = (
        gpu_map[local_rank] if local_rank < len(gpu_map) else local_rank
    )
    monitor = GPUMonitor(actual_gpu_id, interval=0.5)
    task_name = cfg.get("task_name")

    ray.get(
        progress.set_status.remote(task_name, local_rank, "STARTING")
    )
    monitor.start()

    # --------- Split ---------
    split_cfg = training_cfg.get("split", {})
    splitter = DataSplitter(split_cfg)

    if splitter.n_folds > 1:
        _run_kfold(
            trainer=trainer,
            splitter=splitter,
            adata=adata,
            accelerator=accelerator,
            training_cfg=training_cfg,
            cfg=cfg,
            progress=progress,
            task_name=task_name,
            local_rank=local_rank,
            actual_gpu_id=actual_gpu_id,
            timer=timer,
            monitor=monitor,
            t_load_d=t_load_d,
        )
    else:
        _run_single(
            trainer=trainer,
            splitter=splitter,
            adata=adata,
            accelerator=accelerator,
            training_cfg=training_cfg,
            cfg=cfg,
            progress=progress,
            task_name=task_name,
            local_rank=local_rank,
            actual_gpu_id=actual_gpu_id,
            timer=timer,
            monitor=monitor,
            t_load_d=t_load_d,
        )


# ------------------------------------------------------------------ #
#  Single split path
# ------------------------------------------------------------------ #

def _run_single(
    *,
    trainer,
    splitter,
    adata,
    accelerator,
    training_cfg,
    cfg,
    progress,
    task_name,
    local_rank,
    actual_gpu_id,
    timer,
    monitor,
    t_load_d,
):
    """Single-split training path."""
    timer.start("load(d)")
    train_adata, valid_adata, test_adata = splitter.split(adata)
    t_load_ds = timer.stop("load(d)")

    timer.start("load(m)+train")
    fold_results = _train_one_fold(
        trainer=trainer,
        train_adata=train_adata,
        valid_adata=valid_adata,
        accelerator=accelerator,
        training_cfg=training_cfg,
        cfg=cfg,
        progress=progress,
        task_name=task_name,
        local_rank=local_rank,
        actual_gpu_id=actual_gpu_id,
        fold_label="",
        timer=timer,
    )
    t_train = timer.stop("load(m)+train")

    # --------- Embedding extraction ---------
    n_obs = n_dim = 0
    t_embed = 0.0
    save_cfg = cfg.get("save", {})
    output_keys = save_cfg.get("output_keys", [])
    if output_keys:
        ray.get(
            progress.set_status.remote(task_name, local_rank, "EVAL")
        )
        n_obs, n_dim, t_embed = _extract_and_save_embeddings(
            trainer=trainer,
            model=fold_results["model"],
            accelerator=accelerator,
            train_adata=train_adata,
            valid_adata=valid_adata,
            test_adata=test_adata,
            full_adata=adata,
            cfg=cfg,
            task_name=task_name,
            fold_label="",
            fold_results=fold_results,
            timer=timer,
        )

    # --------- Finish ---------
    ray.get(progress.finish.remote(task_name, local_rank))
    util_mean, util_max, mem_max = monitor.stop()
    if util_mean is None:
        util_mean = util_max = mem_max = 0
    t_total = timer.stop("total")

    # --------- Save summary.json ---------
    if accelerator.is_main_process:
        fold_results["fold_label"] = "single"
        _save_summary(
            cfg, [fold_results],
            t_total=t_total, t_load_d=t_load_d,
            gpu_util_mean=util_mean, gpu_util_max=util_max,
            gpu_mem_peak=mem_max,
        )

    ray_train.report(
        {
            "name": task_name,
            "mode": "training",
            "epochs": fold_results["actual_epochs"],
            "final_loss": round(fold_results["avg_loss"], 6),
            "final_val_loss": round(fold_results["val_loss"], 6),
            "early_stopped": fold_results["actual_epochs"]
            < int(training_cfg.get("epochs", 50)),
            "trainable_params": fold_results["train_params"],
            "total_params": fold_results["total_params"],
            "n_obs": n_obs,
            "n_dim": n_dim,
            "t_train": round(t_train, 3),
            "t_load_data": round(t_load_d, 3),
            "t_total": round(t_total, 3),
            "gpu_util_mean": round(util_mean, 1),
            "gpu_util_max": round(util_max, 1),
            "gpu_mem_peak": int(mem_max),
        }
    )


# ------------------------------------------------------------------ #
#  KFold path
# ------------------------------------------------------------------ #

def _run_kfold(
    *,
    trainer,
    splitter,
    adata,
    accelerator,
    training_cfg,
    cfg,
    progress,
    task_name,
    local_rank,
    actual_gpu_id,
    timer,
    monitor,
    t_load_d,
):
    """K-fold cross-validation training path."""
    timer.start("split")
    test_adata, folds = splitter.split_kfold(adata)
    timer.stop("split")

    n_folds = len(folds)
    all_fold_results = []

    for fold_i, (train_adata, valid_adata) in enumerate(folds):
        fold_label = f"fold_{fold_i}"
        logger.info(
            f"[{task_name}] === {fold_label} ({fold_i + 1}/{n_folds}) ==="
        )

        ray.get(
            progress.set_status.remote(
                task_name, local_rank, f"TRAIN {fold_label}"
            )
        )

        timer.start(f"train_{fold_label}")
        fold_results = _train_one_fold(
            trainer=trainer,
            train_adata=train_adata,
            valid_adata=valid_adata,
            accelerator=accelerator,
            training_cfg=training_cfg,
            cfg=cfg,
            progress=progress,
            task_name=task_name,
            local_rank=local_rank,
            actual_gpu_id=actual_gpu_id,
            fold_label=fold_label,
            timer=timer,
        )
        t_fold_train = timer.stop(f"train_{fold_label}")

        # --------- Per-fold embedding extraction ---------
        n_obs = n_dim = 0
        t_embed = 0.0
        save_cfg = cfg.get("save", {})
        output_keys = save_cfg.get("output_keys", [])
        if output_keys:
            ray.get(
                progress.set_status.remote(
                    task_name, local_rank, f"EVAL {fold_label}"
                )
            )
            n_obs, n_dim, t_embed = _extract_and_save_embeddings(
                trainer=trainer,
                model=fold_results["model"],
                accelerator=accelerator,
                train_adata=train_adata,
                valid_adata=valid_adata,
                test_adata=test_adata,
                full_adata=adata,
                cfg=cfg,
                task_name=task_name,
                fold_label=fold_label,
                fold_results=fold_results,
                timer=timer,
            )

        fold_results["fold_label"] = fold_label
        fold_results["t_train"] = t_fold_train
        fold_results["t_embed"] = t_embed
        fold_results["n_obs"] = n_obs
        fold_results["n_dim"] = n_dim
        all_fold_results.append(fold_results)

        # Report per-fold metrics
        ray_train.report(
            {
                "fold": fold_label,
                "fold_i": fold_i,
                "n_folds": n_folds,
                "epochs": fold_results["actual_epochs"],
                "loss": fold_results["avg_loss"],
                "val_loss": fold_results["val_loss"],
                "best_val_loss": fold_results["early_stopper"].best_loss,
            }
        )

        # Free GPU memory before next fold
        del fold_results["model"]
        torch.cuda.empty_cache()

    # --------- Aggregate and final report ---------
    ray.get(progress.finish.remote(task_name, local_rank))
    util_mean, util_max, mem_max = monitor.stop()
    if util_mean is None:
        util_mean = util_max = mem_max = 0
    t_total = timer.stop("total")

    # --------- Save summary.json ---------
    if accelerator.is_main_process:
        _save_summary(
            cfg, all_fold_results,
            t_total=t_total, t_load_d=t_load_d,
            gpu_util_mean=util_mean, gpu_util_max=util_max,
            gpu_mem_peak=mem_max,
        )

    avg_val_losses = [r["val_loss"] for r in all_fold_results]
    best_val_losses = [r["early_stopper"].best_loss for r in all_fold_results]

    ray_train.report(
        {
            "name": task_name,
            "mode": "training_kfold",
            "n_folds": n_folds,
            "mean_val_loss": round(float(np.mean(avg_val_losses)), 6),
            "std_val_loss": round(float(np.std(avg_val_losses)), 6),
            "mean_best_val_loss": round(float(np.mean(best_val_losses)), 6),
            "per_fold_val_loss": [round(v, 6) for v in avg_val_losses],
            "per_fold_best_val_loss": [round(v, 6) for v in best_val_losses],
            "n_obs": all_fold_results[-1]["n_obs"] if all_fold_results else 0,
            "n_dim": all_fold_results[-1]["n_dim"] if all_fold_results else 0,
            "t_total": round(t_total, 3),
            "t_load_data": round(t_load_d, 3),
            "gpu_util_mean": round(util_mean, 1),
            "gpu_util_max": round(util_max, 1),
            "gpu_mem_peak": int(mem_max),
        }
    )
