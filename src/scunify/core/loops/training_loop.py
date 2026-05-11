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
import time
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


def _resolve_monitor(training_cfg: dict) -> tuple[str, str]:
    """Resolve (metric, direction) for best-ckpt + early-stop tracking.

    Source of truth is the yaml ``training.monitor`` block:

        training:
          monitor:
            metric: val_pearson
            direction: max

    When omitted, falls back to ``val_loss`` / ``min`` (BC / CLS recipe).
    Paper-faithful values (e.g. scGPT perturbation = val_pearson/max,
    GEARS perturbation = val_mse_de/min) are expressed in the per-task
    yaml — framework-level configs are user-provided so unified-setting
    visibility is preserved.
    """
    raw = training_cfg.get("monitor")
    block = raw or {}
    metric = block.get("metric") or "val_loss"
    direction = str(block.get("direction") or "min").lower()
    if direction not in ("min", "max"):
        raise ValueError(
            f"monitor.direction must be 'min' or 'max', got {direction!r}"
        )
    return str(metric), direction


def _compute_val_loss(model, trainer, valid_dl, accelerator):
    """Run one pass over validation dataloader and return mean loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in valid_dl:
            with accelerator.autocast():
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
    full_adata,
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

    # --------- Fresh model + freeze policy per fold ---------
    # Three-way ``training.mode`` spectrum (see ``trainer.lora._freeze``):
    #   * ``full``   — Full FT (default; partial-FT via ``freeze.layer_strategy``)
    #   * ``probe``  — backbone fully frozen, head only
    #   * ``lora``   — backbone fully frozen + LoRA on selected layers
    # The single discriminator across all three is ``trainer.is_backbone_param``.
    model = trainer.build_model()
    mode = (training_cfg.get("mode") or "full").lower()
    if mode == "lora":
        model = trainer.inject_lora(model)
    elif mode == "probe":
        from scunify.trainer.lora._freeze import apply_probe_freeze
        apply_probe_freeze(model, is_backbone_param=trainer.is_backbone_param)
    else:
        # mode == "full": apply layer-selective freeze if the yaml specifies
        # one. ``freeze.layer_strategy = all`` (default) keeps every backbone
        # parameter trainable; ``layer_strategy: none`` is equivalent to
        # ``mode: probe``.
        from scunify.trainer.lora._freeze import (
            apply_full_ft_freeze, resolve_layer_indices,
        )
        from scunify.trainer.lora._injection import (
            _count_layers, _find_encoder_layers,
        )
        from scunify.trainer.lora._targets import LAYERS_PATTERN

        freeze_cfg = trainer.freeze_cfg or {}
        if freeze_cfg:
            backbone = getattr(model, "model", model)
            try:
                n_layers = len(_find_encoder_layers(backbone))
            except Exception:
                n_layers = _count_layers(backbone)
            layer_indices = resolve_layer_indices(n_layers, freeze_cfg)
            apply_full_ft_freeze(
                model,
                is_backbone_param=trainer.is_backbone_param,
                layer_indices=layer_indices,
                layers_pattern=LAYERS_PATTERN.get(
                    cfg.get("model_name", "").lower(), "layers"
                ),
            )

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

    # --------- Monitor (best-ckpt + early-stop shared metric/direction) ---------
    # Read from yaml ``training.monitor.{metric, direction}`` block; defaults
    # to ``val_loss`` / ``min`` when absent. Paper-faithful per-task values
    # live in the user-provided yaml (framework-level configs).
    monitor, monitor_direction = _resolve_monitor(training_cfg)

    # --------- Early Stopping ---------
    # yaml ``training.early_stopping`` block is opt-in; when omitted (None or
    # missing), training runs the full ``epochs`` count — matching the BC /
    # classification paper recipes (scGPT, scPEFT, Geneformer, UCE,
    # Nicheformer all use ``best_val_loss`` ckpt selection without breaking
    # the loop). val_loss tracking + best-ckpt save still happen.
    # EarlyStopping shares ``monitor_direction`` with best-ckpt tracking so
    # they always agree.
    es_cfg = training_cfg.get("early_stopping") or {}
    early_stopper = (
        EarlyStopping(
            patience=int(es_cfg["patience"]),
            min_delta=float(es_cfg.get("min_delta", 0.0)),
            direction=monitor_direction,
        )
        if es_cfg
        else None
    )

    # --------- Gradient Clipping ---------
    # ``gradient_clip``: scalar threshold (None disables).
    # ``gradient_clip_type``: ``"norm"`` (default; ``clip_grad_norm_`` —
    # scGPT recipe) or ``"value"`` (``clip_grad_value_`` — GEARS recipe).
    # Both functions exist in HF Accelerate; framework dispatches based on
    # the yaml key per task.
    _clip = training_cfg.get("gradient_clip", 1.0)
    grad_clip = float(_clip) if _clip is not None else None
    grad_clip_type = str(training_cfg.get("gradient_clip_type", "norm")).lower()
    if grad_clip_type not in ("norm", "value"):
        raise ValueError(
            f"training.gradient_clip_type must be 'norm' or 'value', got {grad_clip_type!r}"
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

    # --------- Best-checkpoint setup ---------
    # has_valid: track best-val state (overwrite ckpt on each val improvement);
    #            reload best at end so output extraction uses best ckpt.
    # not has_valid: skip per-epoch best tracking (val_loss = train_loss is
    #                monotonic — every epoch would "improve" → wasted I/O);
    #                save the final-epoch ckpt once after the loop.
    if fold_label:
        fold_save_dir = cfg.save_dir / fold_label
        fold_save_dir.mkdir(parents=True, exist_ok=True)
    else:
        fold_save_dir = cfg.save_dir
    has_valid = valid_dl is not None
    # ``monitor`` / ``monitor_direction`` resolved above via _resolve_monitor.
    # Selects the metric used for best-ckpt tracking + early stopping.
    # Perturbation mixins override (scGPT Tutorial: val_pearson/max; GEARS:
    # val_mse_de/min); BC/CLS use the TaskMixin base default val_loss/min.
    best_val_for_ckpt = float("-inf") if monitor_direction == "max" else float("inf")
    best_val_epoch = -1
    min_delta = float(es_cfg.get("min_delta", 0.0))
    save_cfg = cfg.get("save", {})
    early_stopped_flag = False

    # --------- Unified extraction setup ---------
    # ``save.outputs`` controls what to extract (cell_embedding,
    # gene_embedding, custom predict() keys). ``save.epochs`` schedules
    # when — integers for mid-training snapshots, "final" for end-of-train.
    extract_outputs = save_cfg.get("outputs") or []
    extract_epochs = save_cfg.get("epochs") or ["final"]
    mid_extract_epochs = {int(e) for e in extract_epochs if e != "final"}
    extract_state: dict = {"dl": None}    # holds the lazy inference dl
    mid_outputs: dict = {}                # {epoch_int: {key: {data, storage}}}

    for epoch in range(epochs):
        actual_epochs = epoch + 1
        epoch_loss = 0.0
        _epoch_start = time.time()
        _train_start = time.time()
        for step, batch in enumerate(train_dl):
            with accelerator.autocast():
                loss = trainer.compute_loss(model, batch)
            loss = loss / grad_accum
            accelerator.backward(loss)

            if (step + 1) % grad_accum == 0:
                if grad_clip is not None:
                    if grad_clip_type == "value":
                        accelerator.clip_grad_value_(model.parameters(), grad_clip)
                    else:
                        accelerator.clip_grad_norm_(model.parameters(), grad_clip)
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

        _train_time = time.time() - _train_start
        avg_loss = epoch_loss / max(total_batches, 1)

        # --------- Validation (DDP-safe: all-reduce average) ---------
        _val_time = 0.0
        val_metrics: dict = {}
        if has_valid:
            progress.set_status.remote(task_name, accelerator.process_index, "VALIDATING")
            _val_start = time.time()
            # Trainer hook returns ``{metric_name: float}``. Default
            # implementation in ``BaseTrainer.compute_val_metrics`` returns
            # ``{'val_loss': avg_loss}``; perturbation mixins override to
            # add task-specific metrics (val_pearson / val_mse_de).
            val_metrics = trainer.compute_val_metrics(
                model, valid_dl, accelerator
            )
            val_loss = float(val_metrics.get("val_loss", float("inf")))
            _val_time = time.time() - _val_start

            # Sync each metric across workers so early stopping decision is identical
            if accelerator.num_processes > 1:
                synced = {}
                for k, v in val_metrics.items():
                    _vl = torch.tensor([float(v)], device=accelerator.device)
                    torch.distributed.all_reduce(_vl, op=torch.distributed.ReduceOp.AVG)
                    synced[k] = _vl.item()
                val_metrics = synced
                val_loss = float(val_metrics.get("val_loss", val_loss))
        else:
            # No validation set — surface train loss in logs/history
            # (no role in best-ckpt or early-stop decisions, both gated on
            # ``has_valid``).
            val_loss = avg_loss

        # Record epoch history (include all task-specific val_metrics so
        # users can audit best-ckpt selection vs final-extract metrics).
        _epoch_time = time.time() - _epoch_start
        lr_current = optimizer.param_groups[0].get("lr", 0.0)
        ep_record = {
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 6),
            "val_loss": round(val_loss, 6),
            "epoch_time": round(_epoch_time, 3),
            "train_time": round(_train_time, 3),
            "val_time": round(_val_time, 3),
            "lr": round(lr_current, 8),
        }
        for k, v in val_metrics.items():
            if k == "val_loss":
                continue
            try:
                ep_record[k] = round(float(v), 6)
            except (TypeError, ValueError):
                ep_record[k] = v
        epoch_history.append(ep_record)
        if accelerator.is_main_process:
            logger.info(
                f"[{task_name}] {fold_label} Epoch {epoch + 1} val_metrics: " +
                ", ".join(f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                          for k, v in val_metrics.items())
            )

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

        # --------- Best-checkpoint save (validation-gated) ---------
        # Skip entirely when there is no validation set — monitored metric
        # would trip ``improved`` every epoch and churn the disk. The
        # final-epoch ckpt is written once after the loop instead.
        if has_valid:
            # Pull the monitored metric. Hard-fail if the mixin's
            # compute_val_metrics did not return ``monitor`` — silent
            # fallback to val_loss masks bugs (e.g. KeyError inside the
            # metric calc returning a partial dict).
            if monitor not in val_metrics:
                raise KeyError(
                    f"compute_val_metrics did not return monitor key {monitor!r}. "
                    f"Available keys: {sorted(val_metrics)}. "
                    f"Either fix the mixin's compute_val_metrics to populate "
                    f"{monitor!r}, or change training.monitor.metric in yaml."
                )
            monitor_value = float(val_metrics[monitor])
            if monitor_direction == "max":
                improved = monitor_value > best_val_for_ckpt + min_delta
            else:
                improved = monitor_value < best_val_for_ckpt - min_delta
            if improved:
                best_val_for_ckpt = monitor_value
                best_val_epoch = epoch
                if accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(model)
                    trainer.save_checkpoint(unwrapped, fold_save_dir)
                accelerator.wait_for_everyone()

        # ---- Mid-epoch extraction (unified save.epochs) ----
        # When the current epoch is in ``save.epochs``, run a single
        # inference pass on ``full_adata`` and stash the per-key outputs
        # under their epoch label. Final h5ad merges these with the
        # ``"final"`` pass after best-ckpt restore.
        if extract_outputs and (epoch + 1) in mid_extract_epochs:
            ep_dl = _ensure_extract_dl(
                extract_state,
                trainer=trainer,
                accelerator=accelerator,
                full_adata=full_adata,
            )
            ep_out = _collect_outputs_one_pass(
                trainer=trainer,
                model=model,
                accelerator=accelerator,
                dl=ep_dl,
                output_keys=extract_outputs,
                task_name=task_name,
                fold_label=fold_label,
            )
            if accelerator.is_main_process:
                mid_outputs[epoch + 1] = ep_out

        # Early stopping only when validation set exists. Tracks the same
        # monitor metric as best-ckpt (KeyError already raised above if the
        # mixin omitted it, so val_metrics[monitor] is guaranteed present).
        _es_metric = float(val_metrics[monitor]) if has_valid else val_loss
        if early_stopper is not None and has_valid and early_stopper.step(_es_metric, epoch):
            logger.info(
                f"[{task_name}] {fold_label} Early stopping at epoch {epoch + 1}"
            )
            early_stopped_flag = True
            break

    accelerator.wait_for_everyone()

    # --------- Final checkpoint resolution ---------
    # has_valid: best ckpt was saved during training; reload it so
    #            merge / extraction uses the best model.
    # not has_valid: model in memory is already the final-epoch state;
    #                save it once now (only ckpt produced for this fold).
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        if has_valid and best_val_for_ckpt < float("inf"):
            trainer.load_checkpoint(unwrapped, fold_save_dir)
            logger.info(
                f"[{task_name}] {fold_label or 'single'} Restored best checkpoint "
                f"(val_loss={best_val_for_ckpt:.4f})"
            )
        elif not has_valid:
            trainer.save_checkpoint(unwrapped, fold_save_dir)
            logger.info(
                f"[{task_name}] {fold_label or 'single'} Saved final-epoch ckpt "
                f"(no valid set — train_loss={avg_loss:.4f})"
            )
    accelerator.wait_for_everyone()

    # --------- Merge (if LoRA) ---------
    # The block above ensures the in-memory model is in its final state
    # (best ckpt restored when has_valid; otherwise the final-epoch state)
    # and the matching ckpt is on disk. ``merge_and_save`` only needs to
    # fold LoRA weights into the base — no extra reload required.
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
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
        "early_stopped": early_stopped_flag,
        "best_val_loss": best_val_for_ckpt,
        "best_val_epoch": best_val_epoch,
        "train_params": train_params,
        "total_params": total_params,
        "epoch_history": epoch_history,
        # Forwarded to ``_extract_and_save_outputs`` so the final h5ad merges
        # mid-training snapshots without re-running inference for them.
        "mid_outputs": mid_outputs,
        "_extract_state": extract_state,
    }


# ---------------------------------------------------------------------- #
#  Unified extraction (cell / gene / predict() outputs across epochs)
# ---------------------------------------------------------------------- #
#
# yaml ``save.outputs`` lists named outputs to extract. Two well-known keys
# bypass ``predict()`` and use a dedicated trainer method:
#
#     cell_embedding  →  trainer.forward_embed_step(model, batch)
#     gene_embedding  →  trainer.forward_gene_embed_step(model, batch)
#
# Any other key is looked up in ``trainer.predict(model, batch)`` (which
# returns ``{key: {data, storage}}``). This keeps the dispatcher trivial
# while letting custom mixins surface arbitrary per-cell tensors.
#
# yaml ``save.epochs`` schedules when extraction runs. ``"final"`` triggers
# after the training loop (using the restored best ckpt). Integers trigger
# at the end of that epoch (mid-training snapshots). Per-epoch outputs are
# saved to ``obsm/obs/uns`` with an ``_ep{N}`` suffix; ``final`` keeps the
# bare name.
RESERVED_EXTRACT_KEYS = {"cell_embedding", "gene_embedding"}


def _collect_outputs_one_pass(
    *,
    trainer,
    model,
    accelerator,
    dl,
    output_keys,
    task_name,
    fold_label,
):
    """Single inference pass — returns ``{key: {data: np.ndarray, storage: str}}``.

    Empty dict on non-main ranks. The returned arrays are already ordered
    by ``cid`` so downstream callers can write them straight into
    ``adata.obsm`` / ``adata.obs`` / ``adata.uns``.
    """
    unwrapped_model = accelerator.unwrap_model(model)
    was_training = unwrapped_model.training
    model.eval()

    needs_predict = any(k not in RESERVED_EXTRACT_KEYS for k in output_keys)

    cid_chunks = []
    cell_chunks = []
    gene_chunks = []
    task_collectors: dict[str, dict] = {}  # k → {"chunks": [...], "storage": str}

    with torch.no_grad():
        for batch in dl:
            # Trainer hook — default reads ``batch["cid"]``/``batch[-1]``;
            # perturbation mixins override to read PyG ``batch.cid``.
            cid = trainer.extract_cid(batch)

            if "cell_embedding" in output_keys:
                emb, cid = trainer.forward_embed_step(unwrapped_model, batch)
                cell_chunks.append(accelerator.gather(emb).cpu())

            if "gene_embedding" in output_keys:
                gemb, cid = trainer.forward_gene_embed_step(unwrapped_model, batch)
                gene_chunks.append(accelerator.gather(gemb).cpu())

            if needs_predict:
                task_out = trainer.predict(unwrapped_model, batch)
                for k, v in task_out.items():
                    if k in RESERVED_EXTRACT_KEYS:
                        if accelerator.is_main_process:
                            logger.warning(
                                f"[{task_name}] {fold_label} predict() returned "
                                f"reserved key {k!r} — dropping. Use "
                                f"save.outputs: [{k}] (dedicated path) instead."
                            )
                        continue
                    if k not in output_keys:
                        continue   # not requested
                    task_collectors.setdefault(
                        k, {"chunks": [], "storage": v["storage"]}
                    )
                    task_collectors[k]["chunks"].append(
                        accelerator.gather(v["data"]).cpu()
                    )

            cid_chunks.append(accelerator.gather(cid).cpu())

    if was_training:
        model.train()

    if not accelerator.is_main_process:
        return {}

    cids = torch.cat(cid_chunks).long()
    order = torch.argsort(cids, stable=True)

    result: dict[str, dict] = {}
    if cell_chunks:
        result["cell_embedding"] = {
            "data": torch.cat(cell_chunks)[order].float().numpy(),
            "storage": "obsm",
        }
    if gene_chunks:
        result["gene_embedding"] = {
            "data": torch.cat(gene_chunks)[order].float().numpy(),
            "storage": "obsm",
        }
    for k, c in task_collectors.items():
        result[k] = {
            "data": torch.cat(c["chunks"])[order].numpy(),
            "storage": c["storage"],
        }
    return result


def _ensure_extract_dl(state: dict, *, trainer, accelerator, full_adata):
    """Lazy-build (and prepare) the inference dataloader exactly once.

    Stored on a caller-provided ``state`` dict so the same dl is reused for
    mid-epoch and final extraction.
    """
    if state.get("dl") is None:
        ds = trainer._build_inference_dataset(full_adata)
        dl = trainer.build_dataloader(ds, shuffle=False, drop_last=False)
        state["dl"] = accelerator.prepare(dl)
    return state["dl"]


def _save_collected_h5ad(
    *,
    extract_adata,
    collected,
    fold_save_dir,
    task_name,
    fold_label,
    fold_results,
    train_adata,
    valid_adata,
    test_adata,
):
    """Write a single h5ad combining all epoch-labelled outputs.

    ``collected`` is ``{epoch_label: {key: {data, storage}}}`` where
    ``epoch_label`` is either ``"final"`` (bare key name) or an ``int``
    (suffixed ``_ep{N}``).
    """
    result_adata = extract_adata.copy()
    n_obs = len(result_adata)
    n_dim = 0

    for epoch_label, outputs in collected.items():
        suffix = "" if epoch_label == "final" else f"_ep{int(epoch_label)}"
        for key, val in outputs.items():
            data = val["data"]
            storage = val["storage"]
            if storage == "obsm":
                result_adata.obsm[f"X_{key}{suffix}"] = data
                if key == "cell_embedding" and epoch_label == "final":
                    n_dim = data.shape[1] if data.ndim > 1 else 1
            elif storage == "obs":
                result_adata.obs[f"{key}{suffix}"] = data
            elif storage == "uns":
                result_adata.uns[f"{key}{suffix}"] = data

    result_adata.uns["scunify_meta"] = {
        "task_name": task_name,
        "fold": fold_label or "single",
        "epochs": fold_results["actual_epochs"],
        "final_loss": round(fold_results["avg_loss"], 6),
        "final_val_loss": round(fold_results["val_loss"], 6),
        "trainable_params": fold_results["train_params"],
        "total_params": fold_results["total_params"],
        "split_counts": {
            "train": len(train_adata),
            "valid": len(valid_adata),
            "test": len(test_adata),
        },
    }

    h5ad_path = fold_save_dir / f"{task_name}.h5ad"
    result_adata.write_h5ad(h5ad_path)
    logger.info(f"[{task_name}] {fold_label or 'single'} Saved: {h5ad_path}")
    return n_obs, n_dim


def _extract_and_save_outputs(
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
    """Final extraction wrapper.

    Runs the canonical inference pass on ``full_adata`` for every key in
    ``save.outputs`` and merges the result with any mid-epoch collections
    handed back by ``_train_one_fold`` (``fold_results['mid_outputs']``).

    Always returns ``(n_obs, n_dim, t_output)`` so the caller can populate
    its ray-train report. Returns zeros when ``save.outputs`` is empty (no
    extraction requested).
    """
    if full_adata is None:
        raise ValueError(
            "_extract_and_save_outputs requires full_adata; both _run_single "
            "and _run_kfold pass adata as full_adata."
        )

    save_cfg = cfg.get("save", {})
    output_keys = save_cfg.get("outputs") or []
    epochs_cfg = save_cfg.get("epochs") or ["final"]
    do_final = "final" in epochs_cfg

    timer.start(f"output_{fold_label}")

    collected = dict(fold_results.get("mid_outputs") or {})

    if output_keys and do_final:
        extract_state = fold_results.get("_extract_state") or {}
        dl = _ensure_extract_dl(
            extract_state,
            trainer=trainer,
            accelerator=accelerator,
            full_adata=full_adata,
        )
        final_out = _collect_outputs_one_pass(
            trainer=trainer,
            model=model,
            accelerator=accelerator,
            dl=dl,
            output_keys=output_keys,
            task_name=task_name,
            fold_label=fold_label,
        )
        if accelerator.is_main_process:
            collected["final"] = final_out

    t_output = timer.stop(f"output_{fold_label}")

    if not accelerator.is_main_process or not collected:
        return 0, 0, t_output

    if fold_label:
        fold_save_dir = cfg.save_dir / fold_label
        fold_save_dir.mkdir(parents=True, exist_ok=True)
    else:
        fold_save_dir = cfg.save_dir

    # Trainer hook — perturbation mixins return ``full_adata[fold_0=='test']``
    # so the saved h5ad's obsm/obs dims match the inference subset.
    extract_adata = trainer.inference_adata(full_adata)
    n_obs, n_dim = _save_collected_h5ad(
        extract_adata=extract_adata,
        collected=collected,
        fold_save_dir=fold_save_dir,
        task_name=task_name,
        fold_label=fold_label,
        fold_results=fold_results,
        train_adata=train_adata,
        valid_adata=valid_adata,
        test_adata=test_adata,
    )
    return n_obs, n_dim, t_output


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
            "best_val_loss": round(r.get("best_val_loss", float("inf")), 6),
            "best_epoch": r.get("best_val_epoch", -1) + 1,
            "early_stopped": r.get("early_stopped", False),
            "trainable_params": r["train_params"],
            "total_params": r["total_params"],
            "epoch_history": r["epoch_history"],
        })

    val_losses = [r["val_loss"] for r in all_fold_results]
    best_val_losses = [r.get("best_val_loss", float("inf")) for r in all_fold_results]

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

    # --------- Accelerator ---------
    # DDP (find_unused_parameters=True) only when multi-GPU; skipping on single
    # GPU avoids gradient-bucket overhead for frozen LoRA params.
    acc_kwargs = cfg.get("accelerate", {}) or {}
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], **acc_kwargs)
    else:
        accelerator = Accelerator(**acc_kwargs)

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

    try:
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
    except BaseException as e:
        # Write error.log next to summary.json and mark task FAILED so the
        # progress UI terminates instead of hanging on a killed worker.
        import traceback as _tb
        err_text = _tb.format_exc()
        save_dir = getattr(cfg, "save_dir", None)
        if save_dir:
            try:
                save_dir_path = Path(save_dir)
                save_dir_path.mkdir(parents=True, exist_ok=True)
                with open(save_dir_path / "error.log", "w") as f:
                    f.write(f"Task: {task_name}\n")
                    f.write(f"Rank: {local_rank}\n")
                    f.write(f"GPU: {actual_gpu_id}\n")
                    f.write(f"Error: {type(e).__name__}: {e}\n\n")
                    f.write(err_text)
                logger.error(
                    f"[{task_name}] Error logged to: "
                    f"{save_dir_path / 'error.log'}"
                )
            except Exception as log_err:
                logger.error(
                    f"[{task_name}] Failed to write error.log: {log_err}"
                )
        try:
            ray.get(
                progress.fail.remote(
                    task_name, local_rank, f"{type(e).__name__}: {e}"
                )
            )
        except Exception:
            pass
        raise


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
        full_adata=adata,
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
    extract_outputs = save_cfg.get("outputs") or []
    extract_epochs = save_cfg.get("epochs") or ["final"]
    needs_extraction = bool(extract_outputs) and (
        "final" in extract_epochs or fold_results.get("mid_outputs")
    )
    if needs_extraction:
        ray.get(
            progress.set_status.remote(task_name, local_rank, "EVAL")
        )
        n_obs, n_dim, t_embed = _extract_and_save_outputs(
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
            "early_stopped": fold_results.get("early_stopped", False),
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

    fold_keys = splitter.fold_keys  # user-defined obs column names
    # Base seed for per-fold reseeding — prevents RNG drift across folds.
    # Critical for models whose model-build draws from RNG (e.g. scFoundation
    # Performer's random projection matrices): without this, each successive
    # fold starts from a more-drifted RNG state and can diverge.
    import random
    base_seed = cfg.get("dataloader", {}).get("seed", 42)

    for fold_i, (train_adata, valid_adata) in enumerate(folds):
        fold_label = fold_keys[fold_i] if fold_i < len(fold_keys) else f"fold_{fold_i}"

        # Same seed every fold — folds differ by data split, not by init.
        # Prevents init-sensitivity tails (some seed_i draws unlucky Performer
        # projection matrices and diverges). Standard CV practice.
        fold_seed = base_seed
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fold_seed)

        logger.info(
            f"[{task_name}] === {fold_label} ({fold_i + 1}/{n_folds}) "
            f"seed={fold_seed} ==="
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
            full_adata=adata,
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
        extract_outputs = save_cfg.get("outputs") or []
        extract_epochs = save_cfg.get("epochs") or ["final"]
        needs_extraction = bool(extract_outputs) and (
            "final" in extract_epochs or fold_results.get("mid_outputs")
        )
        if needs_extraction:
            ray.get(
                progress.set_status.remote(
                    task_name, local_rank, f"EVAL {fold_label}"
                )
            )
            n_obs, n_dim, t_embed = _extract_and_save_outputs(
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
                "best_val_loss": fold_results.get("best_val_loss", float("inf")),
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
    best_val_losses = [r.get("best_val_loss", float("inf")) for r in all_fold_results]

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
