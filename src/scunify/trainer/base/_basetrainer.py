from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW

from ...utils import load_yaml


class BaseTrainer(ABC):
    """Base class for model-specific LoRA trainers.

    Mirrors ``BaseInferencer`` structure. Each model trainer inherits this
    and implements dataset/model/LoRA specifics.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.model_param = load_yaml(cfg._architecture_dir)
        self.training_cfg = cfg.get("training", {})
        self.lora_cfg = self.training_cfg.get("lora", {})

    # ------------------------------------------------------------------ #
    #  Dataset
    # ------------------------------------------------------------------ #
    @abstractmethod
    def build_dataset(self, adata):
        """Build training dataset (inherits from registry, adds masking/noise)."""
        ...

    def build_dataloader(self, ds, *, shuffle: bool = True, drop_last: bool = True):
        """Build dataloader. Training default: shuffle=True, drop_last=True."""
        from torch.utils.data import DataLoader

        dl_cfg = self.cfg.get("dataloader", {})
        bs = int(dl_cfg.get("batch_size", 32))
        nw = int(dl_cfg.get("num_workers", 0))
        collator = getattr(ds, "collator", None)

        def worker_init_fn(worker_id):
            import random

            seed = dl_cfg.get("seed", 0)
            np.random.seed(seed + worker_id)
            random.seed(seed + worker_id)
            torch.manual_seed(seed + worker_id)

        dl_kwargs = dict(
            batch_size=bs,
            num_workers=nw,
            shuffle=shuffle,
            collate_fn=collator,
            pin_memory=True,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn if nw > 0 else None,
        )
        if nw > 0:
            dl_kwargs["persistent_workers"] = True
            dl_kwargs["prefetch_factor"] = int(dl_cfg.get("prefetch_factor", 4))

        return DataLoader(ds, **dl_kwargs)

    # ------------------------------------------------------------------ #
    #  Model
    # ------------------------------------------------------------------ #
    @abstractmethod
    def build_model(self):
        """Build training wrapper (inherits from registry, overrides forward)."""
        ...

    @abstractmethod
    def inject_lora(self, model: nn.Module) -> nn.Module:
        """Inject LoRA into the model and freeze base weights."""
        ...

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        """Unwrap DDP/FSDP to access the underlying training wrapper."""
        if hasattr(model, "module"):
            return model.module
        return model

    # ------------------------------------------------------------------ #
    #  Training — loss & embedding API
    # ------------------------------------------------------------------ #
    @abstractmethod
    def compute_loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """Compute training loss. Provided by task Mixin (e.g. PretrainingMixin,
        ClassificationMixin). Model trainers do NOT implement this directly."""
        ...

    @abstractmethod
    def get_cell_embedding(
        self, model: nn.Module, batch: dict
    ) -> torch.Tensor:
        """Extract cell-level embedding (B, D) with gradient flow.

        Must NOT use ``torch.no_grad()``. Implementation delegates to
        the training wrapper's ``get_cell_embedding()`` method.
        """
        ...

    @abstractmethod
    def get_gene_embedding(
        self, model: nn.Module, batch: dict
    ) -> torch.Tensor:
        """Extract gene-level embedding (B, S, D) with gradient flow.

        Must NOT use ``torch.no_grad()``. Implementation delegates to
        the training wrapper's ``get_gene_embedding()`` method.
        """
        ...

    def build_optimizer(self, model: nn.Module) -> Adam | AdamW:
        """Optimizer over trainable (LoRA) parameters only.

        Supports ``"adam"`` (scPEFT default) and ``"adamw"``.
        """
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt_cfg = self.training_cfg.get("optimizer", {})
        opt_type = opt_cfg.get("type", "adam").lower()
        lr = float(opt_cfg.get("lr", 1e-5))
        wd = float(opt_cfg.get("weight_decay", 0.0))
        betas = opt_cfg.get("betas", [0.9, 0.999])
        betas = tuple(float(b) for b in betas)

        if opt_type == "adamw":
            return AdamW(trainable, lr=lr, weight_decay=wd, betas=betas)
        return Adam(trainable, lr=lr, weight_decay=wd, betas=betas)

    def build_scheduler(self, optimizer, total_steps: int):
        """Cosine annealing with linear warmup.

        Supports ``warmup_ratio`` (fraction of total steps, scPEFT default 0.9)
        and legacy ``warmup_steps`` (absolute count).
        """
        from torch.optim.lr_scheduler import (
            CosineAnnealingLR,
            LinearLR,
            SequentialLR,
        )

        sched_cfg = self.training_cfg.get("scheduler", {})
        opt_cfg = self.training_cfg.get("optimizer", {})

        # warmup_ratio takes precedence over warmup_steps
        if "warmup_ratio" in sched_cfg:
            warmup = int(total_steps * float(sched_cfg["warmup_ratio"]))
        elif "warmup_steps" in opt_cfg:
            warmup = int(opt_cfg["warmup_steps"])
        else:
            warmup = int(total_steps * 0.9)  # scPEFT default

        warmup = min(warmup, total_steps)
        warmup_sched = LinearLR(
            optimizer, start_factor=0.01, total_iters=max(warmup, 1)
        )
        cosine_sched = CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup, 1)
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup],
        )

    # ------------------------------------------------------------------ #
    #  Post-training embedding extraction (distributed)
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _build_inference_dataset(self, adata):
        """Build inference dataset (no masking) for embedding extraction."""
        ...

    @abstractmethod
    def forward_embed_step(
        self, model: nn.Module, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-batch embedding extraction. Returns (embedding, cid)."""
        ...

    def extract_embeddings(
        self, model: nn.Module, adata, accelerator
    ) -> np.ndarray | None:
        """Distributed embedding extraction — mirrors inference_loop pattern.

        Uses ``accelerator.prepare(dl)`` for DistributedSampler and
        ``accelerator.gather_for_metrics`` for cross-GPU collection.

        Returns numpy array on main process, ``None`` on other ranks.
        """
        ds = self._build_inference_dataset(adata)
        dl = self.build_dataloader(ds, shuffle=False, drop_last=False)
        dl = accelerator.prepare(dl)

        emb_chunks: list[torch.Tensor] = []
        cid_chunks: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in dl:
                emb, cid = self.forward_embed_step(model, batch)
                gemb = accelerator.gather_for_metrics(emb)
                gcid = accelerator.gather_for_metrics(cid)
                emb_chunks.append(gemb.cpu())
                cid_chunks.append(gcid.cpu())

        if accelerator.is_main_process and emb_chunks:
            E = torch.cat(emb_chunks, dim=0)
            C = torch.cat(cid_chunks, dim=0).long()
            order = torch.argsort(C, stable=True)
            return E[order].float().numpy()
        return None

    # ------------------------------------------------------------------ #
    #  Checkpoint save / load / merge
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, model: nn.Module, output_dir: str | Path) -> Path:
        """Save LoRA adapter weights only (via HF PEFT ``save_pretrained``)."""
        ckpt_dir = Path(output_dir) / "checkpoints" / self.cfg.task_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.model.save_pretrained(ckpt_dir)
        return ckpt_dir

    def merge_and_save(self, model: nn.Module, output_dir: str | Path) -> Path:
        """Merge LoRA into base weights and save as a standard model.

        1. ``merge_and_unload()`` — merge PEFT adapters into base weights.
        2. ``refuse_mha_layers()`` — re-fuse any unfused QKV back to
           ``nn.MultiheadAttention`` (no-op for models that were never unfused).
        3. Save — ``save_pretrained`` for HF models, ``torch.save`` for others.

        The merged model can be loaded by the existing inferencer/ pipeline
        without any code changes — just point config model_dirs to this path.
        """
        from ..lora._unfused_mha import refuse_mha_layers

        merged_dir = Path(output_dir) / "merged" / self.cfg.task_name
        merged_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Merge LoRA adapters into base weights
        merged = model.model.merge_and_unload()

        # Step 2: Re-fuse unfused attention (no-op if none found)
        refuse_mha_layers(merged)

        # Step 3: Save
        if hasattr(merged, "save_pretrained"):
            merged.save_pretrained(merged_dir)
        else:
            torch.save(
                merged.state_dict(),
                merged_dir / "merged_model.pt",
            )

        self._generate_inference_config(merged_dir, output_dir)
        return merged_dir

    def _generate_inference_config(
        self, merged_dir: Path, output_dir: str | Path
    ) -> None:
        """Generate an inference config yaml with merged model path filled in."""
        import copy

        import yaml

        # Start from the original config (which has resources, inference, etc.)
        inf_cfg = copy.deepcopy(self.cfg.config)

        # Remove training-only fields
        inf_cfg.pop("mode", None)
        inf_cfg.pop("training", None)

        # Replace model path with merged model directory
        resources = inf_cfg.get("resources", {})
        model_dirs = resources.get("model_dirs", {})
        if model_dirs:
            # Find which variant was used for training
            variant = inf_cfg.get("model", {}).get("variant")
            if variant and variant in model_dirs:
                model_dirs[variant] = str(merged_dir)
            else:
                # Fallback: replace first entry
                first_key = next(iter(model_dirs))
                model_dirs[first_key] = str(merged_dir)
        elif "model_file" in resources:
            resources["model_file"] = str(merged_dir / "merged_model.pt")

        out_path = Path(output_dir) / f"{self.cfg.task_name}_inference.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(inf_cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # ------------------------------------------------------------------ #
    #  Output helpers
    # ------------------------------------------------------------------ #
    def save_outputs(
        self, outputs: np.ndarray, out_path: str | Path, meta: dict[str, Any]
    ) -> None:
        """Save embeddings (.npy) + metadata (.json)."""
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), outputs.astype(np.float32), allow_pickle=False)
        with open(str(p.with_suffix(".json")), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
