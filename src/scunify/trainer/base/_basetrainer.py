"""BaseTrainer — common interface every backbone-specific trainer subclasses.

Sections (top to bottom):
  A. Lifecycle           — dataset / dataloader / model / LoRA / loss
  B. Embedding API       — cell / gene embedding extraction with grad
  C. Task head hooks     — paper-faithful head attach (TaskMixin v2)
  D. Optimizer/Scheduler
  E. Inference extraction— inference dataset, distributed embedding
  F. Gene-level helpers  — token IDs, vocab, align_to_adata_var
  G. Checkpoint          — save / load / merge_and_save
  H. Output              — npy + json
  I. Internal            — __init__, _unwrap

Most defaults assume the **wrapper-attach** pattern (head sits at
``model.head``, separate from ``model.model``). Geneformer's integrated
classifier (``BertForSequenceClassification``) overrides the C-section
hooks to swap the inner model.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ...utils import load_yaml


class BaseTrainer(ABC):
    """Base class for model-specific LoRA trainers.

    Mirrors ``BaseInferencer`` structure. Each model trainer inherits this
    and implements dataset/model/LoRA specifics.
    """

    # ============================================================ #
    # I. Internal — init & DDP-unwrap
    # ============================================================ #
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.model_param = load_yaml(cfg._architecture_dir)
        self.training_cfg = cfg.get("training", {})
        self.lora_cfg = self.training_cfg.get("lora", {})
        self.freeze_cfg = self.training_cfg.get("freeze") or {}

    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        """Unwrap DDP/FSDP to access the underlying training wrapper."""
        if hasattr(model, "module"):
            return model.module
        return model

    # ============================================================ #
    # A. Lifecycle — dataset / dataloader / model / LoRA / loss
    # ============================================================ #
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

    @abstractmethod
    def build_model(self):
        """Build training wrapper (inherits from registry, overrides forward)."""
        ...

    @abstractmethod
    def inject_lora(self, model: nn.Module) -> nn.Module:
        """Inject LoRA into the model and freeze base weights."""
        ...

    @abstractmethod
    def compute_loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """Compute training loss. Provided by task Mixin (e.g. PretrainingMixin,
        ClassificationMixin). Model trainers do NOT implement this directly."""
        ...

    # ============================================================ #
    # B. Embedding API — cell / gene embedding (gradient flow preserved)
    # ============================================================ #
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

    # ============================================================ #
    # C. Task head hooks (TaskMixin v2)
    #    default = "wrapper-attach" pattern: head sits at ``model.head``.
    #    Geneformer overrides all four to swap the inner model.
    # ============================================================ #
    def default_head(
        self, task_type: str, emb_dim: int, n_classes: int
    ) -> nn.Module | None:
        """Return paper-faithful head for the given task on this backbone.

        Override per-model (e.g. ``ScGPTTrainer``) to return scGPT's
        ``ClsDecoder``, Nicheformer's ``Linear(bias=False)``, etc.

        Return ``None`` if this backbone uses an integrated classifier
        (e.g. Geneformer ``BertForSequenceClassification``). In that case
        also override :meth:`attach_task_head`.
        """
        return None

    def attach_task_head(
        self, model: nn.Module, task_type: str, emb_dim: int, n_classes: int
    ) -> nn.Module:
        """Attach the task head to ``model``.

        Default behaviour: build ``default_head`` and assign as ``model.head``.

        Override to swap the inner model (Geneformer
        ``BertForSequenceClassification``) or use a paper-specific attribute
        name. ``model`` is the (DDP-unwrapped) training wrapper.
        """
        head = self.default_head(task_type, emb_dim, n_classes)
        if head is not None:
            model.head = head
        return model

    # ============================================================ #
    # C2. Backbone-vs-head discrimination — single source of truth used by
    #     LoRA injection, Full-FT freeze (partial-FT), and probe mode.
    # ============================================================ #
    def is_backbone_param(self, name: str) -> bool:
        """Return ``True`` if ``name`` is a parameter of the freezable
        pretrained backbone (as opposed to a task-specific head).

        Default rule: every parameter under ``wrapper.model.*`` is the
        backbone; ``wrapper.head.*`` (and anything else attached at the
        wrapper level) is treated as the task head.

        Override per trainer / mixin when:

        * the head is integrated *inside* ``wrapper.model`` (Geneformer
          ``BertForSequenceClassification`` ``classifier`` / ``pooler``).
        * ``wrapper.model`` itself is a composite where only a subset is
          the pretrained backbone (perturbation: scFoundation
          ``singlecell_model.*`` is the only freezable backbone; the rest
          of ``model.*`` is GEARS heads).

        The hook receives the parameter name as returned by
        ``wrapper.named_parameters()`` (so PEFT-wrapped names like
        ``model.base_model.model.<...>`` are also handled — substring /
        prefix logic typically still works).
        """
        return name.startswith("model.")

    def classifier_logits(
        self, model: nn.Module, batch: dict
    ) -> torch.Tensor:
        """Forward pass returning classification logits ``(B, n_classes)``.

        Default: ``head(get_cell_embedding(batch))`` where ``head`` is
        ``model.head`` (attached by :meth:`attach_task_head`).

        Override for integrated classifiers where the inner model itself
        produces logits (Geneformer ``BertForSequenceClassification``).
        """
        cell_emb = self.get_cell_embedding(model, batch)
        m = self._unwrap(model)
        return m.head(cell_emb)

    # ============================================================ #
    # D. Optimizer / Scheduler
    # ============================================================ #
    def build_optimizer(self, model: nn.Module):
        """Optimizer over trainable parameters — PyTorch passthrough.

        yaml ``training.optimizer.type`` is the exact class name on
        ``torch.optim`` (e.g. ``Adam``, ``AdamW``, ``SGD``); every other
        key is forwarded as a kwarg to the constructor.
        """
        import torch.optim as optim

        trainable = [p for p in model.parameters() if p.requires_grad]
        opt_cfg = dict(self.training_cfg.get("optimizer", {}))

        # ``warmup_steps`` is consumed by build_scheduler; not an optim kwarg.
        opt_cfg.pop("warmup_steps", None)
        cls_name = opt_cfg.pop("type")

        cls = getattr(optim, cls_name, None)
        if cls is None or not callable(cls):
            raise ValueError(
                f"Unknown optimizer type {cls_name!r}. Expected a class on "
                f"``torch.optim`` (e.g. ``Adam``, ``AdamW``, ``SGD``)."
            )

        if "lr" in opt_cfg:
            opt_cfg["lr"] = float(opt_cfg["lr"])
        if "weight_decay" in opt_cfg:
            opt_cfg["weight_decay"] = float(opt_cfg["weight_decay"])
        if "betas" in opt_cfg:
            opt_cfg["betas"] = tuple(float(b) for b in opt_cfg["betas"])

        return cls(trainable, **opt_cfg)

    def build_scheduler(self, optimizer, total_steps: int):
        """Learning rate scheduler — PyTorch passthrough.

        yaml ``training.scheduler.type`` is the exact class name on
        ``torch.optim.lr_scheduler``; every other key is forwarded as a
        kwarg.

        Two scunify-specific extensions are recognised on top of the raw
        passthrough:

        * ``warmup_ratio`` / ``warmup_steps`` — wraps the configured
          scheduler with a linear warmup via ``SequentialLR``.
        * ``step_unit: "epoch"`` (StepLR only) — converts ``step_size``
          from epoch-units to optimizer-step units so ``step_size=1``
          matches the paper convention "decay every epoch".
        """
        from torch.optim.lr_scheduler import LinearLR, SequentialLR
        import torch.optim.lr_scheduler as schedulers

        sched_cfg = dict(self.training_cfg.get("scheduler", {}))
        opt_cfg = self.training_cfg.get("optimizer", {})

        warmup_ratio = sched_cfg.pop("warmup_ratio", None)
        warmup_steps_cfg = sched_cfg.pop("warmup_steps", None) or opt_cfg.get("warmup_steps")
        if warmup_ratio is not None:
            warmup = int(total_steps * float(warmup_ratio))
        elif warmup_steps_cfg is not None:
            warmup = int(warmup_steps_cfg)
        else:
            warmup = 0
        warmup = min(warmup, total_steps)

        cls_name = sched_cfg.pop("type")
        cls = getattr(schedulers, cls_name, None)
        if cls is None or not callable(cls):
            raise ValueError(
                f"Unknown scheduler type {cls_name!r}. Expected a class on "
                f"``torch.optim.lr_scheduler``."
            )

        # ``step_unit: "epoch"`` (StepLR only) — convert step_size from
        # epoch-units to optimizer-step units.
        if cls_name == "StepLR":
            step_unit = str(sched_cfg.pop("step_unit", "epoch")).lower()
            if step_unit == "epoch":
                epochs = int(self.training_cfg.get("epochs", 1))
                steps_per_epoch = max(total_steps // max(epochs, 1), 1)
                sched_cfg["step_size"] = int(sched_cfg["step_size"]) * steps_per_epoch
            if "gamma" in sched_cfg:
                sched_cfg["gamma"] = float(sched_cfg["gamma"])

        # CosineAnnealingLR / CosineAnnealingWarmRestarts — auto-inject
        # ``T_max`` (= total optimizer steps) when omitted, since framework
        # calls ``scheduler.step()`` per optimizer step.
        if cls_name in ("CosineAnnealingLR", "CosineAnnealingWarmRestarts"):
            t_key = "T_max" if cls_name == "CosineAnnealingLR" else "T_0"
            if t_key not in sched_cfg:
                sched_cfg[t_key] = max(total_steps - warmup, 1)

        main_sched = cls(optimizer, **sched_cfg)

        # If no warmup, return main_sched directly (avoids the SequentialLR
        # base_lr-scaling bug that previously dropped lr to 1% silently).
        if warmup == 0:
            return main_sched

        warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
        return SequentialLR(
            optimizer,
            schedulers=[warmup_sched, main_sched],
            milestones=[warmup],
        )

    # ============================================================ #
    # E. Inference / extraction (distributed)
    # ============================================================ #
    @abstractmethod
    def _build_inference_dataset(self, adata):
        """Build inference dataset (no masking) for embedding extraction."""
        ...

    @abstractmethod
    def forward_embed_step(
        self, model: nn.Module, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-batch **cell** embedding extraction.

        Returns ``(embedding, cid)`` — both are required. Implementations
        run under ``torch.no_grad`` and apply the inference recipe for
        their backbone (e.g. scGPT BC L2-normalises the CLS token, paper
        ``finetune_integration.py:657``).

        This is the canonical path for ``cell_embedding`` extraction
        triggered by ``save.outputs: [cell_embedding]``. ``predict()`` must
        NOT also return a ``cell_embedding`` key — the framework drops it
        with a warning to avoid silently overwriting the inference recipe
        with an un-normalised ``encode()``.
        """
        ...

    def forward_gene_embed_step(
        self, model: nn.Module, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-batch **gene** embedding extraction.

        Default: wrap :meth:`get_gene_embedding` in ``no_grad``. Override
        per-backbone if a paper-specific inference recipe (e.g.
        normalisation, layer selection, alignment to ``adata.var``) is
        needed at extraction time.

        Returns ``(gene_embedding, cid)``. Used by the unified extraction
        path triggered by ``save.outputs: [gene_embedding]``.
        """
        with torch.no_grad():
            emb = self.get_gene_embedding(model, batch)
        cid = batch["cid"] if isinstance(batch, dict) else batch[-1]
        return emb, cid

    def inference_adata(self, full_adata):
        """Return the AnnData slice that ``save.outputs`` results map onto.

        Default: ``full_adata`` — every output row corresponds to a row in
        the full input adata (BC/CLS pattern).

        Override in mixins where extraction runs on a subset (perturbation
        runs only on the ``test`` split — paper-faithful evaluation set —
        so output shapes are ``(n_test_cells, ...)``, not ``(n_full, ...)``).
        """
        return full_adata

    def compute_val_metrics(self, model, valid_dl, accelerator) -> dict:
        """Return validation metrics dict for best-ckpt + early-stopping.

        Default: ``{'val_loss': avg(compute_loss(model, batch))}`` — averaged
        across the validation loader. Override in mixin to add task-specific
        metrics in a single pass:

        - **scGPT perturbation**: returns ``val_loss`` + ``val_pearson`` /
          ``val_pearson_de`` (paper Tutorial best-ckpt = ``val_pearson``).
        - **scFoundation perturbation (GEARS)**: returns ``val_loss`` +
          ``val_mse_de`` / ``val_pearson_de`` (paper recipe best-ckpt =
          ``val_mse_de``).

        Yaml ``training.monitor`` selects which key drives best-ckpt save +
        early stopping; ``training.monitor_direction`` (``min``/``max``)
        controls the comparison.
        """
        model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in valid_dl:
                with accelerator.autocast():
                    loss = self.compute_loss(model, batch)
                total_loss += loss.item()
                n_batches += 1
        model.train()
        avg = total_loss / max(n_batches, 1)
        return {"val_loss": avg}

    def extract_cid(self, batch) -> torch.Tensor:
        """Pull the cell-index tensor from a batch.

        Default: dict batch uses ``batch["cid"]``; tuple/list batch uses
        ``batch[-1]``. Override for non-standard batch types — e.g.
        perturbation mixins yielding ``torch_geometric.data.Data`` objects
        that carry cid as an attribute (``batch.cid``).

        The framework extraction loop (``_collect_outputs_one_pass``) calls
        this once per batch to order ``predict()`` outputs back to the
        source ``adata`` row index.
        """
        if isinstance(batch, dict):
            return batch["cid"]
        return batch[-1]

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

    # ============================================================ #
    # F. Gene-level helpers (standard interface across backbones)
    # ============================================================ #
    def get_gene_token_ids(self, batch: dict) -> torch.Tensor:
        """Per-token vocabulary IDs for the gene dimension of ``get_gene_embedding``.

        Returns a ``(B, S)`` int tensor where each entry is the backbone's
        own vocab index. Override per-backbone (each one stores tokens
        under a different batch key — ``batch["gene"]`` for scGPT,
        ``batch["input_ids"]`` for Geneformer/Nicheformer, etc.).
        """
        raise NotImplementedError(
            f"{type(self).__name__}.get_gene_token_ids is not implemented yet."
        )

    def gene_vocab(self) -> dict[int, str]:
        """Return ``{token_id: gene_symbol}`` for this backbone's vocabulary.

        Override per-backbone. Implementations should cache the dict.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.gene_vocab is not implemented yet."
        )

    def align_to_adata_var(
        self,
        gene_emb: torch.Tensor,
        batch: dict,
        adata,
    ) -> np.ndarray:
        """Reorder ``gene_emb`` (B, S, D) to match ``adata.var`` order.

        Returns ``(B, n_var, D)`` numpy array; genes that are not in the
        backbone's vocabulary (or not in this batch's token slice) get NaN.
        Lets users do gene-level analysis (heatmaps, gene-gene attention,
        ...) without needing to know the backbone's tokenization.
        """
        token_ids = self.get_gene_token_ids(batch).detach().cpu().numpy()  # (B, S)
        emb_np = gene_emb.detach().cpu().numpy()                           # (B, S, D)
        vocab = self.gene_vocab()
        # adata.var.index → its token id (or -1 if not in vocab)
        sym_to_tok = {sym: tok for tok, sym in vocab.items()}
        var_token_ids = np.array(
            [sym_to_tok.get(g, -1) for g in adata.var.index], dtype=np.int64
        )

        B, S, D = emb_np.shape
        n_var = adata.shape[1]
        out = np.full((B, n_var, D), np.nan, dtype=np.float32)
        # token_id → var_index (only in-vocab adata genes)
        valid_mask = var_token_ids >= 0
        tok_to_var = dict(zip(var_token_ids[valid_mask], np.where(valid_mask)[0]))
        for b in range(B):
            for s in range(S):
                tok = int(token_ids[b, s])
                v = tok_to_var.get(tok)
                if v is not None:
                    out[b, v] = emb_np[b, s]
        return out

    # ============================================================ #
    # G. Checkpoint — save / load / merge
    # ============================================================ #
    _EXTRAS_FILE = "wrapper_extras.pt"

    @staticmethod
    def _is_peft_model(inner_model) -> bool:
        """Detect PEFT-wrapped model (vs HF-native with save_pretrained)."""
        try:
            from peft import PeftModel
            return isinstance(inner_model, PeftModel)
        except ImportError:
            return False

    @staticmethod
    def _wrapper_extras(model: nn.Module) -> dict:
        """Collect wrapper-level parameters that live OUTSIDE ``model.model``.

        PEFT's ``save_pretrained`` only stores adapter weights from
        ``model.model``; it misses classifier heads attached directly to
        the wrapper (``cls_decoder``, ``cls_norm``, ``cls_head``,
        ``classifier``, ``linear_head``, etc.). We capture those here so
        best-checkpoint restore actually restores the full trainable state.
        """
        return {
            k: v for k, v in model.state_dict().items()
            if not k.startswith("model.")
        }

    def save_checkpoint(self, model: nn.Module, output_dir: str | Path) -> Path:
        """Save checkpoint.

        - LoRA (PEFT) mode: save adapter weights via ``save_pretrained`` AND
          wrapper-level extras (classifier heads) via ``torch.save``.
        - Full-FT mode (HF or custom): save full state_dict via ``torch.save``.
        """
        ckpt_dir = Path(output_dir) / "checkpoints" / self.cfg.task_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if self._is_peft_model(model.model):
            model.model.save_pretrained(ckpt_dir)
            extras = self._wrapper_extras(model)
            if extras:
                torch.save(extras, ckpt_dir / self._EXTRAS_FILE)
        else:
            torch.save(model.state_dict(), ckpt_dir / "pytorch_model.bin")
        return ckpt_dir

    def load_checkpoint(self, model: nn.Module, output_dir: str | Path) -> Path:
        """Load checkpoint written by :meth:`save_checkpoint`.

        - LoRA (PEFT) mode: load adapter weights AND wrapper-level extras.
        - Full-FT mode: load full state_dict.
        """
        ckpt_dir = Path(output_dir) / "checkpoints" / self.cfg.task_name
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        if self._is_peft_model(model.model):
            from peft.utils import set_peft_model_state_dict
            st_path = ckpt_dir / "adapter_model.safetensors"
            bin_path = ckpt_dir / "adapter_model.bin"
            if st_path.exists():
                from safetensors.torch import load_file
                state = load_file(str(st_path))
            elif bin_path.exists():
                state = torch.load(bin_path, map_location="cpu")
            else:
                raise FileNotFoundError(
                    f"No adapter weights in {ckpt_dir} (looked for "
                    f"adapter_model.safetensors / adapter_model.bin)"
                )
            set_peft_model_state_dict(model.model, state)

            extras_path = ckpt_dir / self._EXTRAS_FILE
            if extras_path.exists():
                extras = torch.load(extras_path, map_location="cpu")
                missing, unexpected = model.load_state_dict(extras, strict=False)
                # unexpected should be empty — extras only contains non-'model.*' keys
                if unexpected:
                    raise RuntimeError(
                        f"Unexpected keys when loading wrapper extras: {unexpected}"
                    )
        else:
            state = torch.load(
                ckpt_dir / "pytorch_model.bin", map_location="cpu"
            )
            model.load_state_dict(state)
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

    # ============================================================ #
    # H. Output helpers — npy + json
    # ============================================================ #
    def save_outputs(
        self, outputs: np.ndarray, out_path: str | Path, meta: dict[str, Any]
    ) -> None:
        """Save embeddings (.npy) + metadata (.json)."""
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), outputs.astype(np.float32), allow_pickle=False)
        with open(str(p.with_suffix(".json")), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
