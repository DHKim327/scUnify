"""TaskMixin — user-friendly Mixin base class with smart defaults.

Goal: user writes 5-25 lines for a new task instead of overriding 6-8 methods.

User overrides (in order of likelihood):
- ``compute_loss(model, batch) -> Tensor``  (almost always required)
- ``predict(model, batch) -> dict``         (optional — defaults to logits/preds for CLS)
- ``build_dataset(adata) -> Dataset``       (optional — OOP standard override
                                             when user wants a fully custom dataset)
- ``train_step(batch, optimizers, accelerator)`` (optional — multi-step training, GAN/VAE)

User declares (class attrs, no override needed):
- ``task_type: str``                — "classification" | "regression" | "integration" | "custom"
- ``label_keys: list[str]``         — batch keys carrying labels
- ``n_classes: int``                — for classification head sizing
- ``model_overrides: dict``         — declarative replacement for ``setattr(cfg, ...)`` hack
- ``extra_batch_keys: list[str]``   — batch keys consumed by encode (e.g. ``pert_flag``)
- ``optimizer_groups: dict``        — for multi-optimizer training (GAN, etc.)

Smart defaults route through per-model BaseTrainer hooks so that paper-faithful
architectures are preserved across backbones:

- ``encode(model, batch)``           → ``BaseTrainer.get_cell_embedding(...)``
- ``build_model()``                  → ``BaseTrainer.build_model() + attach_task_head(...)``
- ``inject_lora(model)``             → ``BaseTrainer.inject_lora()`` (driven by
                                       ``BaseTrainer.is_backbone_param`` — head /
                                       non-backbone params kept trainable automatically)
- ``classifier_logits(model, batch)``→ ``BaseTrainer.classifier_logits(...)``

Backbone × task adjustments (e.g. scGPT: disable GEP masking for cls task) are
handled inside each ``BaseTrainer.build_dataset`` by reading ``self.task_type``
— no extra mixin hook required.

Composition: ``type("ScGPTTrainer_MyMixin", (MyMixin, ScGPTTrainer), {})``.
MRO ensures Mixin methods win; ``super().X()`` resolves to BaseTrainer.

Output format for ``get_task_output()``::

    {
        "key_name": {
            "data": tensor,              # (B, ...) per-cell output
            "storage": "obsm|obs|uns",   # where to store in adata
        },
        ...
    }

Storage types:
- ``"obsm"``: adata.obsm[f"X_{key}"] — 2D array (B, D), e.g. logits, embeddings
- ``"obs"``:  adata.obs[key]          — 1D array (B,), e.g. predictions, labels
- ``"uns"``:  adata.uns[key]          — arbitrary, e.g. metadata dict
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class TaskMixin:
    """Base class for all task Mixins.

    Subclass and override the minimum (often just ``compute_loss``).
    """

    # ---- Declarative metadata (override as class attrs) ---- #
    task_type: str = "custom"
    label_keys: list[str] = []
    n_classes: int = 0
    model_overrides: dict[str, Any] = {}
    extra_batch_keys: list[str] = []
    optimizer_groups: dict[str, list[str]] | None = None  # for multi-optimizer (GAN, etc.)

    # ---- Resolved values: yaml takes precedence over class attr ---- #
    @property
    def _label_keys(self) -> list[str]:
        return list(self.training_cfg.get("label_keys") or self.label_keys or [])

    @property
    def _n_classes(self) -> int:
        yaml_val = self.training_cfg.get("task_param", {}).get("n_classes")
        return int(yaml_val if yaml_val is not None else self.n_classes)

    # ---- Cell embedding dimension (utility) ---- #
    def _infer_emb_dim(self) -> int:
        """Infer cell embedding dimension from model architecture config.

        Reads ``cfg.model_param`` (loaded from architecture YAML) and
        returns the expected output dimension of ``get_cell_embedding()``.
        """
        model_name = self.cfg.get("model_name", "").lower().replace(" ", "")
        arch = getattr(self.cfg, "model_param", {})

        if model_name == "scgpt":
            return int(arch.get("d_model", 512))

        if model_name == "geneformer":
            variant = self.cfg.get("model", {}).get("variant", "V2-104M")
            variant_cfg = arch.get(variant, {})
            return int(variant_cfg.get("hidden_size", 512))

        if model_name == "nicheformer":
            default_cfg = arch.get("default", arch)
            return int(default_cfg.get("dim_model", 512))

        if model_name == "scfoundation":
            version = self.cfg.get("model", {}).get("version", "cell")
            hidden = int(
                arch.get(version, {})
                .get("mae_autobin", {})
                .get("encoder", {})
                .get("hidden_dim", 768)
            )
            pool_type = self.cfg.get("model", {}).get("pool_type", "all")
            return hidden * 4 if pool_type == "all" else hidden

        if model_name == "uce":
            nlayers = self.cfg.get("model", {}).get("nlayers", 4)
            layer_cfg = arch.get(nlayers, arch.get(str(nlayers), {}))
            return int(layer_cfg.get("output_dim", 1280))

        return 512

    # ---- Required (no default) ---- #
    def compute_loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__}.compute_loss is required. "
            f"Override it to return a scalar loss tensor."
        )

    # ---- Smart defaults: encoding ---- #
    def encode(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """Cell embedding (B, D) with grad. Wraps ``trainer.get_cell_embedding``."""
        return self.get_cell_embedding(model, batch)

    # ---- DDP-safe accessor for the task head attached by ``build_model`` ---- #
    @staticmethod
    def _head(model: nn.Module) -> nn.Module:
        """Return ``model.head`` regardless of DDP/FSDP wrapping.

        Use this from ``compute_loss`` / ``predict`` so user code never has to
        reach through ``model.module.head`` itself.
        """
        m = model.module if hasattr(model, "module") else model
        return m.head

    # ---- Smart defaults: model + head attach ---- #
    def _apply_model_overrides(self) -> None:
        """Declarative replacement for ``setattr(self.cfg, "model", ...)`` hack."""
        if not self.model_overrides:
            return
        cur = self.cfg.get("model", {}) or {}
        merged = dict(cur)
        merged.update(self.model_overrides)
        setattr(self.cfg, "model", merged)

    def build_head(self, emb_dim: int, n_classes: int) -> nn.Module | None:
        """Return the task head module. Override for custom heads.

        Default: delegate to the per-model trainer's paper-faithful factory
        (``BaseTrainer.default_head``) — so users who don't override get the
        paper recipe, and users who do override plug in their own ``nn.Module``.
        """
        return self.default_head(self.task_type, emb_dim, n_classes)

    def build_model(self):
        self._apply_model_overrides()
        model = super().build_model()
        emb_dim = self._infer_emb_dim()
        # Try user's ``build_head()`` first — covers both default
        # (``trainer.default_head``) and user-override cases.
        head = self.build_head(emb_dim, self._n_classes)
        if head is not None:
            m = model.module if hasattr(model, "module") else model
            m.head = head
            return model
        # Fallback: trainer-side ``attach_task_head`` for backbones that need
        # a non-trivial attachment (e.g. Geneformer's ``BertForSequenceClassification``
        # model swap).
        return self.attach_task_head(model, self.task_type, emb_dim, self._n_classes)

    # ---- Smart defaults: LoRA ---- #
    # ``inject_lora`` is provided by the per-backbone trainer; the
    # injection routine itself uses ``is_backbone_param`` to keep
    # non-backbone (head) params trainable, so no Mixin-side post-step is
    # needed. Override only if a task needs custom adapter wiring.

    # ---- Smart defaults: extraction outputs ---- #
    def predict(self, model: nn.Module, batch: dict) -> dict:
        """Default extraction predictions. Override for custom tasks.

        Should NOT include ``cell_embedding`` — the framework extracts that
        through the dedicated ``forward_embed_step`` path (yaml
        ``save.output_keys: [cell_embedding]``) so the inference-time recipe
        (no_grad, L2-normalize for BC, ...) is applied. Returning
        ``cell_embedding`` here would silently overwrite the normalised
        version with an un-normalised ``encode(model, batch)``.
        """
        if self.task_type == "classification":
            logits = self.classifier_logits(model, batch)
            return {
                "logits": {"data": logits, "storage": "obsm"},
                "predictions": {"data": logits.argmax(dim=-1), "storage": "obs"},
            }
        return {}

    def get_task_output(self, model: nn.Module, batch: dict) -> dict:
        """Per-batch task outputs (logits, predictions, custom). Cell
        embedding is intentionally **not** auto-added here — it comes from
        ``forward_embed_step`` via ``save.output_keys: [cell_embedding]``.
        """
        return self.predict(model, batch)
