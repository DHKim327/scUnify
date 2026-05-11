"""scGPT LoRA trainer — HF PEFT (QKV unfused before injection).

GEP + MVC loss via TransformerModel's ExprDecoder and MVCDecoder.
"""

import torch
import torch.nn as nn

from .base._basetrainer import BaseTrainer
from .dataset._scgpt_dataset import ScGPTTrainingDataset
from .lora._injection import inject_lora_to_model
from .models._scgpt_wrapper import ScGPTTrainingWrapper


class _ClsDecoder(nn.Module):
    """scGPT paper-faithful classification head.

    Verbatim copy of original ``ClsDecoder`` in
    ``Foundations/scGPT/scgpt/model/model.py:890-918``::

        class ClsDecoder(nn.Module):
            def __init__(self, d_model, n_cls, nlayers=3, activation=nn.ReLU):
                super().__init__()
                self._decoder = nn.ModuleList()
                for i in range(nlayers - 1):
                    self._decoder.append(nn.Linear(d_model, d_model))
                    self._decoder.append(activation())
                    self._decoder.append(nn.LayerNorm(d_model))
                self.out_layer = nn.Linear(d_model, n_cls)

    Single source of truth — used by :meth:`ScGPTTrainer.default_head` (which
    the backbone-agnostic ``ClassificationMixin`` calls automatically).
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        self._decoder = nn.ModuleList()
        for _ in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class ScGPTTrainer(BaseTrainer):
    """LoRA trainer for scGPT (GEP + MVC, fused QKV)."""

    def build_dataset(self, adata):
        ds = ScGPTTrainingDataset(adata, self.cfg)
        # CLS task: disable GEP masking (paper Tutorial_Annotation, mask_ratio=0).
        # ``self.task_type`` is provided by the composed TaskMixin (e.g.
        # ClassificationMixin sets task_type="classification").
        if getattr(self, "task_type", None) == "classification":
            ds._base_collator.do_mlm = False
            ds._base_collator.mlm_probability = 0.0
        return ds

    def build_model(self):
        return ScGPTTrainingWrapper(self.cfg)

    def inject_lora(self, model: nn.Module) -> nn.Module:
        return inject_lora_to_model(
            model, "scgpt", self.lora_cfg, self.freeze_cfg, self.is_backbone_param,
        )

    def compute_pretraining_loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """GEP + MVC loss on masked gene expression positions."""
        pad_token_id = batch["pad_token_id"]
        src_key_padding_mask = batch["gene"].eq(pad_token_id)
        return model(
            gene=batch["gene"],
            masked_expr=batch["masked_expr"],
            src_key_padding_mask=src_key_padding_mask,
            target_values=batch["expr"],
        )

    def get_cell_embedding(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """CLS token embedding (B, D). Ref: scGPT Tutorial_Annotation.

        DSBN-aware: when the model was built with ``domain_spec_batchnorm=True``
        and ``batch["batch_id"]`` is present, ``batch_labels`` are passed to the
        encoder so DSBN picks the right per-batch BN stats. Users writing their
        own task mixin never have to touch ``batch_labels`` themselves.
        """
        m = self._unwrap(model)
        pad_mask = batch["gene"].eq(batch["pad_token_id"])
        kwargs = {}
        # Auto-pass batch_labels when DSBN is enabled and the data carries them.
        inner = getattr(m, "model", None)
        if inner is not None and getattr(inner, "domain_spec_batchnorm", False):
            bl = batch.get("batch_id")
            if bl is not None:
                kwargs["batch_labels"] = bl.long()
        return m.get_cell_embedding(
            batch["gene"], batch["masked_expr"], pad_mask, **kwargs
        )

    def get_gene_embedding(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """Per-gene hidden states (B, S, D)."""
        m = self._unwrap(model)
        pad_mask = batch["gene"].eq(batch["pad_token_id"])
        return m.get_gene_embedding(batch["gene"], batch["masked_expr"], pad_mask)

    # ------------------------------------------------------------------ #
    #  Paper-faithful head factory (TaskMixin v2)
    # ------------------------------------------------------------------ #
    def default_head(self, task_type: str, emb_dim: int, n_classes: int) -> nn.Module | None:
        """scGPT paper-faithful heads.

        - classification: ``_ClsDecoder`` (3-layer Linear-ReLU-LN + Linear).
          Ref: scGPT Tutorial_Annotation, Cui et al. Nature Methods 2024.
        - other tasks: not yet provided here; returns None → TaskMixin v2
          generic fallback or user-supplied head.
        """
        if task_type == "classification":
            nlayers = int(self.training_cfg.get("task_param", {}).get("nlayers", 3))
            return _ClsDecoder(emb_dim, max(n_classes, 1), nlayers)
        return None

    def attach_task_head(
        self, model: nn.Module, task_type: str, emb_dim: int, n_classes: int
    ) -> nn.Module:
        """scGPT classification: remove built-in cls_decoder, attach paper head.

        scGPT's pretrained ``TransformerModel`` ships with a built-in
        ``cls_decoder`` (n_cls=1). When fine-tuning for a multi-class task we
        replace it with a fresh ``_ClsDecoder`` of the correct size; we also
        clear the built-in to avoid wasted parameters under full-FT.
        """
        if task_type == "classification":
            m = self._unwrap(model)
            if hasattr(m.model, "cls_decoder"):
                m.model.cls_decoder = None
        return super().attach_task_head(model, task_type, emb_dim, n_classes)

    # ------------------------------------------------------------------ #
    #  Embedding extraction (distributed, via BaseTrainer)
    # ------------------------------------------------------------------ #
    def _build_inference_dataset(self, adata):
        """Inference dataset (no masking) for post-training embedding extraction."""
        from ..registry.dataset import ScGPTDataset

        return ScGPTDataset(adata, self.cfg)

    def forward_embed_step(self, model, batch):
        """Single-batch embedding. Inference dataset uses same batch format."""
        gene = batch["gene"]
        expr = batch["expr"]
        pad_token_id = batch["pad_token_id"]
        src_key_padding_mask = gene.eq(pad_token_id)
        emb = model(
            gene=gene,
            masked_expr=expr,
            src_key_padding_mask=src_key_padding_mask,
        )  # target_values=None → embedding mode
        return emb, batch["cid"]

    # ------------------------------------------------------------------ #
    #  Gene-level helpers (standard interface)
    # ------------------------------------------------------------------ #
    def get_gene_token_ids(self, batch):
        """scGPT batch carries token IDs directly under ``batch["gene"]``."""
        return batch["gene"]

    def gene_vocab(self):
        """``{token_id: gene_symbol}`` for the scGPT vocab (cached)."""
        if getattr(self, "_gene_vocab_cache", None) is None:
            from ..registry.models.modules.scgpt.gene_tokenizer import GeneVocab

            vocab_path = self.cfg.get("resources", {})["vocab_file"]
            v = GeneVocab.from_file(vocab_path)
            # GeneVocab is a torchtext Vocab — use ``get_itos()`` to map idx→symbol
            self._gene_vocab_cache = {i: s for i, s in enumerate(v.get_itos())}
        return self._gene_vocab_cache
