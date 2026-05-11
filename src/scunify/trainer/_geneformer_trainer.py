"""Geneformer LoRA trainer — Phase 1 implementation.

Uses HF PEFT for LoRA injection (separate Q/K/V Linears).
MLM loss via BertForMaskedLM built-in head.
"""

import torch
import torch.nn as nn

from .base._basetrainer import BaseTrainer
from .dataset._geneformer_dataset import GeneformerTrainingDataset
from .lora._injection import inject_lora_to_model
from .models._geneformer_wrapper import GeneformerTrainingWrapper


class GeneformerTrainer(BaseTrainer):
    """LoRA trainer for Geneformer (BertForMaskedLM)."""

    def build_dataset(self, adata):
        return GeneformerTrainingDataset(adata, self.cfg)

    def build_model(self):
        return GeneformerTrainingWrapper(self.cfg)

    def inject_lora(self, model: nn.Module) -> nn.Module:
        return inject_lora_to_model(
            model, "geneformer", self.lora_cfg, self.freeze_cfg, self.is_backbone_param,
        )

    def compute_pretraining_loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """MLM loss — BertForMaskedLM computes CrossEntropyLoss internally."""
        return model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

    def get_cell_embedding(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """Mean-pooled hidden states (B, D). Ref: Geneformer — no CLS token."""
        m = self._unwrap(model)
        return m.get_cell_embedding(batch["input_ids"], batch["attention_mask"])

    def get_gene_embedding(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """Per-token hidden states (B, S, D)."""
        m = self._unwrap(model)
        return m.get_gene_embedding(batch["input_ids"], batch["attention_mask"])

    def _build_inference_dataset(self, adata):
        """Inference dataset (no MLM masking) for embedding extraction."""
        from ..registry.dataset import GeneformerDataset

        return GeneformerDataset(adata, self.cfg)

    def forward_embed_step(self, model, batch):
        """Single-batch embedding. labels=None → embedding mode."""
        input_ids = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        emb = model(input_ids, attn_mask)
        return emb, batch["cid"]

    # ------------------------------------------------------------------ #
    #  Gene-level helpers (standard interface)
    # ------------------------------------------------------------------ #
    def get_gene_token_ids(self, batch):
        """Geneformer batch carries token IDs as ``input_ids`` (B, S).

        Sequence layout: ``[CLS, ranked_gene_tokens..., EOS, PAD...]``. CLS/EOS/PAD
        are not in :meth:`gene_vocab`, so :meth:`align_to_adata_var` will
        NaN-pad those positions automatically.
        """
        return batch["input_ids"]

    def gene_vocab(self):
        """``{token_id: gene_symbol}`` for Geneformer.

        Geneformer's ``token_dict`` maps ``ensembl_id → token_id``, but most
        ``adata.var.index`` values are gene **symbols** (e.g. ``IKZF3``), not
        Ensembl IDs (``ENSG00000...``). We pass through the
        ``gene_name_id_file`` mapping to convert tokens → symbols so
        :meth:`align_to_adata_var` can match ``adata.var.index`` directly.

        The yaml key under ``resources.gene_dicts`` is the **dict variant tag**
        (e.g. ``"104M"``, ``"95M"``), independent of the model checkpoint
        variant (``"V2-104M"``); we strip the ``"V?-"`` prefix and any
        ``"_…"`` suffix.
        """
        if getattr(self, "_gene_vocab_cache", None) is None:
            import pickle

            res = self.cfg.get("resources", {})
            gene_dicts = res.get("gene_dicts", {})
            model_variant = self.cfg.get("model", {}).get("variant", "")
            tag = model_variant.split("-", 1)[-1].split("_", 1)[0] if "-" in model_variant else model_variant
            if tag not in gene_dicts:
                if len(gene_dicts) == 1:
                    tag = next(iter(gene_dicts))
                else:
                    raise KeyError(
                        f"Geneformer gene_dict variant {tag!r} (from model.variant "
                        f"{model_variant!r}) not in resources.gene_dicts "
                        f"keys {list(gene_dicts)}"
                    )
            with open(gene_dicts[tag]["token_dict_file"], "rb") as f:
                token_dict = pickle.load(f)  # {ensembl_id: token_id}
            # Optional: ensembl_id → gene_symbol so vocab values match adata.var.index
            id_to_symbol = {}
            name_id_path = gene_dicts[tag].get("gene_name_id_file")
            if name_id_path:
                with open(name_id_path, "rb") as f:
                    name_id = pickle.load(f)  # {gene_symbol: ensembl_id}
                id_to_symbol = {gid: sym for sym, gid in name_id.items()}
            self._gene_vocab_cache = {
                tok: id_to_symbol.get(gid, gid) for gid, tok in token_dict.items()
            }
        return self._gene_vocab_cache

    # ------------------------------------------------------------------ #
    #  Paper-faithful classifier (TaskMixin v2) — model swap, not head attach
    # ------------------------------------------------------------------ #
    def default_head(self, task_type: str, emb_dim: int, n_classes: int) -> nn.Module | None:
        """Geneformer uses an integrated classifier (HF
        ``BertForSequenceClassification``); see :meth:`attach_task_head`."""
        return None

    def attach_task_head(
        self, model: nn.Module, task_type: str, emb_dim: int, n_classes: int
    ) -> nn.Module:
        """Geneformer paper-faithful classification — swap inner model.

        Ref: Theodoris et al., Nature 2023; ``geneformer/perturber_utils.py:165``::

            BertForSequenceClassification.from_pretrained(
                model_directory,
                num_labels=num_classes,
                output_hidden_states=False,
                output_attentions=False,
            )

        Adds ``ignore_mismatched_sizes=True`` so the swap from MLM to
        classification head succeeds (paper recipe loads a checkpoint sized
        to the new task; we start from MLM weights and discard the
        head-shaped mismatches).
        """
        if task_type != "classification":
            return super().attach_task_head(model, task_type, emb_dim, n_classes)

        from transformers import BertForSequenceClassification

        variant = self.cfg.get("model", {}).get("variant", "V2-104M")
        model_dir = self.cfg.get("resources", {})["model_dirs"][variant]
        seq_model = BertForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=max(n_classes, 1),
            output_attentions=False,
            output_hidden_states=False,
            ignore_mismatched_sizes=True,
        )
        m = self._unwrap(model)
        m.model = seq_model
        return model

    def is_backbone_param(self, name: str) -> bool:
        """Geneformer's classifier head is *integrated* into ``model.model``
        (HF ``BertForSequenceClassification``: ``classifier`` / ``pooler``).
        Treat those as task heads (non-backbone) so the freeze policy
        leaves them trainable.
        """
        if not name.startswith("model."):
            return False
        return ("classifier" not in name) and ("pooler" not in name)

    def classifier_logits(
        self, model: nn.Module, batch: dict
    ) -> torch.Tensor:
        """HF integrated classifier — model itself produces logits."""
        m = self._unwrap(model)
        out = m.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        return out.logits
