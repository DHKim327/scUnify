"""scGPT Perturbation — paper-faithful framework mixin.

Cui et al. 2024 ``Tutorial_Perturbation.ipynb`` recipe::

    Loss = masked_mse_loss(mlm_output, target_values, mask=ones)

Notes
-----
- Uses ``TransformerGenerator`` (not the integration ``TransformerModel``).
  The generator is built directly inside ``build_model()`` so no extra
  wrapper class is needed: the perturbation forward signature
  ``(src, values, input_pert_flags, src_key_padding_mask, ...)`` differs
  from the integration wrapper, and ``model.pred_perturb`` is invoked
  directly by ``predict()``.

- Data path bypasses the standard adata→Dataset pipeline. The framework
  splitter cuts the input ``NORMAN_PERTURB.h5ad`` by ``obs[fold_0]``
  (``train`` / ``valid`` / ``test``) — those labels are byte-level
  identical to ``PertData`` 's simulation split (seed=1, gss=0.75, with
  ctrl→train). ``build_dataset`` returns a tiny marker carrying the
  split label; ``build_dataloader`` then returns the matching PyG loader
  from a single cached ``PertData`` instance.

Usage from yaml — single-file input via ``adata_dir``::

    model_name: scGPT
    adata_dir: /path/to/NORMAN.h5ad      # gene2go / splits / subgroup all in uns
    training:
      task: perturbation
      task_param:
        mixin: ScGPTPerturbationMixin
        # split_seed / train_gene_set_size / split_type optional (paper defaults)
      label_keys: [condition]
      split:
        fold_keys: [fold_0]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ._base import TaskMixin


# ---------------------------------------------------------------------------- #
#  Wrapper — framework's save_checkpoint / inject_lora contract expects
#  ``model.model``. Perturbation has no separate inference wrapper, so we wrap
#  TransformerGenerator in a thin nn.Module that exposes ``self.model``.
# ---------------------------------------------------------------------------- #
class _ScGPTPerturbationWrapper(nn.Module):
    """Thin wrapper exposing ``self.model = TransformerGenerator`` so the
    framework's wrapper-contract hooks (``save_checkpoint``, ``inject_lora``)
    work without modification.

    Forward + ``pred_perturb`` are delegated; the perturbation mixin reads
    ``m.model`` directly from ``compute_loss`` / ``predict`` to call the
    generator.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def pred_perturb(self, *args, **kwargs):
        return self.model.pred_perturb(*args, **kwargs)


# ---------------------------------------------------------------------------- #
#  Helpers
# ---------------------------------------------------------------------------- #
@dataclass
class _PertSplitMarker:
    """Marker dataset returned by ``build_dataset``. Carries the framework's
    split label (``train``/``valid``/``test``) so ``build_dataloader`` knows
    which PyG loader to return from the cached PertData."""

    split: str   # 'train' | 'valid' | 'test'
    n_cells: int = 0

    def __len__(self) -> int:
        return self.n_cells


# Allowed kwargs for TransformerGenerator.__init__ (filter from model yaml +
# user overrides — keys like do_dab/use_batch_labels exist on TransformerModel
# only and would crash the generator).
_TG_ACCEPTED = {
    "ntoken", "d_model", "nhead", "d_hid", "nlayers", "nlayers_cls",
    "n_cls", "vocab", "dropout", "pad_token", "pad_value", "pert_pad_id",
    "do_mvc", "domain_spec_batchnorm", "n_input_bins", "cell_emb_style",
    "mvc_decoder_style", "decoder_activation", "decoder_adaptive_bias",
    "ecs_threshold", "explicit_zero_prob", "use_fast_transformer",
    "fast_transformer_backend", "pre_norm",
}


# ---------------------------------------------------------------------------- #
#  Mixin
# ---------------------------------------------------------------------------- #
class ScGPTPerturbationMixin(TaskMixin):
    """scGPT batch-perturbation prediction mixin.

    - Single forward pass (MLM-only — no MVC/ECS/CLS heads)
    - Input ``pert_flags`` per gene (0=ctrl, 1=perturbed, 2=pad)
    - Loss: full-position MSE (``include_zero_gene='all'``, mask=ones)
    - Final-epoch prediction stored under ``adata.obsm['X_perturbation_pred']``
      when yaml requests ``save.outputs: [perturbation_pred]``,
      ``save.epochs: [final]``.
    """

    task_type = "perturbation"
    label_keys = ["condition"]

    # Tutorial recipe: dropout=0, no MVC/ECS/DAB
    model_overrides = {
        "dropout": 0.0,
        "do_mvc": False,
        "do_dab": False,
        "domain_spec_batchnorm": False,
        "use_batch_labels": False,
    }

    # Framework split label ↔ PertData internal key.
    # ``all`` is a synthetic marker used during ``save.outputs`` extraction —
    # it routes to ``all_loader`` (train+val+test concatenated) so cell
    # embeddings + perturbation_pred are produced for every cell. Paper
    # metric evaluation (compute_val_metrics) still uses ``valid`` only.
    _SPLIT_TO_PD = {"train": "train", "valid": "val", "test": "test", "all": "all"}

    # ------------------------------------------------------------------ #
    #  Freeze policy — paper-faithful (scPEFT freeze_parameters(model,
    #  DownstreamTasks.Perturbation)). The TransformerGenerator's
    #  ``decoder``, ``value_encoder``, and ``pert_encoder`` are
    #  perturbation-specific heads (not the pretrained scGPT backbone) and
    #  must stay trainable in ``probe`` / ``lora`` modes; the rest of
    #  ``model.*`` (token embedding + transformer encoder layers) is the
    #  freezable backbone.
    # ------------------------------------------------------------------ #
    _NON_BACKBONE_KEYWORDS = ("decoder", "value_encoder", "pert_encoder")

    def is_backbone_param(self, name: str) -> bool:
        if any(kw in name for kw in self._NON_BACKBONE_KEYWORDS):
            return False
        return name.startswith("model.")

    # ------------------------------------------------------------------ #
    #  Cid extraction — PyG ``Data`` carries cid as an attribute (set by
    #  ``CidPyGLoaderWrapper`` on inference loaders). Framework's
    #  ``_collect_outputs_one_pass`` calls this once per batch.
    # ------------------------------------------------------------------ #
    def extract_cid(self, batch):
        return batch.cid

    # ------------------------------------------------------------------ #
    #  Lazy-built singletons (per worker)
    # ------------------------------------------------------------------ #
    @property
    def _pert_data(self):
        """Cached ScGPTPerturbationDataset — built once per worker on first access.

        Reads NORMAN.h5ad (or equivalent) from ``cfg.adata_dir``; gene2go,
        splits, subgroup, total_count are all baked into ``adata.uns``.
        """
        if getattr(self, "_pert_data_inst", None) is not None:
            return self._pert_data_inst

        from scunify.registry.dataset.perturbation import ScGPTPerturbationDataset

        tp = self.training_cfg.get("task_param", {}) or {}
        dl_cfg = self.cfg.get("dataloader", {}) or {}
        bs = int(dl_cfg.get("batch_size", 64))
        eval_bs = int(dl_cfg.get("eval_batch_size", bs))

        pd = ScGPTPerturbationDataset(
            adata_path=str(self.cfg.adata_dir),
            split_type=str(tp.get("split_type", "simulation")),
            seed=int(tp.get("split_seed", 1)),
            train_gene_set_size=float(tp.get("train_gene_set_size", 0.75)),
        )
        pd.get_dataloader(batch_size=bs, test_batch_size=eval_bs)

        self._pert_data_inst = pd
        return pd

    @property
    def _gene_ids(self):
        """scGPT vocab ids of NORMAN var.gene_name (cached). Used by
        ``map_raw_id_to_vocab_id`` in compute_loss/predict."""
        if getattr(self, "_gene_ids_arr", None) is not None:
            return self._gene_ids_arr
        from scunify.registry.models.modules.scgpt.gene_tokenizer import GeneVocab

        vocab = GeneVocab.from_file(self.cfg.resources["vocab_file"])
        for s in ("<pad>", "<cls>", "<eoc>"):
            if s not in vocab:
                vocab.append_token(s)
        vocab.set_default_index(vocab["<pad>"])

        adata = self._pert_data.adata
        # PertData uses ``adata.var['gene_name']`` (consistent across paper recipes)
        gene_names = list(adata.var["gene_name"])
        # Tutorial cell-3: vocab[g] returns vocab["<pad>"] for unknown via default_index
        gene_ids = np.array([vocab[g] for g in gene_names], dtype=np.int64)
        self._gene_ids_arr = gene_ids
        return gene_ids

    # ------------------------------------------------------------------ #
    #  Dataset / Dataloader — bypass standard adata→Dataset path
    # ------------------------------------------------------------------ #
    def build_dataset(self, adata):
        """Return a marker carrying the split label inferred from
        ``adata.obs[fold_key]``.

        The framework splitter has already cut the h5ad into
        train/valid/test slices by ``obs.fold_0``. NORMAN_PERTURB.h5ad's
        fold_0 is byte-level identical to PertData's simulation split
        (verified): the actual sample iteration uses PertData's PyG loader.
        """
        # Trigger lazy build so ckpt-only inference also primes PertData
        _ = self._pert_data

        fold_keys = self.training_cfg.get("split", {}).get("fold_keys") or ["fold_0"]
        col = fold_keys[0]
        if col not in adata.obs.columns:
            raise KeyError(
                f"obs column '{col}' missing — perturbation needs the framework "
                f"splitter to have cut the adata first. Got obs cols: "
                f"{list(adata.obs.columns)}"
            )
        labels = adata.obs[col].astype(str).unique()
        if len(labels) != 1:
            raise ValueError(
                f"Expected single split label per build_dataset call, got {labels}"
            )
        return _PertSplitMarker(split=str(labels[0]), n_cells=len(adata))

    def build_dataloader(self, ds, *, shuffle: bool = True, drop_last: bool = True):
        """Return the matching PyG DataLoader from cached PertData.

        ``shuffle/drop_last`` are honoured only as an inference signal —
        PertData's own ``get_dataloader`` already shuffles ``train`` and
        leaves ``test`` deterministic. When the framework requests an
        inference loader (``shuffle=False``), wrap with
        :class:`CidPyGLoaderWrapper` so each batch carries a per-pass
        ``batch.cid`` for the unified extraction loop.
        """
        if not isinstance(ds, _PertSplitMarker):
            raise TypeError(
                f"ScGPTPerturbationMixin.build_dataloader expected "
                f"_PertSplitMarker, got {type(ds).__name__}"
            )
        pd_split = self._SPLIT_TO_PD[ds.split]
        loader = self._pert_data.dataloader[f"{pd_split}_loader"]
        if not shuffle:
            from scunify.registry.dataset.perturbation import CidPyGLoaderWrapper
            return CidPyGLoaderWrapper(loader)
        return loader

    def _build_inference_dataset(self, adata):
        """Inference path — return ``all`` marker (train+val+test concat) so
        ``save.outputs`` produces a perturbation_pred + cell_embedding for
        every cell (paper-faithful metric eval still uses valid via
        compute_val_metrics). Order is the PertData all_loader order
        (split-major, then condition order from set2conditions)."""
        _ = self._pert_data
        return _PertSplitMarker(split="all")

    def inference_adata(self, full_adata):
        """Build a sub-adata whose row order exactly matches the PertData
        all_loader iteration so cid (0..N-1) maps row-by-row onto the
        saved h5ad.

        Source: ``pd.adata`` (already filtered by ``_filter_pert_in_go``,
        i.e. drops the ~7 GO-graph-missing conditions). Within that:
        - split-major (train → val → test, ``set2conditions``)
        - condition order from ``set2conditions[split]``
        - within each condition, ``pd.adata`` row order (= the order
          ``create_cell_graph_dataset`` iterated when building the cache)
        """
        import numpy as np
        pd = self._pert_data
        splits_order = pd._all_split_order
        valid_conditions = set(pd.dataset_processed.keys())
        cond_arr = pd.adata.obs["condition"].astype(str).values
        indices = []
        for split in ("train", "val", "test"):
            for cond in splits_order[split]:
                if cond not in valid_conditions:
                    continue
                idx = np.where(cond_arr == cond)[0]
                indices.extend(idx.tolist())
        return pd.adata[indices].copy()

    # ------------------------------------------------------------------ #
    #  Model — TransformerGenerator (no separate wrapper class)
    # ------------------------------------------------------------------ #
    def build_model(self):
        """Build TransformerGenerator + load scGPT pretrained weights.

        Bypasses TaskMixin's default ``super().build_model()`` because
        perturbation uses a different model class than integration.
        """
        from scunify.utils import load_yaml
        from scunify.registry.models.modules.scgpt.gene_tokenizer import GeneVocab
        from scunify.registry.models.modules.scgpt.generation_model import (
            TransformerGenerator,
        )
        from scunify.registry.models._scgpt_wrapper import load_pretrained

        self._apply_model_overrides()

        # 1. Vocab (scGPT recipe)
        vocab = GeneVocab.from_file(self.cfg.resources["vocab_file"])
        for s in ("<pad>", "<cls>", "<eoc>"):
            if s not in vocab:
                vocab.append_token(s)
        vocab.set_default_index(vocab["<pad>"])

        # 2. Architecture defaults (yaml) + task overrides + paper Tutorial defaults
        model_param = dict(load_yaml(self.cfg._architecture_dir))
        model_param["ntoken"] = len(vocab)
        model_param["vocab"] = vocab
        # Tutorial_Perturbation cell-3 defaults
        model_param.setdefault("pert_pad_id", 0)
        model_param.setdefault("use_fast_transformer", True)

        # User model overrides from yaml (cfg.model after _apply_model_overrides)
        for k, v in (self.cfg.get("model", {}) or {}).items():
            model_param[k] = v

        # Filter to TransformerGenerator-accepted kwargs (drops do_dab / use_batch_labels)
        kwargs = {k: v for k, v in model_param.items() if k in _TG_ACCEPTED}

        # 3. Build + load pretrained, wrap to satisfy framework contract
        generator = TransformerGenerator(**kwargs)
        ckpt = torch.load(self.cfg.resources["model_file"], map_location="cpu")
        generator = load_pretrained(generator, ckpt, verbose=False)
        # PyTorch's ``nn.TransformerEncoder`` switches into a NestedTensor
        # fast path in ``eval()`` mode when the padding mask has no padding.
        # The LoRA-wrapped unfused MHA in ``trainer.lora._unfused_mha`` calls
        # ``query.shape`` directly which fails on NestedTensor with
        # "NestedTensorImpl doesn't support sizes". Disable the fast path
        # so eval-mode extraction (forward_embed_step + pred_perturb in
        # compute_val_metrics) works under LoRA.
        if hasattr(generator, "transformer_encoder"):
            generator.transformer_encoder.use_nested_tensor = False
        return _ScGPTPerturbationWrapper(generator)

    # ------------------------------------------------------------------ #
    #  Loss — Tutorial cell-9, byte-level
    # ------------------------------------------------------------------ #
    def compute_loss(self, model, batch):
        """Single forward → masked_mse_loss on all gene positions.

        ``batch`` is a PyG ``Data`` object from PertData. Layout (from
        Tutorial cell-9)::

            batch.x: (B*N, 2)  — col0 = gene values, col1 = pert flags
            batch.y: (B, N)    — target (perturbed) gene values
            batch.pert: list[str] — condition names
        """
        from scunify.trainer.models._scgpt_wrapper import masked_mse_loss
        from scunify.registry.models.modules.scgpt.utils import map_raw_id_to_vocab_id

        # _unwrap → DDP unwrap; .model → TransformerGenerator (paper's forward)
        m = self._unwrap(model).model
        # Accelerate's ``prepare(valid_dl)`` does not reliably move PyG ``Data``
        # batches (it expects standard tensor-dict batches), so migrate to the
        # model's device here for both train and val.
        target_device = next(m.parameters()).device
        if batch.y.device != target_device:
            batch = batch.to(target_device)
        device = batch.y.device
        batch_size = batch.y.shape[0]
        n_genes = batch.y.shape[1]

        ori_gene_values = batch.x[:, 0].view(batch_size, n_genes)
        pert_flags = batch.x[:, 1].long().view(batch_size, n_genes)
        target_gene_values = batch.y

        # include_zero_gene='all' — every gene position is a target
        input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
        max_seq_len = int(
            self.training_cfg.get("task_param", {}).get("max_seq_len", 1536)
        )
        if n_genes > max_seq_len:
            input_gene_ids = input_gene_ids[
                torch.randperm(n_genes, device=device)[:max_seq_len]
            ]

        input_values = ori_gene_values[:, input_gene_ids]
        target_values = target_gene_values[:, input_gene_ids]
        input_pert_flags = pert_flags[:, input_gene_ids]

        # Map raw indices → scGPT vocab ids. ``map_raw_id_to_vocab_id`` mirrors
        # input type — pass ndarray in, get ndarray out.
        gene_ids = self._gene_ids
        mapped = map_raw_id_to_vocab_id(input_gene_ids.cpu().numpy(), gene_ids)
        mapped_input_gene_ids = (
            torch.from_numpy(mapped).long().to(device).repeat(batch_size, 1)
        )

        src_key_padding_mask = torch.zeros_like(input_values, dtype=torch.bool)

        out = m(
            mapped_input_gene_ids,
            input_values,
            input_pert_flags,
            src_key_padding_mask=src_key_padding_mask,
            CLS=False, CCE=False, MVC=False, ECS=False,
        )

        masked_positions = torch.ones_like(input_values, dtype=torch.bool)
        return masked_mse_loss(out["mlm_output"], target_values, masked_positions)

    # ------------------------------------------------------------------ #
    #  Cell embedding extraction — transformer encoder mean-pool over genes.
    #
    #  perturbation has no dedicated CLS token (``include_zero_gene='all'``
    #  fills the full gene vocab). We reuse ``TransformerGenerator._encode``
    #  to get the encoder output and mean-pool over the sequence axis to
    #  produce a per-cell representation. This mirrors what scGPT INT does
    #  for cell-level analysis.
    # ------------------------------------------------------------------ #
    def encode(self, model, batch):
        emb, _ = self.forward_embed_step(model, batch)
        return emb

    def forward_embed_step(self, model, batch):
        from scunify.registry.models.modules.scgpt.utils import map_raw_id_to_vocab_id

        m = self._unwrap(model).model
        target_device = next(m.parameters()).device
        if batch.y.device != target_device:
            batch = batch.to(target_device)

        batch_size = batch.y.shape[0]
        n_genes = batch.y.shape[1]
        ori_gene_values = batch.x[:, 0].view(batch_size, n_genes)
        pert_flags = batch.x[:, 1].long().view(batch_size, n_genes)

        input_gene_ids = torch.arange(n_genes, device=target_device, dtype=torch.long)
        max_seq_len = int(
            self.training_cfg.get("task_param", {}).get("max_seq_len", 1536)
        )
        if n_genes > max_seq_len:
            input_gene_ids = input_gene_ids[
                torch.randperm(n_genes, device=target_device)[:max_seq_len]
            ]
        input_values = ori_gene_values[:, input_gene_ids]
        input_pert_flags = pert_flags[:, input_gene_ids]

        gene_ids = self._gene_ids
        mapped = map_raw_id_to_vocab_id(input_gene_ids.cpu().numpy(), gene_ids)
        mapped_input_gene_ids = (
            torch.from_numpy(mapped).long().to(target_device).repeat(batch_size, 1)
        )
        src_key_padding_mask = torch.zeros_like(input_values, dtype=torch.bool)

        transformer_output = m._encode(
            mapped_input_gene_ids, input_values, input_pert_flags, src_key_padding_mask
        )   # (B, seq_len, d_model)
        cell_emb = transformer_output.mean(dim=1)   # (B, d_model)
        return cell_emb, batch.cid

    # ------------------------------------------------------------------ #
    #  compute_val_metrics — paper-faithful 1-pass: val_loss + val_pearson
    #  + val_pearson_de + val_pearson_delta + val_pearson_de_delta.
    #
    #  scGPT Tutorial uses ``val_pearson`` (or ``pearson_de``) as best-ckpt
    #  selection metric (Tutorial cell-10). Use the vendored
    #  ``compute_perturbation_metrics`` (paper byte-level).
    # ------------------------------------------------------------------ #
    def compute_val_metrics(self, model, valid_dl, accelerator):
        import numpy as np
        from scunify.registry.models.modules.perturbation.inference import (
            compute_perturbation_metrics_with_de_idx,
            build_cond2de_idx,
        )

        m = self._unwrap(model).model
        target_device = next(m.parameters()).device

        was_training = model.training
        model.eval()

        losses = []
        preds, truths, pert_cats = [], [], []

        with torch.no_grad():
            for batch in valid_dl:
                with accelerator.autocast():
                    loss = self.compute_loss(model, batch)
                losses.append(loss.item())

                if batch.y.device != target_device:
                    batch = batch.to(target_device)
                pred = m.pred_perturb(
                    batch,
                    include_zero_gene="all",
                    gene_ids=self._gene_ids,
                    amp=True,
                )
                preds.append(pred.detach().cpu().numpy())
                truths.append(batch.y.detach().cpu().numpy())
                pert_cats.extend(batch.pert)

        if was_training:
            model.train()

        val_loss = sum(losses) / max(len(losses), 1)
        results = {
            "pred": np.concatenate(preds, axis=0),
            "truth": np.concatenate(truths, axis=0),
            "pert_cat": np.array(pert_cats),
        }
        # Build cond → DE-idx mapping from cell-graph cache once per worker
        # (avoids ENSG↔symbol KeyError in paper compute_perturbation_metrics).
        if not hasattr(self, "_cond2de_idx_cache"):
            self._cond2de_idx_cache = build_cond2de_idx(
                self._pert_data.dataset_processed
            )
        try:
            metrics = compute_perturbation_metrics_with_de_idx(
                results, self._pert_data.ctrl_adata, self._cond2de_idx_cache,
            )
        except Exception as e:
            return {"val_loss": val_loss, "_metric_error": str(e)}

        return {
            "val_loss": val_loss,
            "val_pearson": float(metrics.get("pearson", 0.0)),
            "val_pearson_de": float(metrics.get("pearson_de", 0.0)),
            "val_pearson_delta": float(metrics.get("pearson_delta", 0.0)),
            "val_pearson_de_delta": float(metrics.get("pearson_de_delta", 0.0)),
        }

    # ------------------------------------------------------------------ #
    #  Predict — perturbation_pred (B, n_genes); final-epoch only
    # ------------------------------------------------------------------ #
    def predict(self, model, batch):
        """Return per-batch predicted gene expression (B, n_genes).

        Uses ``model.pred_perturb`` (Tutorial-faithful) which runs the
        TransformerGenerator in eval mode and writes predictions back into
        the full gene vector. Stored under ``obsm['X_perturbation_pred']``
        when yaml requests it.
        """
        m = self._unwrap(model).model   # → TransformerGenerator with pred_perturb
        target_device = next(m.parameters()).device
        if batch.y.device != target_device:
            batch = batch.to(target_device)
        gene_ids = self._gene_ids

        include_zero_gene = "all"
        with torch.no_grad():
            pred = m.pred_perturb(
                batch,
                include_zero_gene=include_zero_gene,
                gene_ids=gene_ids,
                amp=True,
            )
        return {
            "perturbation_pred": {"data": pred, "storage": "obsm"},
        }
