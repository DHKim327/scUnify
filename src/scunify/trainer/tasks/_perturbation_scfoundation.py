"""scFoundation Perturbation — paper-faithful framework mixin.

scFoundation perturbation = GEARS pipeline with the scFoundation backbone
loaded as ``MAEAutobinencoder``. Recipe (RelatedWorks/Foundations/scFoundation/
GEARS/gears/gears.py default ``model_initialize`` + ``train``)::

    model = GEARS_Model(args=
        hidden_size=64, num_go_gnn_layers=1, num_gene_gnn_layers=1,
        decoder_hidden_size=16, coexpress_threshold=0.4,
        direction_lambda=1e-1, model_type='maeautobin',
        load_path=<scFoundation pretrained ckpt>,
    )
    Loss = loss_fct(pred, y, batch.pert,
                    ctrl=pert_data.ctrl_expression,
                    dict_filter=pert_data.dict_filter,
                    direction_lambda=1e-1)
    optimizer = Adam(lr=1e-3, weight_decay=5e-4)
    scheduler = StepLR(step_size=1, gamma=0.5)

Notes
-----
- Builds GEARS_Model directly inside ``build_model()``; the GEARS top-level
  class is reused only for its ``model_initialize`` graph-construction logic
  (co-expression graph from train conditions, GO graph from ``gene2go.pkl``).
- Wrapper exposes ``self.model = GEARS_Model`` so framework
  ``save_checkpoint`` / ``inject_lora`` contracts hold.

Usage from yaml — single-file input via ``adata_dir``::

    model_name: scFoundation
    adata_dir: /path/to/NORMAN.h5ad     # gene2go / splits / subgroup all in uns
    training:
      task: perturbation
      task_param:
        mixin: ScFoundationPerturbationMixin
        hidden_size: 512                # match scFoundation backbone dim
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
#  Wrapper — same contract as ScGPTPerturbationMixin (model.model = GEARS_Model)
# ---------------------------------------------------------------------------- #
class _ScFoundationPerturbationWrapper(nn.Module):
    """Thin wrapper exposing ``self.model = GEARS_Model`` for framework's
    save_checkpoint / inject_lora contracts. forward delegated."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


# ---------------------------------------------------------------------------- #
#  Helpers
# ---------------------------------------------------------------------------- #
@dataclass
class _PertSplitMarker:
    split: str   # 'train' | 'valid' | 'test'
    n_cells: int = 0

    def __len__(self) -> int:
        return self.n_cells


# ---------------------------------------------------------------------------- #
#  Mixin
# ---------------------------------------------------------------------------- #
class ScFoundationPerturbationMixin(TaskMixin):
    """scFoundation perturbation prediction mixin.

    GEARS_Model with scFoundation backbone (``MAEAutobinencoder``).
    Loss = MSE + direction-aware penalty (``loss_fct``).
    """

    task_type = "perturbation"
    label_keys = ["condition"]

    # GEARS perturbation has no integration / cls heads.
    model_overrides = {}

    # ``all`` synthetic marker → all_loader (train+val+test concat) for
    # save.outputs extraction.
    _SPLIT_TO_PD = {"train": "train", "valid": "val", "test": "test", "all": "all"}

    # ------------------------------------------------------------------ #
    #  Freeze policy — only ``singlecell_model`` is the freezable backbone.
    #  Everything else under ``model.*`` (GEARS heads: ``pert_emb``,
    #  ``sim_layers``, ``transform``, ``recovery_w``, ``emb_pos``, etc.) is
    #  the task head and stays trainable. Drives ``mode: probe`` (paper
    #  recipe ``finetune_method='frozen'``) and partial-FT.
    # ------------------------------------------------------------------ #
    def is_backbone_param(self, name: str) -> bool:
        return "singlecell_model" in name

    # ------------------------------------------------------------------ #
    #  Cid extraction — PyG ``Data`` carries cid as an attribute (set by
    #  ``CidPyGLoaderWrapper`` on inference loaders).
    # ------------------------------------------------------------------ #
    def extract_cid(self, batch):
        return batch.cid

    # ------------------------------------------------------------------ #
    #  Lazy build — PertData (graph-aware dataset) and GEARS wrapper
    # ------------------------------------------------------------------ #
    @property
    def _pert_data(self):
        """Cached ScFoundationPerturbationDataset — 19264-gene expanded
        adata + (n_genes+1, 1) cell graphs. ``cfg.adata_dir`` points at the
        original NORMAN.h5ad (5045-gene); the 19264 expand is performed in
        ``_expand_adata`` and cached as a sister ``.expanded_19264.h5ad``.
        """
        if getattr(self, "_pert_data_inst", None) is not None:
            return self._pert_data_inst

        from scunify.registry.dataset.perturbation import ScFoundationPerturbationDataset

        tp = self.training_cfg.get("task_param", {}) or {}
        dl_cfg = self.cfg.get("dataloader", {}) or {}
        bs = int(dl_cfg.get("batch_size", 32))
        eval_bs = int(dl_cfg.get("eval_batch_size", 128))

        pd = ScFoundationPerturbationDataset(
            adata_path=str(self.cfg.adata_dir),
            split_type=str(tp.get("split_type", "simulation")),
            seed=int(tp.get("split_seed", 1)),
            train_gene_set_size=float(tp.get("train_gene_set_size", 0.75)),
        )
        pd.get_dataloader(batch_size=bs, test_batch_size=eval_bs)

        self._pert_data_inst = pd
        return pd

    @property
    def _gears(self):
        """Cached GEARS wrapper (ctrl_expression, dict_filter, graph-built model)."""
        if getattr(self, "_gears_inst", None) is not None:
            return self._gears_inst

        from scunify.registry.models.modules.perturbation.gears import GEARS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tp = self.training_cfg.get("task_param", {}) or {}

        g = GEARS(self._pert_data, device=device)

        # Paper-faithful defaults (RelatedWorks/.../gears/gears.py:93-145).
        # Override via training.task_param.<name>.
        g.model_initialize(
            hidden_size=int(tp.get("hidden_size", 64)),
            num_go_gnn_layers=int(tp.get("num_go_gnn_layers", 1)),
            num_gene_gnn_layers=int(tp.get("num_gene_gnn_layers", 1)),
            decoder_hidden_size=int(tp.get("decoder_hidden_size", 16)),
            num_similar_genes_go_graph=int(tp.get("num_similar_genes_go_graph", 20)),
            num_similar_genes_co_express_graph=int(
                tp.get("num_similar_genes_co_express_graph", 20)
            ),
            coexpress_threshold=float(tp.get("coexpress_threshold", 0.4)),
            uncertainty=bool(tp.get("uncertainty", False)),
            uncertainty_reg=float(tp.get("uncertainty_reg", 1)),
            direction_lambda=float(tp.get("direction_lambda", 1e-1)),
            no_perturb=bool(tp.get("no_perturb", False)),
            cell_fitness_pred=bool(tp.get("cell_fitness_pred", False)),
            model_type=str(tp.get("model_type", "maeautobin")),
            bin_set=tp.get("bin_set", "autobin_resolution_append"),
            load_path=self.cfg.resources.get("model_file"),
            finetune_method=tp.get("finetune_method", None),
            mode=str(tp.get("gears_mode", "v1")),
            accumulation_steps=int(tp.get("accumulation_steps", 1)),
            highres=int(tp.get("highres", 0)),
        )

        self._gears_inst = g
        return g

    # ------------------------------------------------------------------ #
    #  Dataset / Dataloader — same pattern as scgpt mixin
    # ------------------------------------------------------------------ #
    def build_dataset(self, adata):
        _ = self._pert_data
        fold_keys = self.training_cfg.get("split", {}).get("fold_keys") or ["fold_0"]
        col = fold_keys[0]
        if col not in adata.obs.columns:
            raise KeyError(
                f"obs column '{col}' missing — perturbation needs the framework "
                f"splitter to have cut the adata first."
            )
        labels = adata.obs[col].astype(str).unique()
        if len(labels) != 1:
            raise ValueError(
                f"Expected single split label per build_dataset call, got {labels}"
            )
        return _PertSplitMarker(split=str(labels[0]), n_cells=len(adata))

    def build_dataloader(self, ds, *, shuffle: bool = True, drop_last: bool = True):
        if not isinstance(ds, _PertSplitMarker):
            raise TypeError(
                f"ScFoundationPerturbationMixin.build_dataloader expected "
                f"_PertSplitMarker, got {type(ds).__name__}"
            )
        pd_split = self._SPLIT_TO_PD[ds.split]
        loader = self._pert_data.dataloader[f"{pd_split}_loader"]
        if not shuffle:
            from scunify.registry.dataset.perturbation import CidPyGLoaderWrapper
            return CidPyGLoaderWrapper(loader)
        return loader

    def _build_inference_dataset(self, adata):
        """Inference path — ``all`` marker (train+val+test) so cell embedding
        + perturbation_pred are produced for every cell."""
        _ = self._pert_data
        return _PertSplitMarker(split="all")

    def inference_adata(self, full_adata):
        """Sub-adata in PertData all_loader order (train → val → test;
        condition order from set2conditions; within-condition row order
        preserved). Source = ``pd.adata`` which is the 19264-expanded
        adata (scFoundation backbone vocab) already filtered for
        GO-graph-missing conditions."""
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
    #  Model — GEARS wrapper builds graphs + GEARS_Model w/ scFoundation backbone
    # ------------------------------------------------------------------ #
    def build_model(self):
        self._apply_model_overrides()
        return _ScFoundationPerturbationWrapper(self._gears.model)

    # ------------------------------------------------------------------ #
    #  Loss — gears.utils.loss_fct (MSE + direction penalty + dict_filter)
    # ------------------------------------------------------------------ #
    def compute_loss(self, model, batch):
        """GEARS perturbation loss (paper-faithful).

        Mirrors ``RelatedWorks/.../gears/gears.py:381`` train-step::

            loss = loss_fct(pred, y, batch.pert,
                            ctrl=ctrl_expression,
                            dict_filter=dict_filter,
                            direction_lambda=1e-1)
        """
        from scunify.registry.models.modules.perturbation import loss_fct

        m = self._unwrap(model).model
        target_device = next(m.parameters()).device
        if batch.y.device != target_device:
            batch = batch.to(target_device)
        pred = m(batch)

        tp = self.training_cfg.get("task_param", {}) or {}
        direction_lambda = float(tp.get("direction_lambda", 1e-1))

        gears = self._gears
        loss = loss_fct(
            pred,
            batch.y,
            batch.pert,
            ctrl=gears.ctrl_expression,
            dict_filter=gears.dict_filter,
            direction_lambda=direction_lambda,
        )
        return loss

    # ------------------------------------------------------------------ #
    #  Cell embedding extraction — scFoundation backbone output mean-pool
    #  over the gene axis. We replicate ``GEARS_Model.forward``'s backbone
    #  path (singlecell_model call), then pool over genes to a per-cell
    #  ``(B, hidden_size)`` representation. Useful for comparing pre/post
    #  fine-tuning cell representations across modes (LoRA family will
    #  shift backbone activations; probe mode keeps them constant).
    # ------------------------------------------------------------------ #
    def encode(self, model, batch):
        emb, _ = self.forward_embed_step(model, batch)
        return emb

    def forward_embed_step(self, model, batch):
        m = self._unwrap(model).model
        target_device = next(m.parameters()).device
        if batch.y.device != target_device:
            batch = batch.to(target_device)

        if not getattr(m, "pretrained", False):
            raise RuntimeError(
                "ScFoundationPerturbationMixin.forward_embed_step expects a "
                "pretrained ``singlecell_model`` backbone (model_type='maeautobin'). "
                "Got vanilla GEARS."
            )

        num_graphs = int(batch.batch.max().item()) + 1
        pre_in = batch.x.clone().reshape(num_graphs, m.num_genes + 1)
        emb = m.singlecell_model(pre_in)   # (B*n_genes, hidden) or (B, n_genes, hidden)

        # Normalise shape to (B, n_genes, hidden) → mean pool over genes
        if emb.dim() == 2:
            cell_emb = emb.view(num_graphs, -1, m.hidden_size).mean(dim=1)
        elif emb.dim() == 3:
            cell_emb = emb.mean(dim=1)
        else:
            raise RuntimeError(
                f"Unexpected singlecell_model output shape {tuple(emb.shape)}; "
                f"expected 2D or 3D tensor."
            )
        return cell_emb, batch.cid

    # ------------------------------------------------------------------ #
    #  compute_val_metrics — paper-faithful (GEARS gears.py:435 best-ckpt
    #  uses ``val_metrics['mse_de']``). Returns val_loss + val_mse_de +
    #  val_mse + val_pearson + val_pearson_de via vendored
    #  ``inference.evaluate`` + ``compute_metrics``.
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
        # mse / mse_de aggregated online — cell-level de_idx length varies
        # (15-20) because ENSG↔symbol mygene mapping is ~89% hit. Stacking
        # variable-length tensors fails; aggregate per-cell instead and avoid
        # the GEARS ``compute_metrics`` path (which assumes fixed-shape DE
        # arrays). The pearson family stays on
        # ``compute_perturbation_metrics_with_de_idx`` which uses the
        # condition-level cache de_idx and is unaffected by length variance.
        mse_total_sum = 0.0
        mse_total_count = 0
        mse_de_per_cell: list[float] = []

        with torch.no_grad():
            for batch in valid_dl:
                with accelerator.autocast():
                    loss = self.compute_loss(model, batch)
                losses.append(loss.item())

                if batch.y.device != target_device:
                    batch = batch.to(target_device)
                pred = m(batch)   # (B, n_genes)
                truth = batch.y

                preds.append(pred.detach().cpu())
                truths.append(truth.detach().cpu())
                pert_cats.extend(batch.pert)

                diff_sq = (pred - truth).pow(2)
                mse_total_sum += diff_sq.sum().item()
                mse_total_count += diff_sq.numel()
                for i, de_idx in enumerate(batch.de_idx):
                    if len(de_idx) == 0:
                        continue
                    di = torch.as_tensor(de_idx, dtype=torch.long, device=pred.device)
                    di = di[di >= 0]   # drop sentinel -1
                    if di.numel() == 0:
                        continue
                    mse_de_per_cell.append(
                        diff_sq[i].index_select(0, di).mean().item()
                    )

        if was_training:
            model.train()

        val_loss = sum(losses) / max(len(losses), 1)
        val_mse = mse_total_sum / max(mse_total_count, 1)
        val_mse_de = (
            float(np.mean(mse_de_per_cell)) if mse_de_per_cell else float("inf")
        )

        full_results = {
            "pred": np.concatenate([t.numpy() for t in preds], axis=0),
            "truth": np.concatenate([t.numpy() for t in truths], axis=0),
            "pert_cat": np.array(pert_cats),
        }
        if not hasattr(self, "_cond2de_idx_cache"):
            self._cond2de_idx_cache = build_cond2de_idx(
                self._pert_data.dataset_processed
            )
        try:
            metrics_p = compute_perturbation_metrics_with_de_idx(
                full_results, self._pert_data.ctrl_adata, self._cond2de_idx_cache,
            )
        except Exception as e:
            return {"val_loss": val_loss, "_metric_error": str(e)}

        return {
            "val_loss": val_loss,
            "val_mse": val_mse,
            "val_mse_de": val_mse_de,
            "val_pearson": float(metrics_p.get("pearson", 0.0)),
            "val_pearson_de": float(metrics_p.get("pearson_de", 0.0)),
            "val_pearson_delta": float(metrics_p.get("pearson_delta", 0.0)),
            "val_pearson_de_delta": float(metrics_p.get("pearson_de_delta", 0.0)),
        }

    # ------------------------------------------------------------------ #
    #  Predict — perturbation_pred (B, n_genes); final-epoch only
    # ------------------------------------------------------------------ #
    def predict(self, model, batch):
        """Return per-batch predicted gene expression (B, n_genes).

        For GEARS_Model, the forward output IS the prediction (no separate
        ``pred_perturb`` method like scGPT).
        """
        m = self._unwrap(model).model
        target_device = next(m.parameters()).device
        if batch.y.device != target_device:
            batch = batch.to(target_device)
        with torch.no_grad():
            pred = m(batch)
        return {
            "perturbation_pred": {"data": pred, "storage": "obsm"},
        }
