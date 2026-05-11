"""scGPT Batch Correction (Integration) — paper-faithful framework mixin.

Cui et al. 2024 Tutorial_Integration.ipynb recipe::

    Loss = L_GEP(+MVC) + ecs_weight * L_ECS + dab_weight * L_DAB

  - GEP : Masked gene expression prediction (pretrained ExprDecoder)
  - MVC : Masked value prediction for cell embedding (pretrained MVCDecoder)
  - ECS : Elastic Cell Similarity (cosine similarity regularization, ×10)
  - DAB : Domain Adaptation via reverse backprop (AdversarialDiscriminator)

Paper defaults (override via yaml ``training.task_param``):
  DSBN = True, explicit_zero_prob = True, mask_ratio = 0.4,
  ecs_thres = 0.8, dab_weight = 1.0, MVC = True

Usage from yaml — no user mixin required::

    training:
      task_param:
        mixin: ScGPTIntegrationMixin
      label_keys: [celltype, batch_id]
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ._base import TaskMixin
# NOTE: ``masked_mse_loss`` is lazy-imported inside ``compute_loss``. Eager
# import would pull scGPT model deps (torchtext, etc.) into the launching
# env (``scUnify``); those dependencies stay confined to ``scunify_scgpt``
# where the Ray training worker actually runs.


def _neg_log_bernoulli(input, target, mask):
    """Negative log-likelihood of Bernoulli. Ref: scGPT loss.py"""
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


class ScGPTIntegrationMixin(TaskMixin):
    """scGPT batch correction. All losses computed in a single forward pass
    to avoid DDP gradient conflicts.
    """

    task_type = "integration"
    label_keys = ["celltype", "batch_id"]
    extra_batch_keys = ["batch_id"]   # consumed by encode (DSBN)

    MASK_VALUE = -1   # scGPT convention

    # Paper recipe — declarative model setup (replaces setattr(cfg, "model", ...) hack).
    # ``num_batch_labels`` is set dynamically in build_dataset (depends on adata).
    model_overrides = {
        "do_dab": True,
        "do_mvc": True,
        "domain_spec_batchnorm": True,
        "explicit_zero_prob": True,
        "use_batch_labels": True,
        "ecs_threshold": 0.8,
    }

    # ------------------------------------------------------------------ #
    #  Dataset — only adds num_batch_labels (rest delegated to trainer)
    # ------------------------------------------------------------------ #
    def build_dataset(self, adata):
        keys = self._label_keys
        if len(keys) < 2:
            raise ValueError(
                f"{type(self).__name__} requires label_keys=[celltype, batch_id]"
            )
        # Wire num_batch_labels for DSBN/DAB before model build.
        # ``model_overrides`` is consumed by TaskMixin._apply_model_overrides
        # in build_model().
        self.model_overrides = {
            **self.model_overrides,
            "num_batch_labels": int(adata.obs[keys[1]].nunique()),
        }
        return super().build_dataset(adata)

    # ------------------------------------------------------------------ #
    #  Dataloader — paper recipe ``per_seq_batch_sample = True``
    # ------------------------------------------------------------------ #
    def build_dataloader(self, ds, *, shuffle: bool = True, drop_last: bool = True):
        """Per-domain batched dataloader for DSBN-aware training.

        Paper recipe (``finetune_integration.py:96`` ``per_seq_batch_sample =
        True``): every minibatch contains cells from a **single** ``batch_id``
        so DSBN's per-domain BatchNorm running stats are well-defined. We
        wrap the dataset with :class:`SubsetsBatchSampler` (reproduced from
        ``Foundations/scGPT/scgpt/data_sampler.py``).

        Falls back to the BaseTrainer default DataLoader when the dataset
        does not carry ``batch_id`` (e.g. embedding-only inference path).

        Set yaml ``training.task_param.per_seq_batch_sample: false`` to opt
        out (mixed minibatches — non paper-faithful for BC).
        """
        from torch.utils.data import DataLoader

        from ..dataset._samplers import SubsetsBatchSampler

        per_seq = bool(
            self.training_cfg.get("task_param", {})
            .get("per_seq_batch_sample", True)
        )
        batch_id_arr = (
            ds._label_arrays.get(self._label_keys[1])
            if per_seq and hasattr(ds, "_label_arrays")
            else None
        )
        if batch_id_arr is None:
            return super().build_dataloader(ds, shuffle=shuffle, drop_last=drop_last)

        import numpy as np

        dl_cfg = self.cfg.get("dataloader", {})
        bs = int(dl_cfg.get("batch_size", 32))
        nw = int(dl_cfg.get("num_workers", 0))
        collator = getattr(ds, "collator", None)

        unique_bids = np.unique(batch_id_arr)
        subsets = [
            np.where(batch_id_arr == bid)[0].tolist() for bid in unique_bids
        ]

        sampler = SubsetsBatchSampler(
            subsets=subsets,
            batch_size=bs,
            intra_subset_shuffle=shuffle,
            inter_subset_shuffle=shuffle,
            drop_last=drop_last,
        )

        dl_kwargs = dict(
            batch_sampler=sampler,
            num_workers=nw,
            collate_fn=collator,
            pin_memory=True,
        )
        if nw > 0:
            dl_kwargs["persistent_workers"] = True
            dl_kwargs["prefetch_factor"] = int(dl_cfg.get("prefetch_factor", 4))

        return DataLoader(ds, **dl_kwargs)

    # ------------------------------------------------------------------ #
    #  Loss — single forward → GEP + MVC + ECS + DAB
    # ------------------------------------------------------------------ #
    def compute_loss(self, model, batch):
        from scunify.trainer.models._scgpt_wrapper import masked_mse_loss

        m = self._unwrap(model)
        pad_mask = batch["gene"].eq(batch["pad_token_id"])

        batch_labels = batch.get("batch_id")
        if batch_labels is not None:
            batch_labels = batch_labels.long()

        out = m.model(
            batch["gene"],
            batch["masked_expr"],
            src_key_padding_mask=pad_mask,
            batch_labels=batch_labels,
            MVC=hasattr(m.model, "mvc_decoder"),
            ECS=True,
        )

        # GEP
        masked = batch["masked_expr"].eq(self.MASK_VALUE)
        loss = masked_mse_loss(out["mlm_output"], batch["expr"], masked)
        if "mlm_zero_probs" in out:
            loss = loss + _neg_log_bernoulli(out["mlm_zero_probs"], batch["expr"], masked)

        # MVC
        if "mvc_output" in out:
            loss = loss + masked_mse_loss(out["mvc_output"], batch["expr"], masked)
            if "mvc_zero_probs" in out:
                loss = loss + _neg_log_bernoulli(out["mvc_zero_probs"], batch["expr"], masked)

        # Weights from yaml (default = paper recipe)
        tp = self.training_cfg.get("task_param", {})
        ecs_w = float(tp.get("ecs_weight", 10.0))
        dab_w = float(tp.get("dab_weight", 1.0))

        # ECS
        loss = loss + ecs_w * out.get("loss_ecs", torch.tensor(0.0, device=pad_mask.device))

        # DAB (cross-entropy on grad-reverse discriminator output)
        keys = self._label_keys
        loss = loss + dab_w * F.cross_entropy(out["dab_output"], batch[keys[1]].long())
        return loss

    # ------------------------------------------------------------------ #
    #  Extraction — DSBN-aware encode, normalised CLS
    # ------------------------------------------------------------------ #
    def _encode_with_batch(self, model, batch):
        m = self._unwrap(model)
        pad_mask = batch["gene"].eq(batch["pad_token_id"])
        batch_labels = batch.get("batch_id")
        if batch_labels is not None:
            batch_labels = batch_labels.long()
        hidden = m.model._encode(
            batch["gene"], batch["masked_expr"], pad_mask, batch_labels=batch_labels
        )
        return hidden[:, 0, :]   # CLS token

    def encode(self, model, batch):
        """DSBN-aware override of TaskMixin.encode (used by get_task_output's
        auto cell_embedding output). scGPT's default encode skips
        batch_labels and trips DSBN's assert."""
        return self._encode_with_batch(model, batch)

    def forward_embed_step(self, model, batch):
        with torch.no_grad():
            emb = F.normalize(self._encode_with_batch(model, batch), p=2, dim=1)
        return emb, batch["cid"]

    def predict(self, model, batch):
        """Per-batch DAB output (discriminator logits + argmax preds).

        Cell embedding is intentionally NOT returned here — extraction is
        driven by ``save.outputs: [cell_embedding]`` (framework runs
        ``forward_embed_step`` with the paper-faithful L2-normalisation).
        Mid-epoch snapshots also flow through that path automatically when
        the epoch is in ``save.epochs``; no per-mixin buffer is needed.
        """
        m = self._unwrap(model)
        cell_emb = self._encode_with_batch(model, batch)
        dab_logits = m.model.grad_reverse_discriminator(cell_emb)
        return {
            "batch_logits": {"data": dab_logits, "storage": "obsm"},
            "batch_pred":   {"data": dab_logits.argmax(dim=-1), "storage": "obs"},
        }

    def _build_inference_dataset(self, adata):
        """INT inference uses TRAINING dataset (which carries batch_id) with
        masking disabled. DSBN requires batch_labels during embedding extraction."""
        from scunify.trainer.dataset import ScGPTTrainingDataset

        ds = ScGPTTrainingDataset(adata, self.cfg)
        ds._base_collator.do_mlm = False
        ds._base_collator.mlm_probability = 0.0
        return ds
