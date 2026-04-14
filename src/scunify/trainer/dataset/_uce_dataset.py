"""Training dataset for UCE — inherits from registry, adds masked BEP targets.

Follows the original UCE training procedure (Rosen et al. 2023):

1. Mask ``r_mask`` (default 20%) of expressed genes before sampling.
2. Build cell sentence from **remaining** expressed genes (masked genes excluded).
3. Positive targets (GL+) sampled from **masked** expressed genes.
4. Negative targets (GL-) sampled from non-expressed genes.
5. Loss: BCE on binary expression prediction (cell_emb || gene_emb → decoder).
"""

import logging

import numpy as np
import torch

from ...registry.dataset._uce_dataset import (
    UCEDataset,
    _row_from_X,
    sample_cell_sentences,
)

logger = logging.getLogger(__name__)


class UCETrainingDataset(UCEDataset):
    """UCE training dataset with masked binary expression prediction.

    Overrides ``__getitem__`` to:
    1. Mask 20% of expressed genes before cell sentence sampling.
    2. Sample positive targets from masked genes (model never saw them).
    3. Sample negative targets from non-expressed genes.
    """

    def __init__(self, adata, config):
        super().__init__(adata, config)
        training_cfg = config.get("training", {})
        bep_cfg = training_cfg.get("bep", {})

        # Label passthrough from adata.obs
        self._label_arrays = {}
        for key in training_cfg.get("label_keys", []):
            if key in adata.obs.columns:
                col = adata.obs[key]
                self._label_arrays[key] = (
                    col.cat.codes.values.copy()
                    if hasattr(col, "cat")
                    else col.values.copy()
                )
        self.n_pos = int(bep_cfg.get("n_pos_targets", 100))
        self.n_neg = int(bep_cfg.get("n_neg_targets", 100))
        self.mask_ratio = float(bep_cfg.get("mask_ratio", 0.2))

        # Override collator with training-aware version
        self.collator = UCETrainingCollator(
            self.args.pad_length,
            label_keys=list(self._label_arrays.keys()),
        )

    def __getitem__(self, idx):
        # ── Read counts and identify expressed / non-expressed genes ──
        counts_row = _row_from_X(self.X, idx).astype(np.float32)
        expressed_idx = np.nonzero(counts_row)[0]
        non_expressed_idx = np.where(counts_row == 0)[0]

        # ── Step 1: Mask r_mask% of expressed genes ──
        n_mask = max(1, int(len(expressed_idx) * self.mask_ratio))
        mask_sel = np.random.choice(len(expressed_idx), size=n_mask, replace=False)
        masked_gene_idx = expressed_idx[mask_sel]

        # ── Step 2: Build weights with masked genes zeroed out ──
        counts = torch.from_numpy(counts_row).unsqueeze(0)  # (1, G)
        weights = torch.log1p(counts)
        weights[0, masked_gene_idx] = 0.0  # exclude masked genes from sampling
        s = torch.sum(weights)
        weights = weights / s if s.item() > 0 else torch.full_like(weights, 1.0 / weights.shape[-1])

        # ── Step 3: Sample cell sentence (masked genes excluded) ──
        batch_sentences, base_mask, seq_len, _ = sample_cell_sentences(
            counts,
            weights,
            self.name,
            self.args,
            dataset_to_protein_embeddings=self.dataset_to_protein_embeddings,
            dataset_to_chroms=self.dataset_to_chroms,
            dataset_to_starts=self.dataset_to_starts,
        )

        # ── Step 4: Positive targets from masked expressed genes ──
        pe_map = self.dataset_to_protein_embeddings[self.name]

        if len(masked_gene_idx) >= self.n_pos:
            pos_sel = np.random.choice(masked_gene_idx, size=self.n_pos, replace=False)
        else:
            # Paper: use all masked + sample extra from full expressed set (with replacement)
            extra = np.random.choice(expressed_idx, size=self.n_pos - len(masked_gene_idx), replace=True)
            pos_sel = np.concatenate([masked_gene_idx, extra])
        pos_pe = pe_map[pos_sel]

        # ── Step 5: Negative targets from non-expressed genes ──
        if len(non_expressed_idx) >= self.n_neg:
            neg_sel = np.random.choice(non_expressed_idx, size=self.n_neg, replace=False)
        else:
            neg_sel = np.random.choice(non_expressed_idx, size=self.n_neg, replace=True)
        neg_pe = pe_map[neg_sel]

        target_genes = torch.cat([pos_pe, neg_pe])
        target_labels = torch.cat([
            torch.ones(self.n_pos),
            torch.zeros(self.n_neg),
        ])

        result = {
            "batch_sentences": batch_sentences,
            "mask": base_mask,
            "target_genes": target_genes,
            "target_labels": target_labels,
            "cid": torch.tensor(idx, dtype=torch.long),
            "seq_len": seq_len,
        }
        for key, arr in self._label_arrays.items():
            result[key] = torch.tensor(arr[idx], dtype=torch.long)
        return result


class UCETrainingCollator:
    """Custom collator for UCE training — dict format with dynamic padding."""

    def __init__(self, pad_length, label_keys=None):
        self.pad_length = pad_length
        self._label_keys = label_keys or []

    def __call__(self, batch):
        max_len = max(b["seq_len"] for b in batch)
        result = {
            "batch_sentences": torch.stack(
                [b["batch_sentences"].squeeze(0) for b in batch]
            )[:, :max_len],
            "mask": torch.stack(
                [b["mask"].squeeze(0) for b in batch]
            )[:, :max_len],
            "target_genes": torch.stack([b["target_genes"] for b in batch]),
            "target_labels": torch.stack([b["target_labels"] for b in batch]),
            "cid": torch.tensor(
                [b["cid"] for b in batch], dtype=torch.long
            ),
        }
        for key in self._label_keys:
            if key in batch[0]:
                result[key] = torch.stack([b[key] for b in batch])
        return result
