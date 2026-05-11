"""Training dataset for Nicheformer — inherits from registry, adds MLM masking.

Changes from inference version:
- ``__getitem__`` applies 15% MLM masking (80/10/10 rule)
- Returns ``labels`` field for CrossEntropyLoss
- Context tokens (species, assay, modality) are **protected** from masking
"""

import logging

import torch

from ...registry.dataset._nicheformer_dataset import (
    AUX_TOKENS,
    NicheformerDataset,
    PAD_TOKEN,
)

logger = logging.getLogger(__name__)


class NicheformerTrainingDataset(NicheformerDataset):
    """Nicheformer training dataset with MLM masking.

    Inherits ``__init__`` from :class:`NicheformerDataset` — same gene
    alignment, sf-normalization, tokenization, context token assembly.

    Overrides ``__getitem__`` to apply masking and ``collator`` to
    include ``labels``.

    Masking rules:
    - First 3 positions (species, assay, modality) are **never masked**.
    - Gene tokens are masked with probability ``mask_prob`` (default 15%).
    - 80% → mask token (0), 10% → random gene token, 10% → keep original.
    - ``labels`` = original token at masked positions, -100 elsewhere.
    """

    N_CONTEXT_TOKENS = 3  # species, assay, modality

    def __init__(self, adata, config):
        super().__init__(adata, config)

        training_cfg = config.get("training", {})
        mlm_cfg = training_cfg.get("mlm", {})

        # Label passthrough — auto-route obs (scalar/categorical) vs
        # obsm (vector). Tasks like RegressionMixin can target both:
        #   density (obs scalar) → batch[key] is (B,) float
        #   X_niche_N (obsm sparse vector) → batch[key] is (B, D) float
        import numpy as np
        from scipy.sparse import issparse

        self._label_arrays: dict[str, "np.ndarray | sparse.spmatrix"] = {}
        self._label_sources: dict[str, str] = {}   # "obs" | "obsm"
        for key in training_cfg.get("label_keys", []):
            if key in adata.obs.columns:
                col = adata.obs[key]
                self._label_arrays[key] = (
                    col.cat.codes.values.copy()
                    if hasattr(col, "cat")
                    else col.values.copy()
                )
                self._label_sources[key] = "obs"
            elif key in adata.obsm.keys():
                x = adata.obsm[key]
                # keep sparse for memory; densify per-cell in __getitem__
                self._label_arrays[key] = x if issparse(x) else np.asarray(x)
                self._label_sources[key] = "obsm"
            else:
                raise KeyError(
                    f"label_key {key!r} not found in adata.obs or adata.obsm. "
                    f"obs cols: {list(adata.obs.columns)[:10]}..., "
                    f"obsm keys: {list(adata.obsm.keys())}"
                )
        self.mask_ratio = float(mlm_cfg.get("mask_prob", 0.15))
        self.mask_token_prob = float(mlm_cfg.get("mask_token_prob", 0.8))
        self.random_token_prob = float(mlm_cfg.get("random_token_prob", 0.1))

        # Nicheformer original convention: masked positions → 0
        # (see complete_masking in Foundations/nicheformer)
        self._mask_token_id = PAD_TOKEN  # 0

        # Valid gene token range for random replacement
        # Gene tokens = gene_index + AUX_TOKENS, range [AUX_TOKENS, n_ref + AUX_TOKENS)
        self._gene_token_lo = AUX_TOKENS
        self._gene_token_hi = self.n_ref + AUX_TOKENS  # exclusive upper bound

    def __getitem__(self, idx):
        """Tokenise + apply MLM masking."""
        base = super().__getitem__(idx)
        input_ids = base["input_ids"].clone()
        attn_mask = base["attention_mask"].clone()

        # Create labels (original tokens at masked positions, -100 elsewhere)
        labels = torch.full_like(input_ids, -100)

        # Gene token positions: after context tokens, before padding
        n_real = int(attn_mask.sum().item())
        gene_start = self.N_CONTEXT_TOKENS
        gene_end = n_real  # exclusive
        n_genes = gene_end - gene_start

        if n_genes > 0:
            n_mask = max(1, int(n_genes * self.mask_ratio))
            mask_indices = torch.randperm(n_genes)[:n_mask] + gene_start

            # Labels at masked positions = original token IDs
            labels[mask_indices] = input_ids[mask_indices].clone()

            # Apply 80/10/10 rule
            for pos in mask_indices:
                prob = torch.rand(1).item()
                if prob < self.mask_token_prob:
                    # 80% → mask token
                    input_ids[pos] = self._mask_token_id
                elif prob < self.mask_token_prob + self.random_token_prob:
                    # 10% → random gene token
                    input_ids[pos] = torch.randint(
                        self._gene_token_lo, self._gene_token_hi, (1,),
                    ).item()
                # else: 10% → keep original

        result = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "cid": base["cid"],
        }
        for key, arr in self._label_arrays.items():
            if self._label_sources[key] == "obs":
                # Categorical → long, numerical → preserve dtype (cast in mixin)
                v = arr[idx]
                if hasattr(v, "dtype") and getattr(v.dtype, "kind", None) in ("i", "u", "b"):
                    result[key] = torch.tensor(v, dtype=torch.long)
                else:
                    result[key] = torch.tensor(float(v), dtype=torch.float32)
            else:  # obsm — vector
                row = arr[idx]
                if hasattr(row, "toarray"):       # sparse row
                    row = row.toarray().ravel()
                else:
                    row = row.ravel() if hasattr(row, "ravel") else row
                result[key] = torch.tensor(row, dtype=torch.float32)
        return result

    def collator(self, batch):
        """Stack fixed-length sequences (already padded to context_length)."""
        result = {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "cid": torch.tensor([b["cid"] for b in batch], dtype=torch.long),
        }
        for key in self._label_arrays:
            if key in batch[0]:
                result[key] = torch.stack([b[key] for b in batch])
        return result
