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

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "cid": base["cid"],
        }

    @staticmethod
    def collator(batch):
        """Stack fixed-length sequences (already padded to context_length)."""
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "cid": torch.tensor([b["cid"] for b in batch], dtype=torch.long),
        }
