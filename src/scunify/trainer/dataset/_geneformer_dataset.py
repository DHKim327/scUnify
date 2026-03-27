"""Training dataset for Geneformer — inherits from registry, adds MLM masking.

Changes from inference version:
- ``__getitem__`` applies 15% MLM masking (80/10/10 rule)
- Returns ``labels`` field for CrossEntropyLoss
- ``collator`` pads ``labels`` with -100
"""

import logging

import torch

from ...registry.dataset import GeneformerDataset

logger = logging.getLogger(__name__)


class GeneformerTrainingDataset(GeneformerDataset):
    """Geneformer training dataset with MLM masking.

    Inherits ``__init__`` from :class:`GeneformerDataset` — same gene mapping,
    canonical ID collapsing, count matrix, norm factors, tokenization.

    Overrides ``__getitem__`` to apply masking and ``collator`` to pad labels.
    """

    def __init__(self, adata, config):
        super().__init__(adata, config)

        training_cfg = config.get("training", {})
        mlm_cfg = training_cfg.get("mlm", {})
        self.mask_ratio = float(mlm_cfg.get("mask_prob", 0.15))
        self.mask_token_prob = float(mlm_cfg.get("mask_token_prob", 0.8))
        self.random_token_prob = float(mlm_cfg.get("random_token_prob", 0.1))

        # Resolve <mask> token ID (parent now stores it from gene_token_dict)
        if self.mask_token_id is not None:
            self._mask_token_id = self.mask_token_id
        else:
            self._mask_token_id = self.eos_token_id + 1
            logger.warning(
                f"<mask> not found in gene_token_dict, "
                f"falling back to eos_token_id + 1 = {self._mask_token_id}"
            )
        # Vocab size for random token replacement
        self._vocab_size = len(self.gene_tokens) + 4  # +special tokens

    def __getitem__(self, idx):
        """Tokenize + apply MLM masking."""
        base = super().__getitem__(idx)
        input_ids = base["input_ids"].clone()

        # Create labels (original tokens at masked positions, -100 elsewhere)
        labels = torch.full_like(input_ids, -100)

        # Apply MLM masking to gene tokens (exclude CLS at 0, EOS at end)
        gene_start = 1
        gene_end = base["length"] - 1  # exclusive (EOS position)
        n_genes = gene_end - gene_start

        if n_genes > 0:
            n_mask = max(1, int(n_genes * self.mask_ratio))
            mask_indices = torch.randperm(n_genes)[:n_mask] + gene_start

            # Set labels at masked positions to original token IDs
            labels[mask_indices] = input_ids[mask_indices].clone()

            # Apply 80/10/10 rule
            for pos in mask_indices:
                prob = torch.rand(1).item()
                if prob < self.mask_token_prob:
                    input_ids[pos] = self._mask_token_id
                elif prob < self.mask_token_prob + self.random_token_prob:
                    input_ids[pos] = torch.randint(
                        0, self._vocab_size, (1,)
                    ).item()
                # else: keep original (10%)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "length": base["length"],
            "cid": base["cid"],
        }

    @staticmethod
    def collator(batch):
        """Dynamic padding with labels (padded positions = -100)."""
        max_len = max(b["length"] for b in batch)
        bsz = len(batch)

        input_ids = torch.zeros(bsz, max_len, dtype=torch.long)
        attn_mask = torch.zeros(bsz, max_len, dtype=torch.long)
        labels = torch.full((bsz, max_len), -100, dtype=torch.long)
        cids = []

        for i, b in enumerate(batch):
            seq_len = b["length"]
            input_ids[i, :seq_len] = b["input_ids"]
            # Length-based mask (CLS token ID = 0 = pad, so can't use != 0)
            attn_mask[i, :seq_len] = 1
            labels[i, :seq_len] = b["labels"]
            cids.append(b["cid"])

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
            "cid": torch.tensor(cids, dtype=torch.long),
        }
