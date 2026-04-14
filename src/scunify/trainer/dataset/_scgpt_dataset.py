"""Training dataset for scGPT — inherits from registry, enables GEP masking.

Changes from inference version:
- DataCollator with ``do_mlm=True`` — masks expression values (→ -1)
- Binning and padding handled by the same collator
- ``batch["expr"]`` = original binned values (target for MSE loss)
- ``batch["masked_expr"]`` = masked input (with -1 at masked positions)
"""

import logging

import torch

from ...registry.dataset._scgpt_dataset import DataCollator, ScGPTDataset

logger = logging.getLogger(__name__)


class ScGPTTrainingDataset(ScGPTDataset):
    """scGPT training dataset with GEP masking.

    Inherits ``__init__`` from :class:`ScGPTDataset` — same preprocessing,
    tokenization, gene vocabulary.

    Overrides the collator to enable MLM-style masking on binned
    expression values (``mask_value=-1``).
    """

    def __init__(self, adata, config):
        super().__init__(adata, config)
        training_cfg = config.get("training", {})
        gep_cfg = training_cfg.get("gep", {})
        mask_prob = float(gep_cfg.get("mask_prob", 0.15))

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

        # Override collator: enable masking for training
        base_collator = DataCollator(
            do_padding=True,
            pad_token_id=self.pad_token_id,
            pad_value=self.pad_value,
            do_mlm=True,
            mlm_probability=mask_prob,
            max_length=1200,
            sampling=True,
            keep_first_n_tokens=1,
        )
        self._base_collator = base_collator
        self.collator = self._collator_with_labels

    def __getitem__(self, idx):
        result = super().__getitem__(idx)
        for key, arr in self._label_arrays.items():
            result[key] = torch.tensor(arr[idx], dtype=torch.long)
        return result

    def _collator_with_labels(self, batch):
        """Wrap base collator to include label keys."""
        result = self._base_collator(batch)
        for key in self._label_arrays:
            if key in batch[0]:
                result[key] = torch.stack([b[key] for b in batch])
        return result
