"""Training dataset for scFoundation — downstream tasks only.

Inherits from ScFoundationDataset (fixed-length 19266 vector per cell).
Adds label_keys passthrough for downstream tasks (classification, etc.).
Supports batch_size > 1.

Note: Pretraining MAE is not supported in this dataset.
For pretraining, a separate MAE-specific dataset would be needed.
"""

import logging

import torch

from ...registry.dataset._scfoundation_dataset import ScFoundationDataset

logger = logging.getLogger(__name__)


class ScFoundationTrainingDataset(ScFoundationDataset):
    """scFoundation training dataset for downstream tasks.

    Same as ScFoundationDataset (fixed-length 19266) + label passthrough.
    """

    def __init__(self, adata, config):
        super().__init__(adata, config)
        training_cfg = config.get("training", {})

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

    def __getitem__(self, idx):
        base = super().__getitem__(idx)
        for key, arr in self._label_arrays.items():
            base[key] = torch.tensor(arr[idx], dtype=torch.long)
        return base

    @staticmethod
    def collator(batch):
        """Stack fixed-length features + labels."""
        result = {
            "pretrain_gene_x": torch.stack([b["pretrain_gene_x"] for b in batch]),
            "cid": torch.tensor([b["cid"] for b in batch], dtype=torch.long),
        }
        # Passthrough any extra keys (labels, pert_idx, etc.)
        known_keys = {"pretrain_gene_x", "cid"}
        for key in batch[0]:
            if key not in known_keys:
                result[key] = torch.stack([b[key] for b in batch])
        return result
