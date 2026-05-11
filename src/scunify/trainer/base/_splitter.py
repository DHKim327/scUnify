"""Train / Valid / (Test) splitter for scUnify training.

Reads user-defined split assignments from ``adata.obs`` columns. The split
information must be provided by the caller — scUnify does not generate
random / stratified / paper-recipe splits programmatically.

Config structure (``training.split``)::

    split:
      fold_keys:                        # obs column(s) containing split labels
        - "fold_0"                      #   each column: "train" | "valid" | "test"
        - "fold_1"
        - "fold_2"

Each fold column may contain any subset of {"train", "valid", "test"}; the
only requirement is that "train" appears at least once. The four supported
scenarios all map to obs label combinations:

==================  =================  ==============================
Scenario            obs label set      Behaviour
==================  =================  ==============================
train only          {"train"}          No validation, no test held out
train + test        {"train", "test"}  test cells held out from training
                                       (still appear in extracted h5ad)
train + valid       {"train", "valid"} early-stopping, best-ckpt by
                                       val_loss
train + valid+test  {"train",          early-stopping + held-out test
                    "valid", "test"}
==================  =================  ==============================

Single split (no cross-validation)::

    split:
      fold_keys: ["split"]             # single obs column

K-fold cross-validation::

    split:
      fold_keys: ["fold_0", "fold_1", "fold_2"]
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_VALID_LABELS = {"train", "valid", "test"}


class DataSplitter:
    """Split AnnData based on user-provided obs columns."""

    def __init__(self, split_cfg: dict[str, Any]):
        self.fold_keys: list[str] = split_cfg.get("fold_keys") or []
        if isinstance(self.fold_keys, str):
            self.fold_keys = [self.fold_keys]

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    @property
    def n_folds(self) -> int:
        return len(self.fold_keys)

    def split(self, adata):
        """Single split using the first fold_key.

        Returns ``(train_adata, valid_adata, test_adata)``. The valid/test
        slices are empty (zero rows) when the corresponding labels are
        absent from the obs column.
        """
        if not self.fold_keys:
            raise ValueError(
                "split.fold_keys is required. "
                "Set obs columns with 'train'/'valid'/'test' labels."
            )
        return self._split_by_column(adata, self.fold_keys[0])

    def split_kfold(self, adata):
        """Multi-fold split using all fold_keys.

        Each fold column produces its own ``(train, valid)`` partition.
        The shared test set comes from the **first** fold column's "test"
        labels — subsequent folds' "test" labels are ignored. If the first
        fold has no "test" labels, the shared test set is empty.

        Returns ``(test_adata, [(train, valid), ...])``.
        """
        if not self.fold_keys:
            raise ValueError(
                "split.fold_keys is required. "
                "Set obs columns with 'train'/'valid' labels."
            )

        n = len(adata)
        adata.obs["_scunify_orig_idx"] = np.arange(n)

        # Use first fold to determine test set (shared across folds)
        first_train, first_valid, test_adata = self._split_by_column(
            adata, self.fold_keys[0]
        )

        folds = [(first_train, first_valid)]

        for fold_key in self.fold_keys[1:]:
            train_a, valid_a, _ = self._split_by_column(adata, fold_key)
            folds.append((train_a, valid_a))

        for i, (tr, va) in enumerate(folds):
            logger.info(
                "  Fold %d (%s): train=%d, valid=%d",
                i, self.fold_keys[i], len(tr), len(va),
            )

        if len(test_adata) > 0:
            logger.info("  Test: %d cells", len(test_adata))
        else:
            logger.info("  No test set (all cells used for train/valid).")

        return test_adata, folds

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _split_by_column(self, adata, col_name: str):
        """Split adata by a single obs column.

        Returns ``(train_adata, valid_adata, test_adata)``.
        """
        if col_name not in adata.obs.columns:
            raise KeyError(
                f"Fold column '{col_name}' not found in adata.obs. "
                f"Available: {list(adata.obs.columns)}"
            )

        n = len(adata)
        if "_scunify_orig_idx" not in adata.obs.columns:
            adata.obs["_scunify_orig_idx"] = np.arange(n)

        labels = adata.obs[col_name].astype(str).values
        unique = set(labels)

        # Validate: only 'train' is required
        if "train" not in unique:
            raise ValueError(
                f"Column '{col_name}' must contain 'train'. Found: {unique}"
            )
        unknown = unique - _VALID_LABELS
        if unknown:
            raise ValueError(
                f"Column '{col_name}' contains unknown labels: {unknown}. "
                f"Allowed: {_VALID_LABELS}"
            )

        train_mask = labels == "train"
        valid_mask = labels == "valid"
        test_mask = labels == "test"

        train_adata = adata[train_mask].copy()
        valid_adata = adata[valid_mask].copy() if valid_mask.any() else adata[:0].copy()
        test_adata = adata[test_mask].copy() if test_mask.any() else adata[:0].copy()

        logger.info(
            "Split '%s': train=%d, valid=%d, test=%d",
            col_name, len(train_adata), len(valid_adata), len(test_adata),
        )

        return train_adata, valid_adata, test_adata
