"""Train / Valid / Test splitter for LoRA training.

Splits adata by column_key (donor/batch-level) or random (cell-level fallback).
Designed to avoid data leakage: all cells sharing the same group key
stay in the same split.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DataSplitter:
    """Split AnnData into train / valid / test subsets.

    Parameters (from ``training.split`` config)
    --------------------------------------------
    method : str
        ``"column_key"`` (default) or ``"random"``.
    column_key : str | list[str] | None
        obs column(s) for group-level split.
        - str: single column (e.g. ``"donor"``).
        - list[str]: composite key (e.g. ``["donor", "tissue"]``).
        - None: falls back to cell-level random split.
    train_ratio : float
        Fraction of data for training+validation (default 0.8).
    valid_ratio : float
        Fraction of *training set* held out for validation (default 0.1).
    test_ratio : float
        Fraction of data for test (default 0.2).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, split_cfg: dict[str, Any]):
        self.method = split_cfg.get("method", "column_key")
        self.column_key = split_cfg.get("column_key")  # str | list[str] | None
        self.train_ratio = float(split_cfg.get("train_ratio", 0.8))
        self.valid_ratio = float(split_cfg.get("valid_ratio", 0.1))
        self.test_ratio = float(split_cfg.get("test_ratio", 0.2))
        self.seed = int(split_cfg.get("seed", 42))

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def split(self, adata):
        """Split *adata* and return ``(train_adata, valid_adata, test_adata)``.

        Also writes ``adata.obs["_scunify_split"]`` for provenance.
        """
        if self.column_key is not None:
            keys = (
                [self.column_key]
                if isinstance(self.column_key, str)
                else list(self.column_key)
            )
            # Validate columns exist
            missing = [k for k in keys if k not in adata.obs.columns]
            if missing:
                logger.warning(
                    "column_key %s not found in adata.obs. "
                    "Falling back to random split.",
                    missing,
                )
                return self._split_random(adata)
            return self._split_by_column(adata, keys)

        return self._split_random(adata)

    # ------------------------------------------------------------------ #
    #  Group-level split
    # ------------------------------------------------------------------ #
    def _split_by_column(self, adata, keys: list[str]):
        """Split by unique values of *keys* in ``adata.obs``."""
        if len(keys) == 1:
            groups = adata.obs[keys[0]].values
        else:
            # Composite key → tuple per cell
            groups = (
                adata.obs[keys]
                .apply(tuple, axis=1)
                .values
            )

        unique_groups = sorted(set(groups), key=str)
        rng = np.random.RandomState(self.seed)
        rng.shuffle(unique_groups)

        n_total = len(unique_groups)
        n_test = max(1, int(round(n_total * self.test_ratio)))

        test_groups = set(unique_groups[:n_test])
        trainval_groups = unique_groups[n_test:]

        # Split trainval → train + valid
        valid_frac = self.valid_ratio / self.train_ratio  # fraction of trainval
        n_valid = max(1, int(round(len(trainval_groups) * valid_frac)))
        valid_groups = set(trainval_groups[:n_valid])
        train_groups = set(trainval_groups[n_valid:])

        # Assign labels
        labels = np.array(["train"] * len(adata))
        for i, g in enumerate(groups):
            if g in test_groups:
                labels[i] = "test"
            elif g in valid_groups:
                labels[i] = "valid"

        adata.obs["_scunify_split"] = labels
        adata.obs["_scunify_orig_idx"] = np.arange(len(adata))

        train_adata = adata[labels == "train"].copy()
        valid_adata = adata[labels == "valid"].copy()
        test_adata = adata[labels == "test"].copy()

        logger.info(
            "Split by %s: train=%d, valid=%d, test=%d "
            "(groups: train=%d, valid=%d, test=%d)",
            keys,
            len(train_adata),
            len(valid_adata),
            len(test_adata),
            len(train_groups),
            len(valid_groups),
            len(test_groups),
        )

        return train_adata, valid_adata, test_adata

    # ------------------------------------------------------------------ #
    #  Cell-level random split (fallback)
    # ------------------------------------------------------------------ #
    def _split_random(self, adata):
        """Random cell-level split."""
        n = len(adata)
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(n)

        n_test = max(1, int(round(n * self.test_ratio)))
        n_trainval = n - n_test
        valid_frac = self.valid_ratio / self.train_ratio
        n_valid = max(1, int(round(n_trainval * valid_frac)))

        test_idx = indices[:n_test]
        valid_idx = indices[n_test : n_test + n_valid]
        train_idx = indices[n_test + n_valid :]

        labels = np.array(["train"] * n)
        labels[test_idx] = "test"
        labels[valid_idx] = "valid"
        adata.obs["_scunify_split"] = labels
        adata.obs["_scunify_orig_idx"] = np.arange(n)

        train_adata = adata[train_idx].copy()
        valid_adata = adata[valid_idx].copy()
        test_adata = adata[test_idx].copy()

        logger.info(
            "Random split: train=%d, valid=%d, test=%d",
            len(train_adata),
            len(valid_adata),
            len(test_adata),
        )

        return train_adata, valid_adata, test_adata
