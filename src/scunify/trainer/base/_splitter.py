"""Train / Valid / Test splitter for LoRA training.

Splits adata by column_key (donor/batch-level) or random (cell-level fallback).
Designed to avoid data leakage: all cells sharing the same group key
stay in the same split.

Also supports k-fold cross-validation via ``method="kfold"``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DataSplitter:
    """Split AnnData into train / valid / test subsets.

    Config structure (``training.split``)::

        split:
          method: "kfold"          # "column_key" | "random" | "kfold"
          test_ratio: 0.2          # shared
          seed: 42                 # shared

          column_key:              # method-specific
            key: null
            train_ratio: 0.8
            valid_ratio: 0.1

          kfold:                   # method-specific
            n_folds: 5
            stratify_key: null
            group_key: null
    """

    def __init__(self, split_cfg: dict[str, Any]):
        self.method = split_cfg.get("method", "column_key")
        self.test_ratio = float(split_cfg.get("test_ratio", 0.2))
        self.seed = int(split_cfg.get("seed", 42))

        # column_key / random params
        ck_cfg = split_cfg.get("column_key", {}) or {}
        self.column_key = ck_cfg.get("key")  # str | list[str] | None
        self.train_ratio = float(ck_cfg.get("train_ratio", 0.8))
        self.valid_ratio = float(ck_cfg.get("valid_ratio", 0.1))

        # kfold params
        kf_cfg = split_cfg.get("kfold", {}) or {}
        self.n_folds = int(kf_cfg.get("n_folds", 5))
        self.stratify_key = kf_cfg.get("stratify_key")
        self.group_key = kf_cfg.get("group_key")

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def split(self, adata):
        """Split *adata* and return ``(train_adata, valid_adata, test_adata)``.

        Also writes ``adata.obs["_scunify_split"]`` for provenance.
        """
        if self.method == "kfold":
            raise ValueError(
                "For kfold, use split_kfold() instead of split()."
            )

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

    def split_kfold(self, adata):
        """Split *adata* into test set + k folds for cross-validation.

        Returns
        -------
        test_adata : AnnData
            Held-out test set (``test_ratio`` of data).
        folds : list[tuple[AnnData, AnnData]]
            List of ``(train_adata, valid_adata)`` per fold.
        """
        from sklearn.model_selection import StratifiedGroupKFold

        n = len(adata)
        adata.obs["_scunify_orig_idx"] = np.arange(n)

        # --- Build group and stratify arrays ---
        group_key = self.group_key
        stratify_key = self.stratify_key

        if group_key is not None and group_key not in adata.obs.columns:
            logger.warning("group_key %r not found, ignoring.", group_key)
            group_key = None
        if stratify_key is not None and stratify_key not in adata.obs.columns:
            logger.warning("stratify_key %r not found, ignoring.", stratify_key)
            stratify_key = None

        # --- Hold out test set (group-aware if group_key exists) ---
        rng = np.random.RandomState(self.seed)
        indices = np.arange(n)

        if group_key is not None:
            groups_all = adata.obs[group_key].values
            unique_groups = sorted(set(groups_all), key=str)
            rng.shuffle(unique_groups)
            n_test_groups = max(1, int(round(len(unique_groups) * self.test_ratio)))
            test_groups = set(unique_groups[:n_test_groups])
            test_mask = np.array([g in test_groups for g in groups_all])
        else:
            perm = rng.permutation(n)
            n_test = max(1, int(round(n * self.test_ratio)))
            test_mask = np.zeros(n, dtype=bool)
            test_mask[perm[:n_test]] = True

        test_idx = indices[test_mask]
        trainval_idx = indices[~test_mask]
        test_adata = adata[test_idx].copy()

        logger.info(
            "KFold: held out %d test cells (%.1f%%)",
            len(test_adata), 100.0 * len(test_adata) / n,
        )

        # --- StratifiedGroupKFold on trainval portion ---
        trainval_adata = adata[trainval_idx]

        if group_key is not None:
            groups = trainval_adata.obs[group_key].values.astype(str)
        else:
            # No grouping: each cell is its own group
            groups = np.arange(len(trainval_adata)).astype(str)

        if stratify_key is not None:
            stratify = trainval_adata.obs[stratify_key].values.astype(str)
        else:
            # No stratification: single stratum
            stratify = np.zeros(len(trainval_adata), dtype=int)

        sgkf = StratifiedGroupKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )

        folds = []
        dummy_X = np.zeros((len(trainval_adata), 1))

        for fold_i, (train_rel, valid_rel) in enumerate(
            sgkf.split(dummy_X, stratify, groups)
        ):
            train_abs = trainval_idx[train_rel]
            valid_abs = trainval_idx[valid_rel]

            fold_train = adata[train_abs].copy()
            fold_valid = adata[valid_abs].copy()
            folds.append((fold_train, fold_valid))

            logger.info(
                "  Fold %d: train=%d, valid=%d",
                fold_i, len(fold_train), len(fold_valid),
            )

        return test_adata, folds

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
