"""Per-domain batch samplers for backbones whose paper recipe assumes
single-domain minibatches (e.g. scGPT BC with DSBN).

Reference: ``RelatedWorks/Foundations/scGPT/scgpt/data_sampler.py`` —
``SubsetsBatchSampler`` is reproduced verbatim (single-source-of-truth) so
the framework does not need to import the upstream package.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import (
    BatchSampler,
    Sampler,
    SubsetRandomSampler,
)


class SubsetSequentialSampler(Sampler):
    """Sequentially sample indices from a fixed list (no shuffling)."""

    def __init__(self, indices: Sequence[int]):
        self.indices = indices

    def __iter__(self) -> Iterable[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class SubsetsBatchSampler(Sampler[List[int]]):
    """Per-domain batch sampler — every minibatch comes from a single subset.

    Paper-faithful copy of scGPT's ``SubsetsBatchSampler``
    (``Foundations/scGPT/scgpt/data_sampler.py:25``). Used by
    :class:`ScGPTIntegrationMixin` so that DSBN's per-domain BatchNorm
    statistics see a single ``batch_id`` per minibatch, matching the BC
    paper recipe (``per_seq_batch_sample = True``).

    Args:
        subsets: list of index sequences, one per domain.
        batch_size: minibatch size.
        intra_subset_shuffle: shuffle order within each domain.
        inter_subset_shuffle: shuffle the order in which domain-batches
            are yielded across an epoch.
        drop_last: drop a domain's final partial batch if it would be
            smaller than ``batch_size``.
    """

    def __init__(
        self,
        subsets: List[Sequence[int]],
        batch_size: int,
        intra_subset_shuffle: bool = True,
        inter_subset_shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.subsets = subsets
        self.batch_size = batch_size
        self.intra_subset_shuffle = intra_subset_shuffle
        self.inter_subset_shuffle = inter_subset_shuffle
        self.drop_last = drop_last

        if intra_subset_shuffle:
            self.subset_samplers = [SubsetRandomSampler(s) for s in subsets]
        else:
            self.subset_samplers = [SubsetSequentialSampler(s) for s in subsets]

        self.batch_samplers = [
            BatchSampler(s, batch_size, drop_last) for s in self.subset_samplers
        ]

        if inter_subset_shuffle:
            _id_to_batch_sampler = []
            for i, batch_sampler in enumerate(self.batch_samplers):
                _id_to_batch_sampler.extend([i] * len(batch_sampler))
            self._id_to_batch_sampler = np.array(_id_to_batch_sampler)
            assert len(self._id_to_batch_sampler) == len(self)

    def __iter__(self) -> Iterable[List[int]]:
        if self.inter_subset_shuffle:
            # Refresh BatchSampler iterators each epoch so SubsetRandomSampler
            # is re-shuffled (BatchSampler constructs a fresh iter on each
            # __iter__ call).
            iters = [iter(bs) for bs in self.batch_samplers]
            random_idx = torch.randperm(len(self._id_to_batch_sampler))
            for batch_sampler_id in self._id_to_batch_sampler[random_idx]:
                yield next(iters[int(batch_sampler_id)])
        else:
            for batch_sampler in self.batch_samplers:
                yield from batch_sampler

    def __len__(self) -> int:
        return sum(len(b) for b in self.batch_samplers)
