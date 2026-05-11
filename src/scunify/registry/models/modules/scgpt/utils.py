"""scGPT utility helpers (vendored from
``RelatedWorks/Foundations/scGPT/scgpt/utils/util.py``). Currently exposes
only the helper functions required by the perturbation generation_model and
mixin — the rest stays in scgpt source for now.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import torch


def map_raw_id_to_vocab_id(
    raw_ids: Union[np.ndarray, torch.Tensor],
    gene_ids: np.ndarray,
) -> Union[np.ndarray, torch.Tensor]:
    """Map raw indices (positions in raw gene name list) to vocab indices.

    Byte-level identical to ``RelatedWorks/Foundations/scGPT/scgpt/utils/util.py:273``.
    Used by ``TransformerGenerator.pred_perturb`` and by the perturbation mixin
    train loop to translate per-batch gene indices into the model's vocab space.
    """
    if isinstance(raw_ids, torch.Tensor):
        device = raw_ids.device
        dtype = raw_ids.dtype
        return_pt = True
        raw_ids = raw_ids.cpu().numpy()
    elif isinstance(raw_ids, np.ndarray):
        return_pt = False
        dtype = raw_ids.dtype
    else:
        raise ValueError("raw_ids must be either torch.Tensor or np.ndarray.")

    if raw_ids.ndim != 1:
        raise ValueError(f"raw_ids must be 1d, got {raw_ids.ndim}d.")

    if gene_ids.ndim != 1:
        raise ValueError(f"gene_ids must be 1d, got {gene_ids.ndim}d.")

    mapped_ids: np.ndarray = gene_ids[raw_ids]
    assert mapped_ids.shape == raw_ids.shape
    if return_pt:
        return torch.from_numpy(mapped_ids).type(dtype).to(device)
    return mapped_ids.astype(dtype)
