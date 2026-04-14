"""BaseMixin — base class for all task Mixins.

Provides shared utilities:
- ``_infer_emb_dim()``: infer cell embedding dimension from architecture config
- ``_unwrap()``: access BaseTrainer's DDP unwrap
- ``get_task_output()``: override in subclass to define task-specific outputs

Output format for ``get_task_output()``::

    {
        "key_name": {
            "data": tensor,              # (B, ...) per-cell output
            "storage": "obsm|obs|uns",   # where to store in adata
        },
        ...
    }

Storage types:
- ``"obsm"``: adata.obsm[f"X_{key}"] — 2D array (B, D), e.g. logits, embeddings
- ``"obs"``:  adata.obs[key]          — 1D array (B,), e.g. predictions, labels
- ``"uns"``:  adata.uns[key]          — arbitrary, e.g. metadata dict
"""

import torch.nn as nn


class BaseMixin:
    """Base class for all task Mixins."""

    def _infer_emb_dim(self) -> int:
        from ._utils import infer_emb_dim
        return infer_emb_dim(self.cfg)

    def get_task_output(self, model: nn.Module, batch: dict) -> dict:
        """Task-specific output for saving. Override in subclass.

        Returns empty dict by default (no task-specific output).
        """
        return {}
