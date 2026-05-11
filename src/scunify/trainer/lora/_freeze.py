"""Layer-selective freezing — single rule shared by Full FT, probe, and LoRA.

The freeze policy is driven by **one** discriminator:

    ``is_backbone_param(name) -> bool``    # provided by trainer/mixin

Rule
----
* If the parameter belongs to the backbone, apply the freeze policy
  (``mode`` + ``freeze.layer_strategy``).
* If it does *not* belong to the backbone (= task head, GEARS head,
  classifier/pooler, …), it stays trainable.

This eliminates the previous ad-hoc head-matching code (``unfreeze_head``,
``head_attr``, ``integrated_head_keywords``, ``post_lora_unfreeze``).

yaml schema::

    training:
      mode: full | probe | lora        # 3-way spectrum
      freeze:
        layer_strategy: last           # first | last | all | none | indices
        layer_ratio: 2                 # int (count) or float (fraction)
        # layer_indices: [10, 11]      # explicit, overrides strategy/ratio

How modes map onto the rule:

* ``full``  + ``layer_strategy: all``   → no-op (every backbone param trainable)
* ``full``  + ``layer_strategy: last/first/indices`` → backbone except selected layers frozen
* ``full``  + ``layer_strategy: none``  → backbone fully frozen (alias of ``probe``)
* ``probe``                             → backbone fully frozen
* ``lora``  + ``layer_strategy: ...``   → backbone fully frozen + LoRA on selected layers
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import torch.nn as nn

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Resolution
# ------------------------------------------------------------------ #

def resolve_layer_indices(
    n_layers: int, freeze_cfg: dict[str, Any]
) -> list[int] | None:
    """Resolve ``freeze`` block into a list of trainable backbone layer indices.

    Returns
    -------
    list[int] | None
        - ``None`` when ``layer_strategy`` is ``"all"`` (every backbone
          layer trainable; LoRA injects everywhere).
        - empty list ``[]`` when ``layer_strategy`` is ``"none"`` (whole
          backbone frozen; head-only training — ``probe`` semantics).
        - explicit indices otherwise.
    """
    if not freeze_cfg:
        return None

    explicit = freeze_cfg.get("layer_indices")
    if explicit is not None:
        return [int(i) for i in explicit]

    strategy = str(freeze_cfg.get("layer_strategy", "all")).lower()
    if strategy == "all":
        return None
    if strategy == "none":
        return []

    ratio = freeze_cfg.get("layer_ratio")
    if ratio is None:
        raise ValueError(
            f"freeze.layer_strategy={strategy!r} requires layer_ratio "
            f"(int count or float fraction)."
        )
    ratio = float(ratio)
    if ratio >= 1.0:
        n_selected = max(1, min(int(ratio), n_layers))
    else:
        n_selected = max(1, int(n_layers * ratio))

    if strategy == "first":
        return list(range(n_selected))
    if strategy == "last":
        return list(range(n_layers - n_selected, n_layers))
    raise ValueError(
        f"Unknown freeze.layer_strategy={strategy!r}. "
        f"Expected: first | last | all | none | indices."
    )


# ------------------------------------------------------------------ #
#  Trainable-summary + layer-index match
# ------------------------------------------------------------------ #

def _trainable_summary(model: nn.Module) -> tuple[int, int]:
    """Return ``(trainable_params, total_params)`` over ``model``."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def _is_in_layer(name: str, layer_indices: list[int], layers_pattern: str) -> bool:
    """``True`` if ``name`` is a parameter inside one of the selected layers.

    Matches the standard ``...{pattern}.{idx}.{rest}`` naming used by HF /
    torch transformer encoders. ``layers_pattern`` examples: ``layer``,
    ``layers``, ``transformer_encoder``.
    """
    needle = f".{layers_pattern}."
    if needle not in name:
        return False
    after = name.split(needle, 1)[1]
    head = after.split(".", 1)[0]
    try:
        return int(head) in layer_indices
    except ValueError:
        return False


# ------------------------------------------------------------------ #
#  Full-FT (and partial-FT): backbone + selected layers
# ------------------------------------------------------------------ #

def apply_full_ft_freeze(
    wrapper: nn.Module,
    *,
    is_backbone_param: Callable[[str], bool],
    layer_indices: list[int] | None,
    layers_pattern: str,
) -> None:
    """Apply layer-selective freeze for Full FT mode.

    * ``layer_indices is None`` (= ``layer_strategy: all``): no-op — every
      backbone parameter stays trainable (default Full FT).
    * ``layer_indices == []`` (``none``): freeze the entire backbone.
    * Otherwise: freeze backbone except parameters that live inside the
      selected layer indices.

    Non-backbone params (head, GEARS heads, integrated classifier/pooler)
    are *always* left as-is — they are built trainable and the rule never
    freezes them. This is the single difference from a plain ``for p in
    backbone: p.requires_grad=False`` and is what makes per-task head
    handling automatic.
    """
    if layer_indices is None:
        return  # all backbone trainable, head trainable — no-op

    n_back_frozen = 0
    n_back_trainable = 0
    for name, p in wrapper.named_parameters():
        if not is_backbone_param(name):
            continue   # head / non-backbone — leave trainable
        if layer_indices and _is_in_layer(name, layer_indices, layers_pattern):
            p.requires_grad = True
            n_back_trainable += 1
        else:
            p.requires_grad = False
            n_back_frozen += 1

    train, total = _trainable_summary(wrapper)
    pct = (100.0 * train / total) if total else 0.0
    logger.info(
        "[freeze] Full FT: backbone frozen=%d, trainable-by-layer=%d "
        "(layers=%s, pattern=%s) → %d/%d total trainable (%.2f%%)",
        n_back_frozen, n_back_trainable, layer_indices, layers_pattern,
        train, total, pct,
    )


# ------------------------------------------------------------------ #
#  Probe: backbone fully frozen, head only training
# ------------------------------------------------------------------ #

def apply_probe_freeze(
    wrapper: nn.Module,
    *,
    is_backbone_param: Callable[[str], bool],
) -> None:
    """Freeze the entire backbone, leave head/non-backbone trainable.

    Equivalent to ``apply_full_ft_freeze`` with ``layer_indices == []`` —
    exposed as a separate entry point so ``mode: probe`` reads naturally.
    """
    n_frozen = 0
    for name, p in wrapper.named_parameters():
        if is_backbone_param(name):
            p.requires_grad = False
            n_frozen += 1

    train, total = _trainable_summary(wrapper)
    pct = (100.0 * train / total) if total else 0.0
    logger.info(
        "[freeze] Probe: backbone frozen=%d → %d/%d total trainable (%.2f%%)",
        n_frozen, train, total, pct,
    )
