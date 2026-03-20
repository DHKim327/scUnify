"""Unified LoRA injection for all foundation models.

Dispatches to HF PEFT (Geneformer) or custom MergedLinear (others).
"""

from __future__ import annotations

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


def inject_lora_to_model(
    model: nn.Module,
    model_name: str,
    lora_cfg: dict,
) -> nn.Module:
    """Inject LoRA adapters into a foundation model.

    Args:
        model: The wrapper module (e.g. ``GeneformerTrainingWrapper``).
            ``model.model`` should be the actual backbone
            (e.g. ``BertForMaskedLM``).
        model_name: One of ``"geneformer"``, ``"scgpt"``, etc.
        lora_cfg: Dict with keys ``targets``, ``rank``, ``alpha``, ``dropout``.

    Returns:
        The model with LoRA injected. Frozen params have
        ``requires_grad=False``; only LoRA params are trainable.
    """
    name = model_name.lower()
    targets = lora_cfg.get("targets", ["query", "value"])
    rank = int(lora_cfg.get("rank", 8))
    alpha = int(lora_cfg.get("alpha", 16))
    dropout = float(lora_cfg.get("dropout", 0.1))

    if name == "geneformer":
        model = _inject_hf_peft(model, targets, rank, alpha, dropout)
    else:
        raise NotImplementedError(
            f"LoRA injection for {model_name!r} is not yet implemented. "
            f"Custom MergedLinear support is planned for Phase 2+."
        )

    _log_trainable_params(model)
    return model


def _inject_hf_peft(
    model: nn.Module,
    targets: list[str],
    rank: int,
    alpha: int,
    dropout: float,
) -> nn.Module:
    """Inject LoRA via HF PEFT (Geneformer with separate Q/K/V)."""
    from peft import LoraConfig, TaskType, get_peft_model

    from ._targets import resolve_hf_peft_targets

    target_modules = resolve_hf_peft_targets("geneformer", targets)

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )

    # model.model = BertForMaskedLM
    model.model = get_peft_model(model.model, lora_config)
    model.model.print_trainable_parameters()
    return model


def _log_trainable_params(model: nn.Module) -> None:
    """Log trainable vs total parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total > 0 else 0.0
    logger.info(
        f"Trainable: {trainable:,} / {total:,} ({pct:.2f}%)"
    )
