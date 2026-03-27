"""Unified LoRA injection — HF PEFT for all foundation models.

For fused-QKV models (Nicheformer, scFoundation, scGPT, UCE):
    1. Unfuse selected layers' ``nn.MultiheadAttention`` → separate q/k/v_proj
    2. Apply HF PEFT targeting q_proj, k_proj, v_proj, etc.

For Geneformer (already separate Q/K/V):
    1. Apply HF PEFT directly.
"""

from __future__ import annotations

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)

# Models that need QKV unfusing before PEFT injection
_FUSED_QKV_MODELS = {"nicheformer", "scfoundation", "scgpt", "uce"}


def inject_lora_to_model(
    model: nn.Module,
    model_name: str,
    lora_cfg: dict,
) -> nn.Module:
    """Inject LoRA adapters via HF PEFT into any supported model.

    Args:
        model: The wrapper module (e.g. ``UCETrainingWrapper``).
            ``model.model`` is the actual backbone.
        model_name: One of ``"geneformer"``, ``"nicheformer"``, etc.
        lora_cfg: Dict with keys ``targets``, ``rank``, ``alpha``, ``dropout``,
            and optionally ``layer_strategy``, ``layer_ratio``.

    Returns:
        The model with LoRA injected via HF PEFT.
    """
    from peft import LoraConfig, get_peft_model

    from ._targets import LAYERS_PATTERN, resolve_peft_targets
    from ._unfused_mha import unfuse_mha_layers

    name = model_name.lower()
    rank = int(lora_cfg.get("rank", 8))
    alpha = int(lora_cfg.get("alpha", 16))
    dropout = float(lora_cfg.get("dropout", 0.1))
    targets = lora_cfg.get("targets", ["query", "value"])

    # Resolve PEFT target module names
    target_modules = resolve_peft_targets(name, targets)

    # Layer selection
    strategy = lora_cfg.get("layer_strategy", "all")
    ratio = float(lora_cfg.get("layer_ratio", 0.5))
    layers_to_transform = None
    layers_pattern = LAYERS_PATTERN.get(name)

    # --- Unfuse fused-QKV models ---
    if name in _FUSED_QKV_MODELS:
        encoder_layers = _find_encoder_layers(model.model)
        n_layers = len(encoder_layers)
        selected = _select_layers(n_layers, strategy, ratio)

        # Replace nn.MHA → UnfusedMultiheadAttention in selected layers
        unfuse_mha_layers(model.model, encoder_layers, selected)

        if strategy != "all":
            layers_to_transform = selected

        logger.info(
            f"[{model_name}] Unfused QKV in {len(selected)}/{n_layers} layers "
            f"(strategy={strategy}, ratio={ratio}), targets={targets}"
        )

    # --- Non-fused model with layer selection ---
    elif strategy != "all":
        n_layers = _count_layers(model.model)
        selected = _select_layers(n_layers, strategy, ratio)
        layers_to_transform = selected
        logger.info(
            f"[{model_name}] Layer selection: {len(selected)}/{n_layers} "
            f"(strategy={strategy}, ratio={ratio})"
        )

    # --- Freeze base weights ---
    for p in model.model.parameters():
        p.requires_grad = False

    # --- Build and apply HF PEFT ---
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        layers_to_transform=layers_to_transform,
        layers_pattern=layers_pattern,
    )

    model.model = get_peft_model(model.model, lora_config)
    model.model.print_trainable_parameters()

    return model


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _find_encoder_layers(model: nn.Module) -> list[nn.Module]:
    """Locate the list of transformer encoder layers in a model."""
    for attr_chain in [
        "encoder.layers",               # Nicheformer, scGPT
        "encoder.transformer_encoder",   # scFoundation
        "transformer_encoder.layers",    # UCE
    ]:
        obj = model
        for attr in attr_chain.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__"):
            return list(obj)

    # Fallback: collect all TransformerEncoderLayer instances
    layers = [
        m for m in model.modules()
        if isinstance(m, nn.TransformerEncoderLayer)
    ]
    if layers:
        return layers

    raise ValueError(
        "Could not find transformer encoder layers in the model. "
        "Check the model architecture."
    )


def _select_layers(n_layers: int, strategy: str, ratio: float) -> list[int]:
    """Return indices of layers to apply LoRA to."""
    n_selected = max(1, int(n_layers * ratio))

    if strategy == "first":
        return list(range(n_selected))
    elif strategy == "last":
        return list(range(n_layers - n_selected, n_layers))
    else:  # "all"
        return list(range(n_layers))


def _count_layers(model: nn.Module) -> int:
    """Count transformer layers for models that don't need unfusing."""
    # HuggingFace models (Geneformer)
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers
    # Fallback
    try:
        return len(_find_encoder_layers(model))
    except ValueError:
        return 12
