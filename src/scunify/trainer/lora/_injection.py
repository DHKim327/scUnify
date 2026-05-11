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
    freeze_cfg: dict | None,
    is_backbone_param,
) -> nn.Module:
    """Inject PEFT adapters into any supported model.

    Freeze policy is driven by the trainer's ``is_backbone_param`` hook —
    parameters classified as *not backbone* (task heads, GEARS heads,
    integrated classifier/pooler) stay trainable; backbone parameters are
    frozen and only the adapter targeting them is trainable.

    Args:
        model: The wrapper module (e.g. ``UCETrainingWrapper``);
            ``model.model`` is the actual backbone.
        model_name: One of ``"geneformer"``, ``"nicheformer"``, etc.
        lora_cfg: yaml ``training.lora`` block. ``type`` selects the HF
            PEFT config class (``LoraConfig``, ``AdaLoraConfig``,
            ``LoHaConfig``, ``LoKrConfig``, ``IA3Config``, ``OFTConfig``);
            every other key is forwarded to its constructor.
            ``target_modules`` accepts canonical names
            (``Q | K | V | O | FFN | FFN_UP | FFN_DOWN``).
        freeze_cfg: yaml ``training.freeze`` block (``layer_strategy``,
            ``layer_ratio``, ``layer_indices``) — drives where the
            adapter is attached.
        is_backbone_param: ``(name) -> bool`` callable from
            ``BaseTrainer.is_backbone_param`` that classifies each
            parameter as backbone (frozen) vs non-backbone (trainable).

    Returns:
        The model with PEFT adapters injected.
    """
    from peft import get_peft_model

    from ._freeze import resolve_layer_indices
    from ._targets import LAYERS_PATTERN, resolve_peft_targets
    from ._unfused_mha import unfuse_mha_layers

    name = model_name.lower()
    target_modules = resolve_peft_targets(name, lora_cfg["target_modules"])

    if name in _FUSED_QKV_MODELS:
        encoder_layers = _find_encoder_layers(model.model)
        n_layers = len(encoder_layers)
    else:
        n_layers = _count_layers(model.model)

    selected = resolve_layer_indices(n_layers, freeze_cfg or {})
    selected_for_log = list(range(n_layers)) if selected is None else selected
    layers_to_transform = None if selected is None else selected
    layers_pattern = None if selected is None else LAYERS_PATTERN.get(name)

    if name in _FUSED_QKV_MODELS:
        unfuse_mha_layers(model.model, encoder_layers, selected_for_log)
        logger.info(
            f"[{model_name}] Unfused QKV in {len(selected_for_log)}/{n_layers} "
            f"layers, target_modules={lora_cfg['target_modules']}"
        )
    elif selected is not None:
        logger.info(
            f"[{model_name}] Layer selection: {len(selected_for_log)}/{n_layers}"
        )

    peft_config = _build_peft_config(
        lora_cfg, target_modules, layers_to_transform, layers_pattern,
    )
    model.model = get_peft_model(model.model, peft_config)

    # Apply the single freeze rule: backbone → frozen (PEFT already did
    # most of this), non-backbone → trainable. We re-enable non-backbone
    # params that PEFT may have frozen (e.g. integrated classifier/pooler
    # in HF models, GEARS heads inside scFoundation perturbation wrapper).
    n_unfrozen = 0
    for pname, p in model.named_parameters():
        if not is_backbone_param(pname):
            if not p.requires_grad:
                n_unfrozen += 1
            p.requires_grad = True
    if n_unfrozen:
        logger.info(
            f"[{model_name}] Restored requires_grad on {n_unfrozen} "
            f"non-backbone params after PEFT injection"
        )

    model.model.print_trainable_parameters()
    return model


def _build_peft_config(lora_cfg, target_modules, layers_to_transform, layers_pattern):
    """Build a HF PEFT config — pure passthrough.

    yaml ``training.lora.type`` is the exact class name in
    ``peft`` (``LoraConfig``, ``AdaLoraConfig``, ``LoHaConfig``,
    ``LoKrConfig``, ``IA3Config``, ``OFTConfig``); every other yaml key
    is forwarded directly to the constructor — except ``target_modules``
    and the layer-selection keys, which are injected separately so this
    helper can resolve canonical names + freeze-block indices.
    """
    import peft

    cls_name = lora_cfg["type"]
    cls = getattr(peft, cls_name, None)
    if cls is None or not callable(cls):
        raise ValueError(
            f"Unknown PEFT config type {cls_name!r}. Expected a class on "
            f"``peft`` (e.g. ``LoraConfig``, ``AdaLoraConfig``, "
            f"``LoHaConfig``, ``LoKrConfig``, ``IA3Config``, ``OFTConfig``)."
        )

    _INTERNAL_KEYS = {"type", "target_modules"}
    passthrough = {k: v for k, v in lora_cfg.items() if k not in _INTERNAL_KEYS}

    # Layer-selection (``layers_to_transform`` / ``layers_pattern``) is only
    # supported by Lora-family configs; other PEFT classes raise on these
    # kwargs. Filter accordingly.
    kwargs = {"target_modules": target_modules, **passthrough}
    if cls_name in ("LoraConfig", "AdaLoraConfig"):
        kwargs.setdefault("bias", "none")
        if layers_to_transform is not None:
            kwargs["layers_to_transform"] = layers_to_transform
            kwargs["layers_pattern"] = layers_pattern

    return cls(**kwargs)


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _find_encoder_layers(model: nn.Module) -> list[nn.Module]:
    """Locate the list of transformer encoder layers in a model."""
    for attr_chain in [
        "nicheformer.encoder.layers",    # Nicheformer (HF NicheformerForMaskedLM)
        "encoder.layers",                # scGPT (TransformerModel)
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
