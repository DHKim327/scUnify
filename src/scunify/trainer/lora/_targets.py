"""Model-specific LoRA target module name mappings — HF PEFT unified.

Maps abstract target names (``"query"``, ``"value"``, ``"ffn"``) to
HF PEFT ``target_modules`` strings for each foundation model.

Notes:
    - Geneformer: separate Q/K/V Linears → native HF PEFT.
    - Others: fused QKV → unfused to q_proj/k_proj/v_proj before PEFT.
    - PEFT matches by module name suffix (e.g. ``"q_proj"``).
"""

from __future__ import annotations

# ------------------------------------------------------------------ #
#  PEFT target module mappings (all models)
# ------------------------------------------------------------------ #

# Geneformer: native HF structure (BertForMaskedLM)
_GENEFORMER_TARGETS = {
    "query": ["query"],
    "key": ["key"],
    "value": ["value"],
    "attn_out": ["attention.output.dense"],
    "ffn": ["intermediate.dense", "output.dense"],
    "ffn_up": ["intermediate.dense"],
    "ffn_down": ["output.dense"],
}

# Unfused fused-QKV models: after UnfusedMultiheadAttention replacement
# Applies to: Nicheformer, scFoundation, scGPT, UCE
_UNFUSED_TARGETS = {
    "query": ["q_proj"],
    "key": ["k_proj"],
    "value": ["v_proj"],
    "attn_out": ["out_proj"],
    "ffn": ["linear1", "linear2"],
    "ffn_up": ["linear1"],
    "ffn_down": ["linear2"],
}

_PEFT_TARGET_MAP: dict[str, dict[str, list[str]]] = {
    "geneformer": _GENEFORMER_TARGETS,
    "nicheformer": _UNFUSED_TARGETS,
    "scfoundation": _UNFUSED_TARGETS,
    "scgpt": _UNFUSED_TARGETS,
    "uce": _UNFUSED_TARGETS,
}


def resolve_peft_targets(model_name: str, targets: list[str]) -> list[str]:
    """Convert abstract target names to HF PEFT ``target_modules`` list.

    Args:
        model_name: Model identifier (e.g. ``"geneformer"``, ``"uce"``).
        targets: List of abstract names, e.g. ``["query", "value"]``.

    Returns:
        Deduplicated list of HF PEFT target module name suffixes.
    """
    name = model_name.lower()
    target_map = _PEFT_TARGET_MAP.get(name)
    if target_map is None:
        raise ValueError(
            f"Unknown model {model_name!r}. "
            f"Available: {sorted(_PEFT_TARGET_MAP.keys())}"
        )

    modules: list[str] = []
    for t in targets:
        t_lower = t.lower()
        if t_lower not in target_map:
            raise ValueError(
                f"Unknown target {t!r} for {model_name}. "
                f"Available: {sorted(target_map.keys())}"
            )
        modules.extend(target_map[t_lower])

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for m in modules:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


# ------------------------------------------------------------------ #
#  layers_pattern for PEFT layer selection
# ------------------------------------------------------------------ #
LAYERS_PATTERN: dict[str, str] = {
    "geneformer": "layer",                # bert.encoder.layer.{i}
    "nicheformer": "layers",              # encoder.layers.{i}
    "scfoundation": "transformer_encoder",  # encoder.transformer_encoder.{i}
    "scgpt": "layers",                    # encoder.layers.{i}
    "uce": "layers",                      # transformer_encoder.layers.{i}
}
