"""Model-specific LoRA target module name mappings.

Maps abstract target names (``"query"``, ``"value"``, ``"ffn"``) to actual
PyTorch module paths used by each foundation model.

Notes:
    - Geneformer: separate Q/K/V Linears → HF PEFT natively supported.
    - scGPT / scFoundation / UCE / Nicheformer: fused QKV (in_proj_weight)
      → requires custom MergedLinear (not yet implemented).
    - Module names marked with ``{}``: layer index placeholder.
    - Module names need verification via ``model.named_modules()`` for
      scGPT and Nicheformer.
"""

from __future__ import annotations

# ------------------------------------------------------------------ #
#  Full module-path map (for reference / future custom injection)
# ------------------------------------------------------------------ #
LORA_TARGET_MAP: dict[str, dict[str, str]] = {
    "geneformer": {
        "query": "bert.encoder.layer.{}.attention.self.query",
        "key": "bert.encoder.layer.{}.attention.self.key",
        "value": "bert.encoder.layer.{}.attention.self.value",
        "attn_out": "bert.encoder.layer.{}.attention.output.dense",
        "ffn_up": "bert.encoder.layer.{}.intermediate.dense",
        "ffn_down": "bert.encoder.layer.{}.output.dense",
    },
    "scgpt": {
        "qkv_fused": "encoder.layers.{}.self_attn.Wqkv",
        "attn_out": "encoder.layers.{}.self_attn.out_proj",
        "ffn_up": "encoder.layers.{}.ffn.W1",
        "ffn_down": "encoder.layers.{}.ffn.W2",
    },
    "scfoundation": {
        "qkv_fused": "encoder.layers.{}.self_attn",
        "attn_out": "encoder.layers.{}.self_attn.out_proj",
        "ffn_up": "encoder.layers.{}.linear1",
        "ffn_down": "encoder.layers.{}.linear2",
    },
    "uce": {
        "qkv_fused": "transformer_encoder.layers.{}.self_attn",
        "attn_out": "transformer_encoder.layers.{}.self_attn.out_proj",
        "ffn_up": "transformer_encoder.layers.{}.linear1",
        "ffn_down": "transformer_encoder.layers.{}.linear2",
    },
    "nicheformer": {
        "qkv_fused": "encoder.layers.{}.self_attn",
        "attn_out": "encoder.layers.{}.self_attn.out_proj",
        "ffn_up": "encoder.layers.{}.linear1",
        "ffn_down": "encoder.layers.{}.linear2",
    },
}

# ------------------------------------------------------------------ #
#  HF PEFT target resolution (Geneformer only for now)
# ------------------------------------------------------------------ #
# Maps user-friendly target names → HF PEFT target_modules strings
# HF PEFT matches by suffix, so "query" matches all layers automatically.
_HF_PEFT_MAP: dict[str, list[str]] = {
    "query": ["query"],
    "key": ["key"],
    "value": ["value"],
    "attn_out": ["attention.output.dense"],
    "ffn": ["intermediate.dense", "output.dense"],
}


def resolve_hf_peft_targets(model_name: str, targets: list[str]) -> list[str]:
    """Convert abstract target names to HF PEFT ``target_modules`` list.

    Args:
        model_name: Model identifier (currently only ``"geneformer"``).
        targets: List of abstract names, e.g. ``["query", "value", "ffn"]``.

    Returns:
        Deduplicated list of HF PEFT target module name suffixes.

    Raises:
        ValueError: If model is not supported for HF PEFT or target unknown.
    """
    if model_name.lower() != "geneformer":
        raise ValueError(
            f"HF PEFT target resolution only supports Geneformer. "
            f"Got: {model_name!r}"
        )

    modules: list[str] = []
    for t in targets:
        t_lower = t.lower()
        if t_lower not in _HF_PEFT_MAP:
            raise ValueError(
                f"Unknown target {t!r}. "
                f"Available: {list(_HF_PEFT_MAP.keys())}"
            )
        modules.extend(_HF_PEFT_MAP[t_lower])

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for m in modules:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result
