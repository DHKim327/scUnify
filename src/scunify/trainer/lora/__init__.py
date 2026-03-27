"""LoRA injection — lazy imports to avoid heavy deps at package level."""

__all__ = [
    "inject_lora_to_model",
    "resolve_peft_targets",
    "LAYERS_PATTERN",
]

_LAZY = {
    "inject_lora_to_model": "._injection",
    "resolve_peft_targets": "._targets",
    "LAYERS_PATTERN": "._targets",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
