"""Training model wrappers — lazy imports to avoid cross-env dependency errors."""

__all__ = [
    "GeneformerTrainingWrapper",
    "NicheformerTrainingWrapper",
    "ScFoundationTrainingWrapper",
    "ScGPTTrainingWrapper",
    "UCETrainingWrapper",
]

_LAZY = {
    "GeneformerTrainingWrapper": "._geneformer_wrapper",
    "NicheformerTrainingWrapper": "._nicheformer_wrapper",
    "ScFoundationTrainingWrapper": "._scfoundation_wrapper",
    "ScGPTTrainingWrapper": "._scgpt_wrapper",
    "UCETrainingWrapper": "._uce_wrapper",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
