"""Trainer datasets — lazy imports to avoid cross-env dependency errors."""

__all__ = [
    "GeneformerTrainingDataset",
    "NicheformerTrainingDataset",
    "ScFoundationTrainingDataset",
    "ScGPTTrainingDataset",
    "UCETrainingDataset",
]

_LAZY = {
    "GeneformerTrainingDataset": "._geneformer_dataset",
    "NicheformerTrainingDataset": "._nicheformer_dataset",
    "ScFoundationTrainingDataset": "._scfoundation_dataset",
    "ScGPTTrainingDataset": "._scgpt_dataset",
    "UCETrainingDataset": "._uce_dataset",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
