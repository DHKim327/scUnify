__all__ = [
    "NicheformerInferenceWrapper",
    "GeneformerInferenceWrapper",
    "ScGPTInferenceWrapper",
    "ScFoundationInferenceWrapper",
    "UCEInferenceWrapper",
]


def __getattr__(name):
    if name == "NicheformerInferenceWrapper":
        from ._nicheformer_wrapper import NicheformerInferenceWrapper
        return NicheformerInferenceWrapper
    elif name == "GeneformerInferenceWrapper":
        from ._geneformer_wrapper import GeneformerInferenceWrapper
        return GeneformerInferenceWrapper
    elif name == "ScGPTInferenceWrapper":
        from ._scgpt_wrapper import ScGPTInferenceWrapper
        return ScGPTInferenceWrapper
    elif name == "ScFoundationInferenceWrapper":
        from ._scfoundation_wrapper import ScFoundationInferenceWrapper
        return ScFoundationInferenceWrapper
    elif name == "UCEInferenceWrapper":
        from ._uce_wrapper import UCEInferenceWrapper
        return UCEInferenceWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
