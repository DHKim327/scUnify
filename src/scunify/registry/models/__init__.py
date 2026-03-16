# Lazy imports to avoid loading model dependencies at package import time
__all__ = ["ScFoundationWrapper", "ScGPTWrapper", "UCEWrapper", "GeneformerWrapper"]

def __getattr__(name):
    if name == "ScFoundationWrapper":
        from ._scfoundation_wrapper import ScFoundationWrapper
        return ScFoundationWrapper
    elif name == "ScGPTWrapper":
        from ._scgpt_wrapper import ScGPTWrapper
        return ScGPTWrapper
    elif name == "UCEWrapper":
        from ._uce_wrapper import UCEWrapper
        return UCEWrapper
    elif name == "GeneformerWrapper":
        from ._geneformer_wrapper import GeneformerWrapper
        return GeneformerWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
