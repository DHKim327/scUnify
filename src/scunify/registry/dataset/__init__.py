# Lazy imports - loaded only when needed
__all__ = ["ScFoundationDataset", "ScGPTDataset", "UCEDataset"]

_IMPORTED = {}


def __getattr__(name):
    """Lazy import dataset classes"""
    if name == "ScFoundationDataset":
        if name not in _IMPORTED:
            from ._scfoundation_dataset import ScFoundationDataset
            _IMPORTED[name] = ScFoundationDataset
        return _IMPORTED[name]
    elif name == "ScGPTDataset":
        if name not in _IMPORTED:
            from ._scgpt_dataset import ScGPTDataset
            _IMPORTED[name] = ScGPTDataset
        return _IMPORTED[name]
    elif name == "UCEDataset":
        if name not in _IMPORTED:
            from ._uce_dataset import UCEDataset
            _IMPORTED[name] = UCEDataset
        return _IMPORTED[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
