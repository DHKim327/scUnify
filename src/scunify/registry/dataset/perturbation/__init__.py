"""Perturbation dataset wrappers — Layer 1 paper preprocessing.

Each backbone has its own per-paper recipe (cell graph format, gene-vocab
expand). The base ``BasePertData`` reads metadata from a single
``adata.uns`` (gene2go, splits, subgroup) so user-facing input is just a
single ``NORMAN.h5ad`` — no auxiliary pickles or sub-folders.

Public API:
    BasePertData                      — base class
    ScGPTPerturbationDataset          — paper Tutorial recipe (n,2)
    ScFoundationPerturbationDataset   — newer GEARS recipe (n+1,1) +
                                        5045→19264 main_gene_selection expand
"""
from .pertdata import BasePertData, CidPyGLoaderWrapper
from ._scgpt_perturbation_dataset import ScGPTPerturbationDataset
from ._scfoundation_perturbation_dataset import ScFoundationPerturbationDataset

__all__ = [
    "BasePertData",
    "CidPyGLoaderWrapper",
    "ScGPTPerturbationDataset",
    "ScFoundationPerturbationDataset",
]
