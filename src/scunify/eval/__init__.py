# scUnify Evaluation Module
# Integrated embedding evaluation framework
# scIB + scGraph (individual model) + CKA + N2O (multi-model)

from ._benchmarker import Evaluator
from ._scib import ScibWrapper
from ._scgraph import ScGraphWrapper
from ._cka import CKAWrapper, linear_cka
from ._n2o import N2OWrapper
from ._plotting import plot_combined_table

__all__ = [
    "Evaluator",
    "ScibWrapper",
    "ScGraphWrapper",
    "CKAWrapper",
    "linear_cka",
    "N2OWrapper",
    "plot_combined_table",
]
