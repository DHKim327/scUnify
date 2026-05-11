"""Perturbation task — GEARS package vendored from
``RelatedWorks/Foundations/scFoundation/GEARS/gears/`` (paper-faithful
byte-level copy).

Used by both ScGPTPerturbationMixin (PertData + metrics only) and
ScFoundationPerturbationMixin (PertData + GEARS_Model). The vendored copy
keeps the framework self-contained and avoids sys.path manipulation.

Public API:
    PertData          — dataset wrapper (train/val/test split, PyG cell graphs)
    loss_fct          — MSE + direction penalty (scFoundation perturbation)
    uncertainty_loss_fct
    compute_perturbation_metrics  — pearson/pearson_de/pearson_delta/pearson_de_delta
    deeper_analysis, non_dropout_analysis  — subgroup metrics
"""
from .pertdata import PertData
from .utils import loss_fct, uncertainty_loss_fct
from .inference import (
    compute_perturbation_metrics,
    deeper_analysis,
    non_dropout_analysis,
)

__all__ = [
    "PertData",
    "loss_fct",
    "uncertainty_loss_fct",
    "compute_perturbation_metrics",
    "deeper_analysis",
    "non_dropout_analysis",
]
