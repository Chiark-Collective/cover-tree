"""Covertreex: High-performance cover tree for k-NN queries.

Quick Start
-----------
>>> from covertreex import CoverTree, Runtime, Residual
>>>
>>> # Basic Euclidean k-NN
>>> tree = CoverTree().fit(points)
>>> neighbors = tree.knn(query_points, k=10)
>>>
>>> # Residual correlation metric for Vecchia GP
>>> residual = Residual(v_matrix=V, p_diag=p_diag, coords=coords)
>>> runtime = Runtime(metric="residual", residual=residual)
>>> tree = CoverTree(runtime).fit(points)
>>> neighbors = tree.knn(points, k=50)

Classes
-------
CoverTree : Main interface for building trees and running k-NN queries.
Runtime : Configuration for backend, metric, and engine selection.
Residual : Configuration for residual correlation metric (Vecchia GP).
"""

from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("covertreex")
except Exception:  # pragma: no cover - best effort during local development
    __version__ = "0.0.1"

# Primary user-facing API
from .api import CoverTree, Runtime, Residual, PCCT

# Internal/advanced APIs
from .engine import CoverTree as EngineCoverTree, build_tree, get_engine
from .core import (
    PCCTree,
    TreeBackend,
    TreeLogStats,
    available_metrics,
    configure_residual_metric,
    get_metric,
    reset_residual_metric,
)
from .metrics.residual import (
    ResidualCorrHostData,
    configure_residual_correlation,
)
from .baseline import (
    BaselineCoverTree,
    BaselineNode,
    ExternalCoverTreeBaseline,
    GPBoostCoverTreeBaseline,
    MlpackCoverTreeBaseline,
    has_external_cover_tree,
    has_gpboost_cover_tree,
    has_mlpack_cover_tree,
)

__all__ = [
    # Primary API
    "__version__",
    "CoverTree",
    "Runtime",
    "Residual",
    "PCCT",  # Deprecated alias
    # Engine-level API
    "build_tree",
    "get_engine",
    "EngineCoverTree",
    # Internal
    "PCCTree",
    "TreeBackend",
    "TreeLogStats",
    "available_metrics",
    "configure_residual_metric",
    "configure_residual_correlation",
    "get_metric",
    "reset_residual_metric",
    "ResidualCorrHostData",
    "BaselineCoverTree",
    "BaselineNode",
    "ExternalCoverTreeBaseline",
    "GPBoostCoverTreeBaseline",
    "MlpackCoverTreeBaseline",
    "has_external_cover_tree",
    "has_gpboost_cover_tree",
    "has_mlpack_cover_tree",
]
