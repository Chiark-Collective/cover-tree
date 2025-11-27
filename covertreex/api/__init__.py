"""Public API for covertreex cover tree library.

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
"""

from .pcct import CoverTree, PCCT  # PCCT is deprecated alias
from .runtime import Residual, Runtime

__all__ = [
    "CoverTree",
    "PCCT",  # Deprecated, use CoverTree
    "Runtime",
    "Residual",
]
