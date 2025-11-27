from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Tuple

from covertreex.algo.batch import BatchInsertPlan, batch_insert
from covertreex.algo.batch_delete import BatchDeletePlan, batch_delete
from covertreex.api.runtime import Runtime
from covertreex.core.tree import PCCTree, TreeBackend
from covertreex.queries.knn import knn as knn_query


def _ensure_points(backend: TreeBackend, value: Any) -> Any:
    arr = backend.asarray(value, dtype=backend.default_float)
    if arr.ndim == 0:
        arr = backend.xp.reshape(arr, (1, 1))
    elif arr.ndim == 1:
        length = int(arr.shape[0])
        if length == 0:
            arr = backend.xp.reshape(arr, (0, 0))
        else:
            arr = backend.xp.reshape(arr, (1, length))
    return backend.device_put(arr)


def _ensure_indices(backend: TreeBackend, value: Any) -> Any:
    arr = backend.asarray(value, dtype=backend.default_int)
    return backend.device_put(arr)


def _convert_tree(tree: PCCTree, backend: TreeBackend) -> PCCTree:
    if tree.backend is backend:
        return tree
    same_backend = (
        tree.backend.name == backend.name
        and tree.backend.default_float == backend.default_float
        and tree.backend.default_int == backend.default_int
    )
    if same_backend:
        return tree
    return tree.to_backend(backend)


def _unwrap_tree(tree: "PCCTree | CoverTree | None") -> PCCTree | None:
    """Extract PCCTree from CoverTree wrapper if needed."""
    if tree is None:
        return None
    if isinstance(tree, CoverTree):
        return tree.tree
    return tree


@dataclass(frozen=True)
class CoverTree:
    """High-performance cover tree for k-NN queries with custom distance metrics.

    Covertreex provides fast nearest-neighbor queries optimized for Vecchia-style
    Gaussian process pipelines. It supports both Euclidean distance and residual
    correlation metrics.

    Parameters
    ----------
    runtime : Runtime, optional
        Configuration for metric type, backend, and engine selection.
        Default uses Euclidean metric with auto-detected backend.
    tree : PCCTree, optional
        Pre-existing tree structure. Usually None; use fit() to build.

    Examples
    --------
    Basic Euclidean k-NN:

        >>> import numpy as np
        >>> from covertreex import CoverTree, Runtime
        >>>
        >>> points = np.random.randn(1000, 3)
        >>> tree = CoverTree().fit(points)
        >>> neighbors, distances = tree.knn(points[:10], k=5, return_distances=True)

    Residual correlation metric for Vecchia GP:

        >>> from covertreex import CoverTree, Runtime, Residual
        >>>
        >>> # V-matrix from inducing points: V[i] = L_mm^{-1} @ K(x_i, inducing)
        >>> # p_diag = diag(K) - ||V||^2  (residual variance)
        >>> residual = Residual(
        ...     v_matrix=V,           # (n, m) whitened inducing features
        ...     p_diag=p_diag,        # (n,) diagonal residual variances
        ...     coords=coords,        # (n, d) spatial coordinates
        ...     kernel_type=0,        # 0=RBF, 1=Matérn 5/2
        ... )
        >>> runtime = Runtime(metric="residual", residual=residual)
        >>> tree = CoverTree(runtime).fit(points)
        >>> neighbors = tree.knn(points, k=50)

    Using the fast Rust backend:

        >>> runtime = Runtime(engine="rust-hilbert")
        >>> tree = CoverTree(runtime).fit(points)

    Notes
    -----
    The tree is immutable. Operations like fit() and insert() return new
    CoverTree instances rather than modifying in-place.

    For Vecchia GP integration, the residual metric computes:

        d_c(i,j) = sqrt(1 - |rho_c(i,j)|)

    where rho_c is the residual correlation after conditioning on inducing points.

    See Also
    --------
    Runtime : Configuration for backend, metric, and engine selection.
    Residual : Configuration for residual correlation metric.
    """

    runtime: Runtime = field(default_factory=Runtime)
    tree: PCCTree | None = None

    def fit(
        self,
        points: Any,
        *,
        apply_batch_order: bool = True,
        mis_seed: int | None = None,
        return_plan: bool = False,
    ) -> "CoverTree | Tuple[CoverTree, BatchInsertPlan]":
        """Build a cover tree from points.

        Parameters
        ----------
        points : array-like, shape (n, d)
            Points to insert into the tree.
        apply_batch_order : bool, default True
            Apply Hilbert curve ordering for cache efficiency.
        mis_seed : int, optional
            Seed for MIS (maximal independent set) randomization.
        return_plan : bool, default False
            If True, also return the BatchInsertPlan for diagnostics.

        Returns
        -------
        tree : CoverTree
            A new CoverTree instance with the constructed tree.
        plan : BatchInsertPlan, optional
            Insertion plan (only if return_plan=True).

        Examples
        --------
        >>> tree = CoverTree().fit(points)
        >>> neighbors = tree.knn(query_points, k=10)
        """
        context = self.runtime.activate()
        backend = context.get_backend()
        batch = _ensure_points(backend, points)
        dimension = int(batch.shape[1])
        base_tree = PCCTree.empty(dimension=dimension, backend=backend)
        new_tree, plan = batch_insert(
            base_tree,
            batch,
            backend=backend,
            mis_seed=mis_seed,
            apply_batch_order=apply_batch_order,
            context=context,
        )
        result = CoverTree(runtime=self.runtime, tree=new_tree)
        return (result, plan) if return_plan else result

    def insert(
        self,
        batch_points: Any,
        *,
        mis_seed: int | None = None,
        apply_batch_order: bool | None = None,
        return_plan: bool = False,
    ) -> "CoverTree | Tuple[CoverTree, BatchInsertPlan]":
        """Insert additional points into the tree.

        Returns a new CoverTree instance (trees are immutable).
        """
        tree = self._require_tree()
        context = self.runtime.activate()
        backend = context.get_backend()
        batch = _ensure_points(backend, batch_points)
        tree_backend = _convert_tree(tree, backend)
        new_tree, plan = batch_insert(
            tree_backend,
            batch,
            backend=backend,
            mis_seed=mis_seed,
            apply_batch_order=apply_batch_order,
            context=context,
        )
        result = CoverTree(runtime=self.runtime, tree=new_tree)
        return (result, plan) if return_plan else result

    def delete(
        self,
        indices: Any,
        *,
        return_plan: bool = False,
    ) -> "CoverTree | Tuple[CoverTree, BatchDeletePlan]":
        """Delete points by index from the tree.

        Returns a new CoverTree instance (trees are immutable).
        """
        tree = self._require_tree()
        context = self.runtime.activate()
        backend = context.get_backend()
        remove = _ensure_indices(backend, indices)
        tree_backend = _convert_tree(tree, backend)
        new_tree, plan = batch_delete(
            tree_backend,
            remove,
            backend=backend,
            context=context,
        )
        result = CoverTree(runtime=self.runtime, tree=new_tree)
        return (result, plan) if return_plan else result

    def knn(
        self,
        query_points: Any,
        *,
        k: int,
        return_distances: bool = False,
    ) -> Any:
        """Find k nearest neighbors for query points.

        Parameters
        ----------
        query_points : array-like, shape (n_queries, d)
            Points to query.
        k : int
            Number of neighbors to return per query.
        return_distances : bool, default False
            If True, also return distances to neighbors.

        Returns
        -------
        indices : ndarray, shape (n_queries, k), dtype int32
            Indices of k nearest neighbors. Padded with -1 if fewer
            than k neighbors exist.
        distances : ndarray, shape (n_queries, k), optional
            Distances to neighbors (only if return_distances=True).

        Examples
        --------
        >>> neighbors = tree.knn(query_points, k=10)
        >>> neighbors, distances = tree.knn(query_points, k=10, return_distances=True)

        Notes
        -----
        For residual metric, distances are residual correlation distances:
        d_c(i,j) = sqrt(1 - |rho_c(i,j)|)
        """
        tree = self._require_tree()
        context = self.runtime.activate()
        backend = context.get_backend()
        tree_backend = _convert_tree(tree, backend)
        queries = _ensure_points(tree_backend.backend, query_points)
        return knn_query(
            tree_backend,
            queries,
            k=k,
            return_distances=return_distances,
            backend=tree_backend.backend,
            context=context,
        )

    def nearest(self, query_points: Any, *, return_distances: bool = False) -> Any:
        """Find the single nearest neighbor for each query point.

        Convenience wrapper around knn(k=1).
        """
        return self.knn(query_points, k=1, return_distances=return_distances)

    @property
    def num_points(self) -> int:
        """Number of points in the tree."""
        return self._require_tree().num_points

    @property
    def dimension(self) -> int:
        """Dimensionality of points in the tree."""
        return self._require_tree().dimension

    def _require_tree(self) -> PCCTree:
        if self.tree is None:
            raise ValueError("CoverTree requires an existing tree; call fit() first.")
        return self.tree


# Deprecated alias for backwards compatibility
PCCT = CoverTree  # Simple alias; deprecation warning would require wrapper
