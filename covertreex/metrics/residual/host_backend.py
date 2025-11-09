from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import ArrayLike

from .core import ResidualCorrHostData

__all__ = ["build_residual_backend"]


def _rbf_kernel(
    x: np.ndarray,
    y: np.ndarray,
    *,
    variance: float,
    lengthscale: float,
) -> np.ndarray:
    diff = x[:, None, :] - y[None, :, :]
    sq_dist = np.sum(diff * diff, axis=2, dtype=np.float64)
    denom = max(lengthscale, 1e-12)
    scaled = -0.5 * sq_dist / (denom * denom)
    return float(variance) * np.exp(scaled, dtype=np.float64)


def _point_decoder_factory(points: np.ndarray) -> Callable[[ArrayLike], np.ndarray]:
    points_contig = np.ascontiguousarray(points, dtype=np.float64)
    point_keys = [tuple(row.tolist()) for row in points_contig]
    index_map: dict[tuple[float, ...], int] = {}
    for idx, key in enumerate(point_keys):
        index_map.setdefault(key, idx)

    def decoder(values: ArrayLike) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != points_contig.shape[1]:
            raise ValueError(
                "Residual point decoder expected payload dimensionality "
                f"{points_contig.shape[1]}, received {arr.shape[1]}."
            )
        rows = np.ascontiguousarray(arr, dtype=np.float64)
        indices = np.empty(rows.shape[0], dtype=np.int64)
        for i, row in enumerate(rows):
            key = tuple(row.tolist())
            if key not in index_map:
                raise KeyError("Residual point decoder received unknown payload.")
            indices[i] = index_map[key]
        return indices

    return decoder


def build_residual_backend(
    points: np.ndarray,
    *,
    seed: int,
    inducing_count: int,
    variance: float,
    lengthscale: float,
    chunk_size: int = 512,
    rng: Generator | None = None,
) -> ResidualCorrHostData:
    """
    Build a :class:`ResidualCorrHostData` cache for the residual-correlation metric.
    """

    if points.size == 0:
        raise ValueError("Residual metric requires at least one point to configure caches.")

    generator = rng or default_rng(seed)
    n_points = points.shape[0]
    inducing = min(inducing_count, n_points)
    if inducing <= 0:
        inducing = min(32, n_points)
    if inducing < n_points:
        inducing_idx = np.sort(generator.choice(n_points, size=inducing, replace=False))
    else:
        inducing_idx = np.arange(n_points)
    inducing_points = points[inducing_idx]

    k_mm = _rbf_kernel(inducing_points, inducing_points, variance=variance, lengthscale=lengthscale)
    jitter = 1e-6 * variance
    k_mm = k_mm + np.eye(inducing_points.shape[0], dtype=np.float64) * jitter
    l_mm = np.linalg.cholesky(k_mm)

    k_xm = _rbf_kernel(points, inducing_points, variance=variance, lengthscale=lengthscale)
    solve_result = np.linalg.solve(l_mm, k_xm.T)
    v_matrix = solve_result.T

    kernel_diag = np.full(n_points, variance, dtype=np.float64)
    p_diag = np.maximum(kernel_diag - np.sum(v_matrix * v_matrix, axis=1), 1e-9)

    point_decoder = _point_decoder_factory(points)

    def kernel_provider(row_indices: np.ndarray, col_indices: np.ndarray) -> np.ndarray:
        rows = points[np.asarray(row_indices, dtype=np.int64, copy=False)]
        cols = points[np.asarray(col_indices, dtype=np.int64, copy=False)]
        return _rbf_kernel(rows, cols, variance=variance, lengthscale=lengthscale)

    host_backend = ResidualCorrHostData(
        v_matrix=np.asarray(v_matrix, dtype=np.float64, copy=False),
        p_diag=np.asarray(p_diag, dtype=np.float64, copy=False),
        kernel_diag=kernel_diag,
        kernel_provider=kernel_provider,
        point_decoder=point_decoder,
        chunk_size=int(chunk_size),
    )
    return host_backend
