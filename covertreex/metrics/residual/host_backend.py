from __future__ import annotations

from collections.abc import Callable

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


def _matern52_kernel(
    x: np.ndarray,
    y: np.ndarray,
    *,
    variance: float,
    lengthscale: float,
) -> np.ndarray:
    """Matern 5/2 kernel: K = var * (1 + a + a²/3) * exp(-a) where a = sqrt(5) * r."""
    diff = x[:, None, :] - y[None, :, :]
    sq_dist = np.sum(diff * diff, axis=2, dtype=np.float64)
    denom = max(lengthscale, 1e-12)
    r = np.sqrt(sq_dist) / denom
    sqrt5 = np.sqrt(5.0)
    a = sqrt5 * r
    poly = 1.0 + a + (a * a) / 3.0
    return float(variance) * poly * np.exp(-a, dtype=np.float64)


def _build_sgemm_rbf_provider(
    points: np.ndarray,
    *,
    variance: float,
    lengthscale: float,
) -> tuple[np.ndarray, np.ndarray, Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    points_f32 = np.ascontiguousarray(points, dtype=np.float32)
    row_norms = np.sum(points_f32 * points_f32, axis=1, dtype=np.float32)
    variance32 = np.float32(variance)
    denom = max(lengthscale, 1e-12)
    gamma = np.float32(1.0 / (denom * denom))

    def provider(row_indices: np.ndarray, col_indices: np.ndarray) -> np.ndarray:
        row_idx = np.asarray(row_indices, dtype=np.int64)
        col_idx = np.asarray(col_indices, dtype=np.int64)
        if row_idx.size == 0 or col_idx.size == 0:
            return np.zeros((row_idx.size, col_idx.size), dtype=np.float32)
        rows = points_f32[row_idx]
        cols = points_f32[col_idx]
        gram = rows @ cols.T  # float32 SGEMM
        dist2 = row_norms[row_idx][:, None] + row_norms[col_idx][None, :]
        dist2 -= 2.0 * gram
        np.maximum(dist2, 0.0, out=dist2)
        # reuse dist2 buffer for scaled distances
        dist2 *= (-0.5 * gamma)
        np.exp(dist2, out=dist2)
        dist2 *= variance32
        return dist2

    return points_f32, row_norms, provider


def _build_sgemm_matern52_provider(
    points: np.ndarray,
    *,
    variance: float,
    lengthscale: float,
) -> tuple[np.ndarray, np.ndarray, Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """Build a Matern 5/2 kernel provider using SGEMM for fast computation."""
    points_f32 = np.ascontiguousarray(points, dtype=np.float32)
    row_norms = np.sum(points_f32 * points_f32, axis=1, dtype=np.float32)
    variance32 = np.float32(variance)
    denom = max(lengthscale, 1e-12)
    inv_ls = np.float32(1.0 / denom)
    sqrt5 = np.float32(np.sqrt(5.0))
    inv_three = np.float32(1.0 / 3.0)

    def provider(row_indices: np.ndarray, col_indices: np.ndarray) -> np.ndarray:
        row_idx = np.asarray(row_indices, dtype=np.int64)
        col_idx = np.asarray(col_indices, dtype=np.int64)
        if row_idx.size == 0 or col_idx.size == 0:
            return np.zeros((row_idx.size, col_idx.size), dtype=np.float32)
        rows = points_f32[row_idx]
        cols = points_f32[col_idx]
        gram = rows @ cols.T  # float32 SGEMM
        dist2 = row_norms[row_idx][:, None] + row_norms[col_idx][None, :]
        dist2 -= 2.0 * gram
        np.maximum(dist2, 0.0, out=dist2)
        # r = sqrt(dist2) / lengthscale
        r = np.sqrt(dist2, dtype=np.float32) * inv_ls
        # a = sqrt(5) * r
        a = sqrt5 * r
        # poly = 1 + a + a^2/3
        poly = 1.0 + a + (a * a) * inv_three
        # K = var * poly * exp(-a)
        result = variance32 * poly * np.exp(-a, dtype=np.float32)
        return result

    return points_f32, row_norms, provider


def _point_decoder_factory(points: np.ndarray) -> Callable[[ArrayLike], np.ndarray]:
    points_contig = np.ascontiguousarray(points, dtype=np.float64)
    point_keys = [tuple(row.tolist()) for row in points_contig]
    point_keys_f32 = [tuple(row.astype(np.float32).tolist()) for row in points_contig]
    index_map: dict[tuple[float, ...], int] = {}
    for idx, key in enumerate(point_keys):
        index_map.setdefault(key, idx)
    for idx, key in enumerate(point_keys_f32):
        index_map.setdefault(key, idx)

    def decoder(values: ArrayLike) -> np.ndarray:
        # Accept either integer dataset ids (shape (N,) or (N,1)) or coordinate payloads.
        arr = np.asarray(values)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.dtype.kind in {"i", "u"}:
            flat = arr.reshape(-1)
            flat_i64 = flat.astype(np.int64, copy=False)
            if flat_i64.min(initial=0) < 0 or flat_i64.max(initial=-1) >= points_contig.shape[0]:
                raise ValueError("Residual point decoder received out-of-range indices.")
            return flat_i64
        arr_f64 = np.asarray(arr, dtype=np.float64)
        if arr_f64.ndim == 1:
            arr_f64 = arr_f64.reshape(1, -1)
        if arr_f64.shape[1] == 1:
            col = arr_f64[:, 0]
            rounded = np.rint(col).astype(np.int64)
            if np.allclose(col, rounded, atol=1e-8):
                if rounded.min(initial=0) < 0 or rounded.max(initial=-1) >= points_contig.shape[0]:
                    raise ValueError("Residual point decoder received out-of-range indices.")
                return rounded
        if arr_f64.shape[1] != points_contig.shape[1]:
            raise ValueError(
                "Residual point decoder expected payload dimensionality "
                f"{points_contig.shape[1]}, received {arr_f64.shape[1]}."
            )
        rows = np.ascontiguousarray(arr_f64, dtype=np.float64)
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
    kernel_type: int = 0,
) -> ResidualCorrHostData:
    """
    Build a :class:`ResidualCorrHostData` cache for the residual-correlation metric.

    Parameters
    ----------
    kernel_type : int
        Kernel type to use: 0 = RBF (default), 1 = Matern 5/2.
        The V-matrix, p_diag, and kernel provider will all use the specified kernel
        to ensure mathematical consistency in the residual correlation computation.
    """

    if points.size == 0:
        raise ValueError("Residual metric requires at least one point to configure caches.")

    points_np = np.asarray(points, dtype=np.float64)
    generator = rng or default_rng(seed)
    n_points = points.shape[0]
    inducing = min(inducing_count, n_points)
    if inducing <= 0:
        inducing = min(32, n_points)
    if inducing < n_points:
        inducing_idx = np.sort(generator.choice(n_points, size=inducing, replace=False))
    else:
        inducing_idx = np.arange(n_points)
    inducing_points = points_np[inducing_idx]

    # Select kernel function based on kernel_type
    if kernel_type == 1:
        kernel_fn = _matern52_kernel
    else:
        kernel_fn = _rbf_kernel

    k_mm = kernel_fn(inducing_points, inducing_points, variance=variance, lengthscale=lengthscale)
    jitter = 1e-6 * variance
    k_mm = k_mm + np.eye(inducing_points.shape[0], dtype=np.float64) * jitter
    l_mm = np.linalg.cholesky(k_mm)

    k_xm = kernel_fn(points_np, inducing_points, variance=variance, lengthscale=lengthscale)
    solve_result = np.linalg.solve(l_mm, k_xm.T)
    v_matrix = solve_result.T

    kernel_diag = np.full(n_points, variance, dtype=np.float64)
    p_diag = np.maximum(kernel_diag - np.sum(v_matrix * v_matrix, axis=1), 1e-9)

    point_decoder = _point_decoder_factory(points_np)

    # Select SGEMM provider based on kernel_type
    if kernel_type == 1:
        kernel_points_f32, kernel_row_norms, kernel_provider = _build_sgemm_matern52_provider(
            points_np,
            variance=variance,
            lengthscale=lengthscale,
        )
    else:
        kernel_points_f32, kernel_row_norms, kernel_provider = _build_sgemm_rbf_provider(
            points_np,
            variance=variance,
            lengthscale=lengthscale,
        )

    host_backend = ResidualCorrHostData(
        v_matrix=np.asarray(v_matrix, dtype=np.float32),
        p_diag=np.asarray(p_diag, dtype=np.float32),
        kernel_diag=np.asarray(kernel_diag, dtype=np.float32),
        kernel_provider=kernel_provider,
        point_decoder=point_decoder,
        chunk_size=int(chunk_size),
        kernel_points_f32=kernel_points_f32,
        kernel_row_norms_f32=kernel_row_norms,
    )
    return host_backend
