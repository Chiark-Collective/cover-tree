# VIF Residual Correlation → Host Cover Tree (High-Level Sketch)

This sketch captures the control flow and residency boundaries between the VIF
Gaussian process residual computations (JAX, accelerator) and the downstream
cover-tree neighbour selection (Numba, CPU). Function bodies are intentionally
stubbed; comments describe the work each stage performs.

```python
"""
Sketch of a production pipeline:
  - Device (GPU/NPU via JAX): build residual-correlation factors once.
  - Host (CPU/Numba): construct and query a cover tree using those factors.
  - Scheduler: streams residual rows in batches to keep host workers saturated.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterator, Sequence

import jax.numpy as jnp
import numpy as np


# --------------------------------------------------------------------------- #
# Data containers / boundaries

@dataclass(frozen=True)
class ResidualDeviceCaches:
    """Lives on the accelerator (JAX arrays)."""
    L_mm: jax.Array            # lower-triangular factor of K_mm
    V_all: jax.Array           # V = L_mm^{-1} K(X, U)  (n, m)
    p_diag: jax.Array          # residual diagonal for each training point (n,)


@dataclass(frozen=True)
class ResidualHostCaches:
    """Pinned host copies consumed by Numba cover-tree kernels."""
    V_all_host: np.ndarray     # (n, m), dtype float64
    p_diag_host: np.ndarray    # (n,)
    kernel_diag: np.ndarray    # (n,)
    meta: dict                 # bookkeeping (lengthscale, signal variance, etc.)


@dataclass(frozen=True)
class CoverTreeArtifacts:
    """Opaque handle returned by the host stage."""
    structure: object          # frozen CSR buffers / root index
    stats: dict                # build/query timings, lower-bound counters


# --------------------------------------------------------------------------- #
# Accelerator-side preparation

def compute_residual_caches_device(params, X_train, inducing_points, kernel_strategy) -> ResidualDeviceCaches:
    """
    JIT-ed JAX routine.
      - Computes K_mm, factors via cholesky.
      - Forms V = L_mm^{-1} K(X, U).
      - Derives p_diag = max(diag(K_xx) - ||V_i||^2, eps).
    Output stays on device to hide host latency until explicitly staged.
    """
    raise NotImplementedError


def stage_caches_to_host(device_caches: ResidualDeviceCaches, *, force_f64: bool = True) -> ResidualHostCaches:
    """
    Transfers arrays to host (blocking jax.device_get).
      - Optionally casts to float64 for Numba stability.
      - Precomputes scalar kernel metadata reused across host distance calls.
      - Ensures buffers land in pinned memory (so multiple host workers can
        reuse without extra copies).
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Streaming residual correlation blocks to cover tree builders

def residual_corr_block_iterator(
    host_caches: ResidualHostCaches,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Produces host blocks (indices, dense residual rows) needed by hybrid paths:
      - GPU screeners request V rows by index.
      - Pure cover tree uses on-demand callbacks; this iterator mirrors that
        contract when warm-starting candidate sets.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Host-side cover tree construction / query (Numba)

def build_cover_tree_host(
    host_caches: ResidualHostCaches,
    *,
    candidate_cap: int,
    parallel_blocks: int,
) -> CoverTreeArtifacts:
    """
    Launches the Numba cover-tree builder.
      - SPMD-style: each worker receives disjoint index ranges.
      - Distance callback closes over host_caches (V, p_diag, kernel_diag).
      - Aggregates timing counters for diagnostics.
    """
    raise NotImplementedError


def query_cover_tree_host(
    artifacts: CoverTreeArtifacts,
    *,
    host_caches: ResidualHostCaches,
    X_star: np.ndarray,
    batch_scheduler,
) -> np.ndarray:
    """
    Queries the frozen cover tree for prediction.
      - `batch_scheduler` yields (x_star block, candidate cap) tuples.
      - Reuses staged V_star, p_star computed via JAX but transferred in blocks.
      - Returns neighbor indices (shape n_star × k).
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# Orchestrator tying device + host stages together

def build_cover_tree_with_vif_residuals(
    params,
    X_train,
    inducing_points,
    kernel_strategy,
    *,
    candidate_cap: int = 2048,
    parallel_blocks: int = 4,
) -> CoverTreeArtifacts:
    """
    Entry point for training-time neighbour selection.
      1. Run the accelerator prep once (JIT warmup amortised here).
      2. Stage residual factors to host.
      3. Launch threaded cover-tree build while we still have the device handle
         in case we need extra kernel rows (fallback path).
    """
    device_caches = compute_residual_caches_device(params, X_train, inducing_points, kernel_strategy)
    host_caches = stage_caches_to_host(device_caches, force_f64=True)
    artifacts = build_cover_tree_host(
        host_caches,
        candidate_cap=candidate_cap,
        parallel_blocks=parallel_blocks,
    )
    return artifacts


def refresh_neighbors_async(
    params,
    X_train,
    inducing_points,
    kernel_strategy,
    *,
    num_workers: int,
    request_queue,
    result_sink,
) -> None:
    """
    Background loop used by `NeighborRefreshController`.
      - Device stage happens once; host buffers are captured in the closure.
      - Each incoming request asks for either:
          * a full rebuild (e.g., new Vecchia order)
          * a partial query (new X_star batch)
      - Work items dispatched across a `ThreadPoolExecutor`, one worker per host
        partition to avoid GIL interference with Numba.
    """
    caches_device = compute_residual_caches_device(params, X_train, inducing_points, kernel_strategy)
    caches_host = stage_caches_to_host(caches_device)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        while True:
            work = request_queue.get()
            if work.kind == "shutdown":
                break
            if work.kind == "rebuild":
                pool.submit(
                    _handle_rebuild,
                    caches_host,
                    work.candidate_cap,
                    work.parallel_blocks,
                    result_sink,
                )
            elif work.kind == "query":
                pool.submit(
                    _handle_query,
                    caches_host,
                    work.artifacts,
                    work.X_star,
                    work.batch_scheduler,
                    result_sink,
                )


def _handle_rebuild(host_caches, candidate_cap, parallel_blocks, result_sink):
    """Host worker body for rebuild tasks (Numba only)."""
    artifacts = build_cover_tree_host(host_caches, candidate_cap=candidate_cap, parallel_blocks=parallel_blocks)
    result_sink.put(("rebuild_done", artifacts))


def _handle_query(host_caches, artifacts, X_star, batch_scheduler, result_sink):
    """Host worker body for query tasks (Numba only)."""
    neighbors = query_cover_tree_host(
        artifacts,
        host_caches=host_caches,
        X_star=X_star,
        batch_scheduler=batch_scheduler,
    )
    result_sink.put(("query_done", neighbors))
```

## Key Residency / Scheduling Notes

- **Device → Host Handoff**: `compute_residual_caches_device` runs once per refresh
  (JAX side). Results are staged to pinned host buffers exactly once unless the
  inducing set changes.
- **Host Parallelism**: All cover-tree work (`build_cover_tree_host`, `_handle_*`)
  executes in CPU threads; Numba kernels release the GIL, so `ThreadPoolExecutor`
  holds one worker per partition block.
- **Streaming X\*_batches**: Prediction queries reuse the staged training caches.
  Fresh `V_star`/`p_star` batches are computed on device (mirroring `prepare_star`
  in the production code) and staged through the `batch_scheduler`.
- **Fallback Hooks**: Because `device_caches` remain valid, host workers can
  request extra kernel rows by invoking a small helper that dispatches back to
  the accelerator (e.g., for numerical repair) without re-running the full
  pipeline.

# Follow-up

• Residual Metric Sources

  - Core JAX definition is vif_residual_correlation_rows (survi/models/vif/ops/prediction.py:457), which builds
    K_mm, solves for V = L_mm^{-1} K(X,U), forms p_diag = max(K_xx - ||V_i||^2, 1e-9), and returns ρ_c(i,j) =
    (K(X_i,X_j) - V_i·V_j) / sqrt(p_i p_j).
  - The selector reuses the same math in pure JAX via compute_residual_corr_block (survi/models/selectors/
    residual_corr.py:17), so downstream code already has a jit-friendly block helper if you want batched host
    precompute.

  Host Distance + Bounds in Cover Tree

  - _ResidualCorrBackend materialises the host copies of V/p_diag and exposes the NumPy implementations (survi/
    models/selectors/cover_tree.py:131).
      - dc_to_many_train at survi/models/selectors/cover_tree.py:257 maps ρ_c to the cover-tree metric d =
        sqrt(max(0, 1 - |ρ|)), forcing exact zeros on self pairs.
      - dc_star_to_many_train (survi/models/selectors/cover_tree.py:292) is the same formula for external points
        using staged (V_star, p_star) from prepare_star.
  - The lower-bound hook used by the streaming (sparse) traversal is _ResidualCorrBackend.lower_bound_train (survi/
    models/selectors/cover_tree.py:308). It keeps a cached raw kernel entry K_ij, normalises it by sqrt(K_ii K_jj),
    clamps to [-1,1], and returns sqrt(max(0, 1 - ratio)). That is the “residual bound”: it assumes worst-case
    cancellation of the low-rank part, so the true residual metric is always ≤ this bound.
  - When the bound survives pruning, _ResidualCorrBackend.dc_single_train_from_kernel (survi/models/selectors/
    cover_tree.py:329) finishes the exact residual distance using the cached K_ij plus V_i, V_j, and their residual
    diagonals.

  Radius / Pruning Mechanics

  - Cover-tree radii are initialised to 1.0 and shrink geometrically by / base per level (survi/models/selectors/
    cover_tree.py:401). Because d ∈ [0,1], level-0 radius covers the entire set.
  - During query, the pruning threshold is kth_distance + R_{ℓ-1} (survi/models/selectors/cover_tree.py:624 for
    predecessors, :764 for externals). The radius we add back is the previous level’s value, matching the standard
    cover-tree guarantee.
  - query_predecessors (lines survi/models/selectors/cover_tree.py:568-717) shows how lower bounds plug in: when
    streaming mode is enabled, it calls lower_bound_fn/dist_single_fn per candidate, skipping full evaluations if
    the bound already exceeds the current worst neighbour.

  Numba Path / Contracts

  - The Numba kernels mirror these helpers; see the helpers referenced in query_all_predecessors_numba and
    query_all_external_numba (boundaries documented in notes/cover_tree_gpboost_numba_implementation.md:68, :175).
    Those snippets call _residual_corr_to_distance, which is the same sqrt(1 - |ρ|) transformation with the same
    eps floors.

  With these references you should be able to thread the metric-aware bounds into your traversal kernels without
  guessing: precompute V, p_diag, optionally retain K_diag, and reuse the exact/bound helpers above when porting to
  the parallel host implementation.
