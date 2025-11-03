from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import jax
import numpy as np

from covertreex.algo.traverse import TraversalResult
from covertreex.core.tree import PCCTree, TreeBackend


@dataclass(frozen=True)
class ConflictGraph:
    """Conflict graph encoded in CSR form."""

    indptr: Any
    indices: Any
    scope_indptr: Any
    scope_indices: Any
    radii: Any
    annulus_bounds: Any
    annulus_bins: Any
    annulus_bin_indptr: Any
    annulus_bin_indices: Any
    annulus_bin_ids: Any

    @property
    def num_nodes(self) -> int:
        return int(self.indptr.shape[0] - 1)

    @property
    def num_edges(self) -> int:
        return int(self.indices.shape[0])

    @property
    def num_scopes(self) -> int:
        return int(self.scope_indptr.shape[0] - 1)


def build_conflict_graph(
    tree: PCCTree,
    traversal: TraversalResult,
    batch_points: Any,
    *,
    backend: TreeBackend | None = None,
) -> ConflictGraph:
    """Construct a conflict graph with distance-aware pruning."""

    backend = backend or tree.backend
    scopes = traversal.conflict_scopes
    batch = backend.asarray(batch_points, dtype=backend.default_float)
    xp = backend.xp
    if batch.size:
        diff = batch[:, None, :] - batch[None, :, :]
        pairwise = xp.sqrt(xp.sum(diff * diff, axis=-1))
        pairwise_np = backend.to_numpy(pairwise)
    else:
        pairwise_np = []

    batch_size = int(traversal.parents.shape[0])
    adjacency: List[set[int]] = [set() for _ in range(batch_size)]
    membership: Dict[int, List[int]] = {}

    parents_np = backend.to_numpy(traversal.parents)
    levels_np = backend.to_numpy(traversal.levels)

    radii: List[float] = []
    for parent, level in zip(parents_np, levels_np):
        if parent < 0:
            radii.append(float("inf"))
        else:
            base = 2.0 ** (float(level) + 1.0)
            si = float(tree.si_cache[parent])
            radii.append(max(base, si))

    for point_idx, scope in enumerate(scopes):
        for node_idx in scope:
            membership.setdefault(int(node_idx), []).append(point_idx)

    for nodes in membership.values():
        if len(nodes) < 2:
            continue
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                distance = pairwise_np[u][v] if batch_size else 0.0
                threshold = min(radii[u], radii[v])
                if distance <= threshold:
                    adjacency[u].add(v)
                    adjacency[v].add(u)

    indptr_list: List[int] = [0]
    indices_list: List[int] = []
    for neighbors in adjacency:
        sorted_neighbors = sorted(neighbors)
        indices_list.extend(sorted_neighbors)
        indptr_list.append(len(indices_list))

    indptr = backend.asarray(indptr_list, dtype=backend.default_int)
    indices = backend.asarray(indices_list, dtype=backend.default_int)
    scope_indptr_arr = traversal.scope_indptr
    scope_indices_arr = traversal.scope_indices
    radii_arr = backend.asarray(radii, dtype=backend.default_float)
    annulus_bounds = backend.asarray(
        [(0.0, float(radius)) for radius in radii], dtype=backend.default_float
    )
    log_r = xp.log2(xp.maximum(radii_arr, 1.0))
    annulus_bins = xp.floor(log_r).astype(backend.default_int)
    annulus_bins = xp.where(xp.isinf(radii_arr), -1, annulus_bins)

    annulus_bins_host = np.asarray(backend.to_numpy(annulus_bins), dtype=np.int64)
    if annulus_bins_host.size:
        sorted_order = np.argsort(annulus_bins_host, kind="stable")
        sorted_bins = annulus_bins_host[sorted_order]
        bin_ids: List[int] = []
        counts: List[int] = []
        current = int(sorted_bins[0])
        count = 0
        for val in sorted_bins:
            if int(val) != current:
                bin_ids.append(current)
                counts.append(count)
                current = int(val)
                count = 1
            else:
                count += 1
        bin_ids.append(current)
        counts.append(count)
        bin_indptr_np = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])
    else:
        sorted_order = np.array([], dtype=np.int64)
        bin_ids = []
        bin_indptr_np = np.array([0], dtype=np.int64)

    annulus_bin_indices = backend.asarray(sorted_order, dtype=backend.default_int)
    annulus_bin_indptr = backend.asarray(bin_indptr_np, dtype=backend.default_int)
    annulus_bin_ids = backend.asarray(bin_ids, dtype=backend.default_int)

    return ConflictGraph(
        indptr=indptr,
        indices=indices,
        scope_indptr=scope_indptr_arr,
        scope_indices=scope_indices_arr,
        radii=radii_arr,
        annulus_bounds=annulus_bounds,
        annulus_bins=backend.device_put(annulus_bins),
        annulus_bin_indptr=annulus_bin_indptr,
        annulus_bin_indices=annulus_bin_indices,
        annulus_bin_ids=annulus_bin_ids,
    )
