from __future__ import annotations

import numpy as np
from numba import njit, objmode
from typing import Tuple

# We need some way to interact with the Python kernel provider.
# Objmode is one way.
# Or, we can assume that we are only using the FAST RBF path if `coords` are provided.

# Let's implement a simple binary min-heap for candidates (Priority Queue)
# and a fixed-size max-heap (or sorted array) for K-NN results.

# ------------------------------------------------------------------------------
# Heap Utils
# ------------------------------------------------------------------------------

@njit(fastmath=True)
def _push_min_heap(
    heap_keys: np.ndarray,
    heap_vals: np.ndarray,
    heap_extras: np.ndarray,
    size: int,
    key: float,
    val: int,
    extra: int
) -> int:
    """Push (key, val, extra) onto a min-heap. Returns new size."""
    i = size
    size += 1
    while i > 0:
        p = (i - 1) >> 1
        if heap_keys[p] <= key:
            break
        # Swap
        heap_keys[i] = heap_keys[p]
        heap_vals[i] = heap_vals[p]
        heap_extras[i] = heap_extras[p]
        i = p
    
    heap_keys[i] = key
    heap_vals[i] = val
    heap_extras[i] = extra
    return size

@njit(fastmath=True)
def _pop_min_heap(
    heap_keys: np.ndarray,
    heap_vals: np.ndarray,
    heap_extras: np.ndarray,
    size: int,
) -> Tuple[float, int, int, int]:
    """Pop min element. Returns (key, val, extra, new_size)."""
    if size <= 0:
        return 0.0, -1, -1, 0
        
    ret_key = heap_keys[0]
    ret_val = heap_vals[0]
    ret_extra = heap_extras[0]
    
    size -= 1
    last_key = heap_keys[size]
    last_val = heap_vals[size]
    last_extra = heap_extras[size]
    
    i = 0
    while (i << 1) + 1 < size:
        child = (i << 1) + 1
        if child + 1 < size and heap_keys[child + 1] < heap_keys[child]:
            child += 1
        
        if last_key <= heap_keys[child]:
            break
            
        heap_keys[i] = heap_keys[child]
        heap_vals[i] = heap_vals[child]
        heap_extras[i] = heap_extras[child]
        i = child
        
    heap_keys[i] = last_key
    heap_vals[i] = last_val
    heap_extras[i] = last_extra
    
    return ret_key, ret_val, ret_extra, size

@njit(fastmath=True)
def _update_knn_sorted(
    keys: np.ndarray,
    indices: np.ndarray,
    k: int,
    current_size: int,
    new_key: float,
    new_idx: int,
) -> int:
    """
    Insert (new_key, new_idx) into a SORTED array of size k (Min-K distances).
    Since we want k-NN, we want the SMALLEST distances.
    If size < k, insert.
    If size == k and new_key < max_key, insert and shift.
    """
    if current_size < k:
        # Insert and sort
        pos = current_size
        while pos > 0 and keys[pos - 1] > new_key:
            keys[pos] = keys[pos - 1]
            indices[pos] = indices[pos - 1]
            pos -= 1
        keys[pos] = new_key
        indices[pos] = new_idx
        return current_size + 1
    
    # Else check against worst (last)
    if new_key >= keys[k - 1]:
        return k
        
    # Insert
    pos = k - 1
    while pos > 0 and keys[pos - 1] > new_key:
        keys[pos] = keys[pos - 1]
        indices[pos] = indices[pos - 1]
        pos -= 1
    keys[pos] = new_key
    indices[pos] = new_idx
    return k

# ------------------------------------------------------------------------------
# Tree Traversal Logic
# ------------------------------------------------------------------------------

@njit(fastmath=True)
def _get_children(
    node_idx: int,
    children_arr: np.ndarray,
    next_arr: np.ndarray,
    out_buffer: np.ndarray
) -> int:
    """Fill out_buffer with children of node_idx. Returns count."""
    count = 0
    if node_idx < 0 or node_idx >= children_arr.shape[0]:
        return 0
        
    child = children_arr[node_idx]
    while child >= 0:
        out_buffer[count] = child
        count += 1
        child = next_arr[child]
        # Safety break for cycles? Assuming tree is DAG.
        if child == children_arr[node_idx]: # Should not happen in valid tree
            break
            
    return count

# ------------------------------------------------------------------------------
# Residual Distance Logic (RBF Fast Path)
# ------------------------------------------------------------------------------

@njit(fastmath=True)
def _compute_residual_dist_rbf_batch(
    q_idx: int,
    candidates: np.ndarray,
    count: int,
    # Residual Data
    v_matrix: np.ndarray,
    p_diag: np.ndarray,
    v_norm_sq: np.ndarray,
    # RBF Data
    coords: np.ndarray,
    var: float,
    ls_sq: float,
    # Output
    out_dists: np.ndarray
) -> None:
    """Compute residual distance for RBF kernel fully in Numba."""
    
    # 1. Pre-load Query Data
    # q_idx is index into V-matrix (Dataset Index)
    vq = v_matrix[q_idx]
    pq = p_diag[q_idx]
    nq = v_norm_sq[q_idx]
    xq = coords[q_idx]
    
    for i in range(count):
        c_idx = candidates[i]
        
        # K(q, c) - RBF
        xc = coords[c_idx]
        
        # Euclidean Dist Sq
        d2 = 0.0
        for d in range(xq.shape[0]):
            diff = xq[d] - xc[d]
            d2 += diff * diff
            
        k_val = var * np.exp(-0.5 * d2 / ls_sq)
        
        # Residual Math
        # rho = (K - vq . vc) / sqrt(pq * pc)
        # Actually p_i is (K_ii - ||v_i||^2). The DENOMINATOR is sqrt(p_i * p_j).
        # Wait, check definitions in `covertreex/metrics/residual/core.py`.
        # In `distance_block_no_gate`:
        # denom = sqrt(p_i * p_chunk)
        # num = (kernel - dot)
        # dist = sqrt(1 - abs(num/denom))
        
        vc = v_matrix[c_idx]
        pc = p_diag[c_idx]
        
        dot = 0.0
        for r in range(vq.shape[0]):
            dot += vq[r] * vc[r]
            
        denom = np.sqrt(pq * pc)
        if denom < 1e-9:
            val = 0.0 # correlation 0 -> dist 1? Or max correlation?
            # If variance is 0, correlation is undefined. Usually return 0 dist?
            # If p_i is residual variance. If 0, then point is fully explained.
            # Let's assume safely 1.0 distance if degenerate?
            out_dists[i] = 1.0
        else:
            rho = (k_val - dot) / denom
            # Clamp rho
            if rho > 1.0: rho = 1.0
            if rho < -1.0: rho = -1.0
            out_dists[i] = np.sqrt(1.0 - abs(rho))

# ------------------------------------------------------------------------------
# Main BFS
# ------------------------------------------------------------------------------

@njit(fastmath=True)
def residual_knn_search_numba(
    # Tree Arrays
    children: np.ndarray,
    next_node: np.ndarray,
    parents: np.ndarray,
    
    # Mapping (Node Index -> Dataset Index)
    node_to_dataset: np.ndarray,
    
    # Residual Data
    v_matrix: np.ndarray,
    p_diag: np.ndarray,
    v_norm_sq: np.ndarray,
    
    # Kernel Data (RBF)
    coords: np.ndarray,
    var: float,
    lengthscale: float,
    
    # Query
    q_dataset_idx: int,
    k: int,
    
    # Scratchpads
    heap_keys: np.ndarray,
    heap_vals: np.ndarray,
    heap_extras: np.ndarray,
    knn_keys: np.ndarray,
    knn_indices: np.ndarray,
    visited_bitset: np.ndarray, # If needed, or hashset?
    # Numba set is slow? Bitset is better if N is known.
    
) -> Tuple[np.ndarray, np.ndarray]:
    
    # BFS Heap: (priority, node_idx, extra)
    heap_size = 0
    
    # Initialize with roots
    # Roots are nodes with parent -1
    # We can iterate parents array once or pass roots?
    # Assuming root is 0 if -1 not found?
    # Let's iterate to find roots (slow if many roots?)
    # Usually just one root at 0.
    # Or scan parents.
    # Optimization: Pass roots indices.
    
    # For now, assume root is 0.
    # Push root
    heap_size = _push_min_heap(heap_keys, heap_vals, heap_extras, heap_size, 0.0, 0, 0)
    
    knn_size = 0
    # Initialize knn_keys with inf
    knn_keys[:] = 1e30 # INF
    
    visited_bitset[:] = 0
    
    # Scratch for children
    child_buf = np.empty(1024, dtype=np.int64) # Max degree assumption?
    
    # Scratch for batch eval
    batch_nodes = np.empty(32, dtype=np.int64)
    batch_dists = np.empty(32, dtype=np.float64)
    
    ls_sq = lengthscale * lengthscale
    
    while heap_size > 0:
        # Pop batch
        batch_count = 0
        
        while heap_size > 0 and batch_count < 32:
            prio, node_idx, _, heap_size = _pop_min_heap(heap_keys, heap_vals, heap_extras, heap_size)
            
            # Check visited
            # bitset logic
            word_idx = node_idx >> 6
            bit_idx = node_idx & 63
            if not (visited_bitset[word_idx] & (1 << bit_idx)):
                visited_bitset[word_idx] |= (1 << bit_idx)
                batch_nodes[batch_count] = node_idx
                batch_count += 1
        
        if batch_count == 0:
            break
            
        # Evaluate Batch
        # 1. Map to Dataset Indices
        # We need a scratch for dataset indices?
        # Or map on the fly inside compute.
        # Let's map inside compute loop to avoid allocation.
        
        # We need candidates array of Dataset Indices for the dist func
        # But we can just pass node indices and map inside.
        # Let's simplify `_compute_residual_dist_rbf_batch` to take node indices + map.
        
        # Reuse batch_nodes as candidates (they are node indices)
        # We need to pass `node_to_dataset` to the compute function?
        # Or resolve here.
        
        dataset_indices = np.empty(batch_count, dtype=np.int64)
        for i in range(batch_count):
            dataset_indices[i] = node_to_dataset[batch_nodes[i]]
            
        _compute_residual_dist_rbf_batch(
            q_dataset_idx,
            dataset_indices,
            batch_count,
            v_matrix, p_diag, v_norm_sq,
            coords, var, ls_sq,
            batch_dists
        )
        
        # Process results
        for i in range(batch_count):
            dist = batch_dists[i]
            node_idx = batch_nodes[i]
            
            # Update KNN
            knn_size = _update_knn_sorted(knn_keys, knn_indices, k, knn_size, dist, node_idx)
            
            # Expand children
            # Priority = dist (Heuristic)
            # Pruning? If dist > knn_keys[k-1] + bound?
            # Without strict bound, we just push.
            
            n_children = _get_children(node_idx, children, next_node, child_buf)
            for c in range(n_children):
                child_idx = child_buf[c]
                # Check visited?
                # bitset check
                word_idx = child_idx >> 6
                bit_idx = child_idx & 63
                if not (visited_bitset[word_idx] & (1 << bit_idx)):
                    heap_size = _push_min_heap(heap_keys, heap_vals, heap_extras, heap_size, dist, child_idx, 0)

    return knn_indices[:k], knn_keys[:k]
