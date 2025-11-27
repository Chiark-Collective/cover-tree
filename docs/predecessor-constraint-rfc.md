# RFC: Predecessor Constraint for Vecchia GP Neighbor Selection

## Summary

Add native support for the **predecessor constraint** to covertreex's k-NN queries. This constraint requires that for each query point with index `i`, only points with index `j < i` can be returned as neighbors. This is essential for Vecchia GP approximations.

## Motivation

Vecchia approximations for Gaussian Processes require a specific ordering constraint: when selecting neighbors for point `i`, only points that appear *before* it in the ordering (indices `j < i`) are valid candidates. This ensures the conditional independence structure required by the approximation.

Currently, survi (and other Vecchia GP implementations) must:
1. Query k-NN without the constraint
2. Post-filter results to keep only predecessors
3. Hope enough valid neighbors remain

This is inefficient because:
- The tree may return mostly invalid neighbors (especially for early points)
- Post-filtering can leave fewer than k neighbors
- Wasted computation on neighbors that will be discarded

## Proposed API

### High-level Python API

```python
from covertreex import CoverTree, Runtime

tree = CoverTree(runtime).fit(points)

# Standard k-NN (existing behavior)
neighbors = tree.knn(query_points, k=10)

# Predecessor-constrained k-NN (NEW)
neighbors = tree.knn(query_points, k=10, predecessor_mode=True)
```

When `predecessor_mode=True`:
- For query at index `i`, only return neighbors with index `j < i`
- Query 0 returns empty (no predecessors exist)
- Query 1 can only return index 0
- Query `i` can return up to `min(k, i)` neighbors

### Function-level API

```python
# queries/knn.py
def knn(
    tree: PCCTree,
    query_points: Any,
    *,
    k: int,
    return_distances: bool = False,
    predecessor_mode: bool = False,  # NEW
    backend: TreeBackend | None = None,
    context: RuntimeContext | None = None,
) -> Tuple[Any, Any] | Any:
```

## Implementation Plan

### Phase 1: Python Fallback Path

**File:** `covertreex/queries/knn.py`

Modify `_single_query_knn()` to accept `query_idx` parameter:

```python
def _single_query_knn(
    query: np.ndarray,
    *,
    query_idx: int | None = None,  # NEW
    points: np.ndarray,
    si_cache: np.ndarray,
    child_cache: ChildChainCache,
    root_indices: Sequence[int],
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # ...

    def _push_candidate(idx: int) -> None:
        nonlocal counter
        if idx < 0 or idx >= num_points:
            return

        # NEW: Predecessor constraint
        if query_idx is not None and idx >= query_idx:
            return

        if idx in visited or idx in enqueued:
            return
        # ... rest unchanged
```

Update the caller loop in `_knn_impl()`:

```python
for i, query in enumerate(batch_np):
    q_idx = i if predecessor_mode else None
    indices, distances = _single_query_knn(
        query,
        query_idx=q_idx,  # Pass index for filtering
        points=points_np,
        # ...
    )
```

### Phase 2: Rust Core Algorithm

**File:** `src/algo.rs`

#### 2a. Update `single_residual_knn_query` signature

```rust
fn single_residual_knn_query<'a, T>(
    tree: &'a CoverTreeData<T>,
    node_to_dataset: &[i64],
    metric: &ResidualMetric<'a, T>,
    q_dataset_idx: usize,
    k: usize,
    scope_caps: Option<&HashMap<i32, T>>,
    max_neighbor_idx: Option<usize>,  // NEW: if Some(n), only ds_idx < n allowed
    mut telemetry: Option<&mut ResidualQueryTelemetry>,
    ctx: &mut SearchContext<T>,
) -> (Vec<i64>, Vec<T>)
```

#### 2b. Filter at root handling (line ~636-644)

```rust
// Root distance and seed result heap
let root_payload = tree.get_point_row(0);
let root_dataset_idx = root_payload[0].to_usize().unwrap();

// NEW: Check predecessor constraint for root
let root_valid = match max_neighbor_idx {
    Some(max_idx) => root_dataset_idx < max_idx,
    None => true,
};

if root_valid {
    let root_dist = metric.distance_idx(q_dataset_idx, root_dataset_idx);
    result_heap.push(Neighbor {
        dist: OrderedFloat(root_dist),
        node_idx: 0,
    });
    kth_dist = if k > 0 { root_dist } else { T::max_value() };
    survivors_count += 1;
}
```

#### 2c. Filter at child iteration (lines 776-813)

```rust
while child != -1 {
    let child_idx = child as usize;
    let ds_idx = node_to_dataset[child_idx] as usize;

    // NEW: Predecessor constraint - skip if ds_idx >= max_neighbor_idx
    if let Some(max_idx) = max_neighbor_idx {
        if ds_idx >= max_idx {
            // Still need to traverse to next sibling
            let next = tree.next_node[child_idx];
            if next == child || next == tree.children[parent] {
                break;
            }
            child = next;
            continue;
        }
    }

    // Visited check (existing code)
    if use_visited {
        // ... existing logic
    }
    // ...
}
```

#### 2d. Add telemetry counter

**File:** `src/telemetry.rs`

```rust
#[derive(Default, Debug, Clone)]
pub struct ResidualQueryTelemetry {
    // ... existing fields ...
    pub filtered_predecessor: usize,  // NEW: count of nodes filtered by constraint
}
```

#### 2e. Update `batch_residual_knn_query`

```rust
pub fn batch_residual_knn_query<'a, T>(
    tree: &'a CoverTreeData<T>,
    query_indices: ndarray::ArrayView1<i64>,
    node_to_dataset: &[i64],
    metric: &ResidualMetric<'a, T>,
    k: usize,
    scope_caps: Option<&HashMap<i32, T>>,
    predecessor_mode: bool,  // NEW
    telemetry: Option<&mut ResidualQueryTelemetry>,
) -> (Vec<Vec<i64>>, Vec<Vec<T>>)
{
    // ...
    for &q_idx in query_indices.iter() {
        let max_neighbor = if predecessor_mode {
            Some(q_idx as usize)
        } else {
            None
        };

        let res = single_residual_knn_query(
            tree,
            node_to_dataset,
            metric,
            q_idx as usize,
            k,
            scope_caps,
            max_neighbor,  // NEW parameter
            // ...
        );
        // ...
    }
}
```

### Phase 3: PyO3 Bindings

**File:** `src/lib.rs`

Update `knn_query_residual` method:

```rust
#[pyo3(signature = (
    query_indices,
    node_to_dataset,
    v_matrix,
    p_diag,
    coords,
    rbf_var,
    rbf_ls,
    k,
    kernel_type=None,
    predecessor_mode=None  // NEW
))]
fn knn_query_residual<'py>(
    &mut self,
    py: Python<'py>,
    query_indices: numpy::PyReadonlyArray1<i64>,
    node_to_dataset: Vec<i64>,
    v_matrix: PyObject,
    p_diag: PyObject,
    coords: PyObject,
    rbf_var: f64,
    rbf_ls: PyObject,
    k: usize,
    kernel_type: Option<i32>,
    predecessor_mode: Option<bool>,  // NEW
) -> PyResult<(Bound<'py, numpy::PyArray2<i64>>, PyObject)> {
    let pred_mode = predecessor_mode.unwrap_or(false);
    // ... pass pred_mode to batch_residual_knn_query
}
```

### Phase 4: Python Wiring

**File:** `covertreex/queries/knn.py`

Update `_rust_knn_query()`:

```python
def _rust_knn_query(
    tree: PCCTree,
    batch: Any,
    *,
    k: int,
    return_distances: bool,
    predecessor_mode: bool = False,  # NEW
    backend: TreeBackend,
    context: cx_config.RuntimeContext,
    op_log: Any | None = None,
) -> Any:
    # ...
    if is_residual:
        indices, dists = wrapper.knn_query_residual(
            query_indices,
            node_to_dataset,
            v_matrix,
            p_diag,
            coords,
            rbf_var,
            rbf_ls,
            k,
            predecessor_mode=predecessor_mode  # NEW
        )
```

**File:** `covertreex/api/pcct.py`

```python
def knn(
    self,
    query_points: Any,
    *,
    k: int,
    return_distances: bool = False,
    predecessor_mode: bool = False,  # NEW
) -> Any:
    """Find k nearest neighbors for query points.

    Parameters
    ----------
    predecessor_mode : bool, default False
        If True, for query at index i, only return neighbors with index j < i.
        This is required for Vecchia GP approximations.
    """
    # ... pass through to knn_query
```

## Testing Strategy

### Unit Tests

```python
def test_predecessor_mode_basic():
    """Each query i should only have neighbors j < i."""
    points = np.random.randn(100, 2)
    tree = CoverTree().fit(points)

    indices = np.arange(100).reshape(-1, 1)
    neighbors = tree.knn(indices, k=10, predecessor_mode=True)

    for i in range(100):
        valid = neighbors[i][neighbors[i] >= 0]
        assert all(j < i for j in valid), f"Query {i} has invalid neighbor >= {i}"


def test_predecessor_mode_early_queries():
    """Early queries should have fewer neighbors."""
    points = np.random.randn(100, 2)
    tree = CoverTree().fit(points)

    indices = np.arange(100).reshape(-1, 1)
    neighbors = tree.knn(indices, k=10, predecessor_mode=True)

    # Query 0 has no predecessors
    assert np.all(neighbors[0] == -1)

    # Query 1 can only have neighbor 0
    valid_1 = neighbors[1][neighbors[1] >= 0]
    assert len(valid_1) == 1 and valid_1[0] == 0


def test_predecessor_mode_residual_metric():
    """Test with residual correlation metric."""
    # Setup residual metric...
    neighbors = tree.knn(query_indices, k=10, predecessor_mode=True)
    # Verify constraint holds
```

### Integration Tests

```python
def test_survi_vecchia_integration():
    """End-to-end test with survi's Vecchia GP."""
    # Build tree with residual metric
    # Query with predecessor_mode=True
    # Verify no post-filtering needed
    # Verify correctness against reference implementation
```

## Performance Considerations

1. **Early pruning**: Filtering at child iteration (not just at result insertion) allows pruning entire subtrees when all descendants violate the constraint.

2. **Branch skipping**: For trees built with Hilbert ordering, points with similar indices tend to cluster spatially. This could enable aggressive branch pruning.

3. **Reduced heap operations**: With fewer valid candidates, the result heap sees less churn.

4. **Query 0 fast-path**: When `predecessor_mode=True` and query index is 0, return immediately with empty results.

## Backward Compatibility

- Default `predecessor_mode=False` preserves existing behavior
- No changes to tree construction or serialization
- Existing code continues to work unchanged

## Timeline

| Phase | Scope | Effort |
|-------|-------|--------|
| 1 | Python fallback | 1 hour |
| 2 | Rust core | 2-3 hours |
| 3 | PyO3 bindings | 30 min |
| 4 | Python wiring | 30 min |
| Tests | All layers | 1-2 hours |

**Total estimated effort:** 5-7 hours

## Open Questions

1. **Should we support arbitrary index masks?** Instead of just `j < i`, allow `j in allowed_set[i]`? This would be more general but slower.

2. **Padding strategy**: When fewer than k predecessors exist, should we:
   - Pad with -1 (current plan)
   - Return variable-length results
   - Raise an error if k > i

3. **Telemetry**: What metrics are useful for debugging predecessor-constrained queries?

## References

- Vecchia approximation: Vecchia (1988), "Estimation and model identification for continuous spatial processes"
- survi implementation: `survi/models/selectors/cover_tree.py`
- Cover tree: Beygelzimer et al. (2006), "Cover trees for nearest neighbor"
