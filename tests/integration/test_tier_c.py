import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from covertreex.algo import batch_insert
from covertreex.core.tree import PCCTree, TreeLogStats, get_runtime_backend
from covertreex.queries import knn


def _backend_array(backend, values):
    return backend.asarray(values, dtype=backend.default_float)


def _base_tree() -> PCCTree:
    backend = get_runtime_backend()
    points = _backend_array(
        backend,
        np.asarray(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [3.0, 3.0],
            ],
            dtype=np.float64,
        ),
    )
    top_levels = backend.asarray([2, 1, 0], dtype=backend.default_int)
    parents = backend.asarray([-1, 0, 1], dtype=backend.default_int)
    children = backend.asarray([1, 2, -1], dtype=backend.default_int)
    level_offsets = backend.asarray([0, 1, 2, 3], dtype=backend.default_int)
    si_cache = backend.asarray([4.0, 2.0, 1.0], dtype=backend.default_float)
    next_cache = backend.asarray([1, 2, -1], dtype=backend.default_int)
    return PCCTree(
        points=points,
        top_levels=top_levels,
        parents=parents,
        children=children,
        level_offsets=level_offsets,
        si_cache=si_cache,
        next_cache=next_cache,
        stats=TreeLogStats(num_batches=1),
        backend=backend,
    )


def test_async_batch_insert_harness():
    backend = get_runtime_backend()
    tree = _base_tree()
    update_batch = _backend_array(
        backend,
        np.asarray([[0.2, 0.1], [2.9, 2.8]], dtype=np.float64),
    )
    query = _backend_array(
        backend,
        np.asarray([[0.15, 0.05]], dtype=np.float64),
    )
    baseline_idx = np.asarray(knn(tree, query, k=1)).tolist()

    def _perform_update() -> PCCTree:
        time.sleep(0.05)
        new_tree, _ = batch_insert(tree, update_batch, mis_seed=0)
        return new_tree

    observed: list[list[int]] = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_perform_update)
        while not future.done():
            current = np.asarray(knn(tree, query, k=1)).tolist()
            observed.append(current)
            time.sleep(0.005)
        updated_tree = future.result()

    assert observed, "expected to capture at least one in-flight query"
    assert all(sample == baseline_idx for sample in observed)

    post_idx = np.asarray(knn(updated_tree, query, k=1)).tolist()
    assert post_idx != baseline_idx
    assert np.asarray(knn(tree, query, k=1)).tolist() == baseline_idx
