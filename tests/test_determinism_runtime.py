from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from covertreex.algo import batch_insert
from covertreex.api import Runtime
from covertreex.core.tree import PCCTree
from covertreex.telemetry import BenchmarkLogWriter


def _gaussian_points(*, dimension: int, tree_points: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal(size=(tree_points, dimension)).astype(np.float64)


def _tree_snapshot(tree) -> dict[str, np.ndarray]:
    backend = tree.backend
    return {
        "points": np.asarray(backend.to_numpy(tree.points), dtype=np.float64),
        "parents": np.asarray(backend.to_numpy(tree.parents), dtype=np.int32),
        "children": np.asarray(backend.to_numpy(tree.children), dtype=np.int32),
        "levels": np.asarray(backend.to_numpy(tree.top_levels), dtype=np.int32),
    }


def _run_once(tmp_path: Path, *, run_id: str, points: np.ndarray):
    runtime = Runtime(
        backend="numpy",
        precision="float64",
        enable_numba=False,
        enable_sparse_traversal=False,
        metric="euclidean",
        global_seed=7,
    )
    model_snapshot = runtime.to_model().model_dump()
    log_path = tmp_path / f"{run_id}.jsonl"
    with runtime.activate() as context:
        backend = context.get_backend()
        batch = backend.asarray(points, dtype=backend.default_float)
        tree = PCCTree.empty(dimension=int(points.shape[1]), backend=backend)
        writer = BenchmarkLogWriter(
            str(log_path),
            run_id=run_id,
            runtime=model_snapshot,
            metadata={"benchmark": "determinism-test"},
        )
        try:
            tree, plan = batch_insert(
                tree,
                batch,
                backend=backend,
                context=context,
            )
            writer.record_batch(
                batch_index=0,
                batch_size=int(points.shape[0]),
                plan=plan,
            )
        finally:
            writer.close()
    return tree, log_path


def test_repeated_runs_share_tree_and_hash(tmp_path):
    points = _gaussian_points(dimension=4, tree_points=16)
    first_tree, first_log = _run_once(tmp_path, run_id="det-1", points=points)
    second_tree, second_log = _run_once(tmp_path, run_id="det-2", points=points)

    first_snapshot = _tree_snapshot(first_tree)
    second_snapshot = _tree_snapshot(second_tree)
    for key in first_snapshot:
        np.testing.assert_array_equal(first_snapshot[key], second_snapshot[key])

    first_record = json.loads(first_log.read_text(encoding="utf-8").strip())
    second_record = json.loads(second_log.read_text(encoding="utf-8").strip())
    assert first_record["run_hash"] == second_record["run_hash"]
    assert first_record["runtime_digest"] == second_record["runtime_digest"]
    assert first_record["seed_pack"] == second_record["seed_pack"]

    for key in ("candidates", "selected", "dominated", "mis_iterations"):
        assert first_record[key] == second_record[key]
