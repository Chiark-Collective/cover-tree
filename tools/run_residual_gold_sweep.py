#!/usr/bin/env python
from __future__ import annotations

import json
import os
import statistics
import subprocess
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = REPO_ROOT / "artifacts" / "benchmarks" / "residual_gold_sweep"
LOG_ROOT.mkdir(parents=True, exist_ok=True)

DIMS = [2, 3]
TREE_POINTS = [8_192, 16_384, 32_768]
KS = [15, 25, 50]
RUNS = 7

BASE_ENV = os.environ.copy()
BASE_ENV.update(
    {
        "COVERTREEX_ENABLE_NUMBA": "1",
        "COVERTREEX_SCOPE_CHUNK_TARGET": "0",
        "COVERTREEX_ENABLE_SPARSE_TRAVERSAL": "0",
        "COVERTREEX_BATCH_ORDER": "natural",
        "COVERTREEX_PREFIX_SCHEDULE": "doubling",
        "COVERTREEX_ENABLE_DIAGNOSTICS": "0",
    }
)

FIELDS = [
    "traversal_ms",
    "traversal_pairwise_ms",
    "traversal_semisort_ms",
    "traversal_kernel_provider_ms",
    "conflict_graph_ms",
    "conflict_adjacency_ms",
    "conflict_annulus_ms",
    "conflict_scope_group_ms",
    "conflict_pairwise_ms",
    "mis_ms",
]


def _summarise_values(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"sum": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "sum": float(sum(values)),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "p90": float(statistics.quantiles(values, n=10)[8]),
        "max": float(max(values)),
    }


def summarise_log(path: Path) -> Dict[str, Dict[str, float]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    build_totals = []
    stats: Dict[str, Dict[str, float]] = {}
    for field in FIELDS:
        values = [float(rec.get(field, 0.0) or 0.0) for rec in records]
        stats[field] = _summarise_values(values)
    for rec in records:
        total = (
            float(rec.get("traversal_ms", 0.0) or 0.0)
            + float(rec.get("conflict_graph_ms", 0.0) or 0.0)
            + float(rec.get("mis_ms", 0.0) or 0.0)
        )
        build_totals.append(total)
    stats["build_total_ms"] = _summarise_values(build_totals)
    return stats


def run_configuration(dimension: int, tree_points: int, k: int) -> Dict[str, object]:
    best_total = None
    best_stats: Dict[str, Dict[str, float]] | None = None
    best_log: Path | None = None
    for run_idx in range(1, RUNS + 1):
        log_path = LOG_ROOT / f"gold_dim{dimension}_k{k}_n{tree_points}_run{run_idx}.jsonl"
        if log_path.exists():
            log_path.unlink()
        cmd = [
            "python",
            "-m",
            "cli.pcct",
            "query",
            "--metric",
            "residual",
            "--dimension",
            str(dimension),
            "--tree-points",
            str(tree_points),
            "--batch-size",
            "512",
            "--queries",
            "1024",
            "--k",
            str(k),
            "--seed",
            str(42 + run_idx - 1),
            "--baseline",
            "gpboost",
            "--log-file",
            str(log_path),
        ]
        print(f"[sweep] dim={dimension} n={tree_points} k={k} run={run_idx}")
        subprocess.run(cmd, cwd=REPO_ROOT, env=BASE_ENV, check=True)
        stats = summarise_log(log_path)
        total = stats["build_total_ms"]["sum"]
        if best_total is None or total < best_total:
            best_total = total
            best_stats = stats
            best_log = log_path
    assert best_stats is not None and best_log is not None
    return {
        "dimension": dimension,
        "tree_points": tree_points,
        "k": k,
        "best_build_ms": best_total,
        "best_log": str(best_log.relative_to(REPO_ROOT)),
        "stats": best_stats,
    }


def main() -> None:
    results: List[Dict[str, object]] = []
    for dimension in DIMS:
        for tree_points in TREE_POINTS:
            for k in KS:
                results.append(run_configuration(dimension, tree_points, k))
    summary_path = LOG_ROOT / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"[sweep] summary written to {summary_path}")


if __name__ == "__main__":
    main()
