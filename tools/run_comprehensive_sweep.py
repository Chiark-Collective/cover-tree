#!/usr/bin/env python3
"""Automate a comprehensive benchmark sweep comparing PCCT against all available baselines.

Sweeps over:
- Scale (N): 10k, 50k, 100k, (500k, 1M excluded for speed in this default run)
- Dimension (D): 3 (Bio/Physics), 64 (Embeddings)
- Metrics: Euclidean (all baselines), Residual (PCCT only)

Baselines:
- mlpack (C++)
- scikit-learn (BallTree)
- scipy (cKDTree)
- covertree (Python/Cython)
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
DEFAULT_OUTPUT_BASE = REPO_ROOT / "artifacts" / "benchmarks" / "sweep"
DEFAULT_ENV = {
    "COVERTREEX_BACKEND": "numpy",
    "COVERTREEX_ENABLE_NUMBA": "1",
    # Ensure baselines are visible
    "COVERTREEX_ENABLE_DIAGNOSTICS": "0", 
}


@dataclass(frozen=True)
class BenchmarkJob:
    name: str
    metric: str = "euclidean"
    tree_points: int = 10000
    dimension: int = 8
    queries: int = 1000
    batch_size: int = 512
    k: int = 10
    baseline: str = "none"
    description: str = ""
    env: Dict[str, str] = field(default_factory=dict)

    def build_command(self, log_path: Path, python_executable: str) -> List[str]:
        cmd = [
            python_executable,
            "-m",
            "cli.pcct",
            "query",
            "--metric",
            self.metric,
            "--dimension",
            str(self.dimension),
            "--tree-points",
            str(self.tree_points),
            "--batch-size",
            str(self.batch_size),
            "--queries",
            str(self.queries),
            "--k",
            str(self.k),
            "--seed",
            "42",
            "--baseline",
            self.baseline,
            "--log-file",
            str(log_path),
        ]
        return cmd

    def metadata(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "metric": self.metric,
            "tree_points": self.tree_points,
            "dimension": self.dimension,
            "queries": self.queries,
            "batch_size": self.batch_size,
            "k": self.k,
            "baseline": self.baseline,
            "env": self.env,
            "description": self.description,
        }


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return DEFAULT_OUTPUT_BASE / timestamp


def _generate_jobs() -> Dict[str, BenchmarkJob]:
    jobs = {}
    
    scales = [10_000, 50_000, 100_000]
    dimensions = [3, 64]
    
    # 1. Euclidean Comparisons (PCCT vs Baselines)
    # We run "all" baselines in one go per (N, D) tuple to save overhead of multiple python startups
    # providing the CLI supports running multiple baselines. 
    # Currently cli.pcct query --baseline takes a string. 
    # 'all' triggers run_baseline_comparisons with mode='all'.
    # This includes: sequential, gpboost, mlpack, external (covertree), sklearn, scipy.
    
    for n in scales:
        for d in dimensions:
            # PCCT Euclidean Sparse (Default if Numba) vs Fast Baselines
            # Excludes "external" (PyPI covertree) as it is too slow.
            job_name = f"sweep_euclidean_n{n}_d{d}"
            jobs[job_name] = BenchmarkJob(
                name=job_name,
                metric="euclidean",
                tree_points=n,
                dimension=d,
                baseline="mlpack,sklearn,scipy",
                description=f"Euclidean N={n} D={d} PCCT vs Fast Baselines",
                # Force sparse traversal for PCCT just in case (though it's default now)
                env={"COVERTREEX_ENABLE_SPARSE_TRAVERSAL": "1"}
            )

    # 2. Residual PCCT Scaling (No external baselines possible)
    for n in scales:
        for d in dimensions:
            job_name = f"sweep_residual_n{n}_d{d}"
            jobs[job_name] = BenchmarkJob(
                name=job_name,
                metric="residual",
                tree_points=n,
                dimension=d,
                baseline="none",
                description=f"Residual N={n} D={d} PCCT Scaling",
            )

    return jobs


def _export_csv(log_path: Path, csv_path: Path, python_executable: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_executable,
        str(TOOLS_DIR / "export_benchmark_diagnostics.py"),
        "--output",
        str(csv_path),
        str(log_path),
    ]
    subprocess.run(cmd, check=True)


def _run_job(
    job: BenchmarkJob,
    log_path: Path,
    csv_path: Path,
    env: Dict[str, str],
    python_executable: str,
    skip_existing: bool,
) -> Dict[str, str]:
    if skip_existing and log_path.exists():
        print(f"[skip] {job.name} (log exists)")
        return {"log": str(log_path), "csv": str(csv_path), "skipped": True}
    
    cmd = job.build_command(log_path, python_executable)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {job.name}...")
    
    env_vars = os.environ.copy()
    env_vars.update(DEFAULT_ENV)
    env_vars.update(job.env)
    env_vars.update(env)
    
    try:
        subprocess.run(cmd, check=True, env=env_vars, cwd=str(REPO_ROOT))
        _export_csv(log_path, csv_path, python_executable)
        return {"log": str(log_path), "csv": str(csv_path), "skipped": False}
    except subprocess.CalledProcessError as e:
        print(f"[error] Job {job.name} failed: {e}")
        return {"log": str(log_path), "error": str(e), "skipped": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--jobs", type=str, help="Filter jobs (comma separated)")
    parser.add_argument("--list-jobs", action="store_true", help="List jobs")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing logs")
    args = parser.parse_args(argv)

    jobs = _generate_jobs()
    
    if args.list_jobs:
        for name in jobs:
            print(name)
        return 0

    output_dir = (args.output_dir or _default_output_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    selected_jobs = jobs.values()
    if args.jobs:
        filter_names = set(args.jobs.split(","))
        selected_jobs = [j for j in selected_jobs if j.name in filter_names]

    print(f"Running {len(selected_jobs)} benchmark jobs...")
    print(f"Output directory: {output_dir}")

    python_exe = sys.executable
    results = []
    
    for job in selected_jobs:
        log_path = output_dir / f"{job.name}.jsonl"
        csv_path = output_dir / f"{job.name}.csv"
        res = _run_job(job, log_path, csv_path, {}, python_exe, args.skip_existing)
        meta = job.metadata()
        meta.update(res)
        results.append(meta)

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump({"jobs": results}, f, indent=2)
        
    print(f"Sweep complete. Manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
