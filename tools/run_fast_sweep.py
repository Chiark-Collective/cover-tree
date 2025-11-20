#!/usr/bin/env python3
import subprocess
import sys
import re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class BenchmarkResult:
    implementation: str
    n: int
    d: int
    build_time: float
    throughput: float

def run_benchmark(n: int, d: int, baseline: str) -> List[BenchmarkResult]:
    print(f"Running N={n} D={d} Baseline={baseline}...")
    cmd = [
        "python", "-m", "cli.pcct", "query",
        "--metric", "euclidean",
        "--tree-points", str(n),
        "--dimension", str(d),
        "--baseline", baseline,
        "--no-log-file"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout + result.stderr
        return parse_output(output, n, d, baseline)
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark: {e}")
        print(e.stdout)
        print(e.stderr)
        return []

def parse_output(output: str, n: int, d: int, baseline_name: str) -> List[BenchmarkResult]:
    results = []
    
    # Parse PCCT
    # pcct | build=1.3748s queries=1024 k=8 time=0.0208s latency=0.0203ms throughput=49,294.6 q/s
    pcct_match = re.search(r"pcct | build=([\d\.]+)s .* throughput=([\d,]+) q/s", output)
    if pcct_match:
        results.append(BenchmarkResult(
            implementation="PCCT",
            n=n, d=d,
            build_time=float(pcct_match.group(1)),
            throughput=float(pcct_match.group(2).replace(",", ""))
        ))
        
    # Parse Baseline
    # baseline[scipy_ckdtree] | build=0.0061s time=0.3063s latency=0.2991ms throughput=3,343.1 q/s
    baseline_match = re.search(r"baseline\[(.*?)\] | build=([\d\.]+)s .* throughput=([\d,]+) q/s", output)
    if baseline_match:
        results.append(BenchmarkResult(
            implementation=baseline_match.group(1),
            n=n, d=d,
            build_time=float(baseline_match.group(2)),
            throughput=float(baseline_match.group(3).replace(",", ""))
        ))
        
    return results

def main():
    # 1. Low-D Scale Test (D=8)
    # We skip 10k as 50k is more representative for "at scale"
    results = []
    
    # 50k D=8
    results.extend(run_benchmark(50000, 8, "sklearn"))
    results.extend(run_benchmark(50000, 8, "scipy"))
    
    # 2. High-D Comparison (D=64)
    # 10k D=64 (PCCT wins here)
    results.extend(run_benchmark(10000, 64, "sklearn"))
    results.extend(run_benchmark(10000, 64, "scipy"))
    
    # Print Table
    print("\n\n# Benchmark Results\n")
    print("| N | D | Implementation | Build (s) | Throughput (q/s) |")
    print("|---|---|---|---|---|")
    
    # Deduplicate PCCT entries if multiple baselines run for same N/D
    seen = set()
    
    for r in results:
        key = (r.implementation, r.n, r.d)
        # Hack: PCCT runs every time, so we only print the one from the first run of that N/D pair?
        # Or just print all to show variance? Let's print all but sort.
        print(f"| {r.n} | {r.d} | {r.implementation} | {r.build_time:.4f} | {r.throughput:,.1f} |")

if __name__ == "__main__":
    main()
