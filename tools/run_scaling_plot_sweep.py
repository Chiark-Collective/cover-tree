#!/usr/bin/env python3
import subprocess
import sys
import json
import csv
import re
from typing import List
from pathlib import Path
from datetime import datetime
import time

# Configurations
DIMENSIONS = [8, 64]
SCALES = [10_000, 50_000, 100_000, 200_000]
K_VALUES = [10, 50]
QUERIES = 2000
BASELINES = "sklearn,scipy"
OUTPUT_DIR = Path("artifacts/scaling_data")

def run_benchmark(n: int, d: int, k: int, log_path: Path) -> str:
    print(f"--> Benchmarking N={n} D={d} k={k}...")
    cmd = [
        "python", "-m", "cli.pcct", "query",
        "--metric", "euclidean",
        "--tree-points", str(n),
        "--dimension", str(d),
        "--queries", str(QUERIES),
        "--k", str(k),
        "--baseline", BASELINES,
        "--no-log-file" # We parse stdout now
    ]
    
    start = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.time() - start
        print(f"    Done in {elapsed:.2f}s")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"    FAILED! Output:\n{e.stdout}\nError:\n{e.stderr}")
        return ""

def parse_output(output: str, n: int, d: int, k: int) -> List[dict]:
    rows = []
    
    # Parse PCCT
    # pcct | build=1.3748s queries=1024 k=8 time=0.0208s latency=0.0203ms throughput=49,294.6 q/s
    pcct_match = re.search(r"pcct \| build=([\d\.]+)s .* time=([\d\.]+)s latency=([\d\.]+)ms throughput=([\d,\.]+) q/s", output)
    if pcct_match:
        rows.append({
            "n": n, "d": d, "k": k, "queries": QUERIES,
            "impl": "PCCT",
            "build_time_s": float(pcct_match.group(1)),
            "query_time_s": float(pcct_match.group(2)),
            "latency_ms": float(pcct_match.group(3)),
            "qps": float(pcct_match.group(4).replace(",", ""))
        })
        
    # Parse Baselines
    # baseline[scipy_ckdtree] | build=0.0061s time=0.3063s latency=0.2991ms throughput=3,343.1 q/s
    # Note: name might contain brackets or underscores
    baseline_matches = re.finditer(r"baseline\[(.*?)\] \| build=([\d\.]+)s time=([\d\.]+)s latency=([\d\.]+)ms throughput=([\d,\.]+) q/s", output)
    for match in baseline_matches:
        rows.append({
            "n": n, "d": d, "k": k, "queries": QUERIES,
            "impl": match.group(1),
            "build_time_s": float(match.group(2)),
            "query_time_s": float(match.group(3)),
            "latency_ms": float(match.group(4)),
            "qps": float(match.group(5).replace(",", ""))
        })
        
    return rows

def main():
    print("Starting benchmark sweep...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_rows = []
    
    for d in DIMENSIONS:
        for n in SCALES:
            for k in K_VALUES:
                log_file = OUTPUT_DIR / f"n{n}_d{d}_k{k}.jsonl"
                # Force run since we need stdout
                output = run_benchmark(n, d, k, log_file)
                if output:
                    rows = parse_output(output, n, d, k)
                    all_rows.extend(rows)
                
    csv_path = OUTPUT_DIR / f"scaling_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = ["n", "d", "k", "queries", "impl", "build_time_s", "query_time_s", "latency_ms", "qps"]
    
    # Sort rows
    all_rows.sort(key=lambda x: (x["d"], x["n"], x["k"], x["impl"]))
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
        
    print(f"\nWrote {len(all_rows)} results to {csv_path}")
    
    # Quick Peek
    print("\n--- Quick Peek (D=64, N=Max) ---")
    for row in all_rows:
        if row["d"] == 64 and row["n"] == max(SCALES) and row["k"] == 10:
             print(f"Impl: {row['impl']:<20} QPS: {row['qps']:,.1f}")

if __name__ == "__main__":
    main()

