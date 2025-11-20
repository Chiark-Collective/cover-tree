#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Dict, List

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_sweep.py <manifest_path>")
        sys.exit(1)

    manifest_path = Path(sys.argv[1])
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    with manifest_path.open() as f:
        data = json.load(f)

    jobs = data.get("jobs", [])
    
    # Group by (N, D, Metric)
    # We want to compare PCCT vs Baselines for Euclidean, and just show PCCT for Residual
    
    print(f"## Benchmark Results: {manifest_path.parent.name}")
    print("")
    
    # Parse results from JSONL logs
    results = []

    for job in jobs:
        name = job["name"]
        n = job["tree_points"]
        d = job["dimension"]
        metric = job["metric"]
        log_path = job.get("log")
        
        if not log_path or job.get("skipped"):
            continue

        # We need to read the JSONL to get the actual numbers
        # The log file has multiple lines.
        # For PCCT run: we look for "benchmark_summary" event at the end.
        # For Baselines: we look for "baseline_comparison" events.
        
        pcct_build = None
        pcct_qps = None
        baselines = {} # name -> {build, qps}

        try:
            with open(log_path, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        event = entry.get("event")
                        
                        if event == "benchmark_summary" and entry.get("benchmark") == "pcct.query":
                            # This is PCCT
                            pcct_build = entry.get("build_time_sec")
                            pcct_qps = entry.get("queries_per_second")
                        
                        if event == "baseline_comparison":
                            b_name = entry.get("name")
                            b_build = entry.get("build_time_sec")
                            b_qps = entry.get("queries_per_second")
                            baselines[b_name] = {"build": b_build, "qps": b_qps}
                            
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading log {log_path}: {e}")
            continue

        # Add PCCT entry
        if pcct_build is not None:
            results.append({
                "impl": "PCCT",
                "metric": metric,
                "n": n,
                "d": d,
                "build_s": pcct_build,
                "qps": pcct_qps
            })
            
        # Add Baseline entries
        for b_name, stats in baselines.items():
             results.append({
                "impl": b_name,
                "metric": metric,
                "n": n,
                "d": d,
                "build_s": stats["build"],
                "qps": stats["qps"]
            })

    # Print Table
    # Columns: Metric, N, D, Impl, Build (s), QPS, Speedup (vs MLPack for Euclid)
    
    print("| Metric | N | D | Implementation | Build (s) | Throughput (QPS) | vs MLPack (QPS) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    # Sort by Metric, N, D, QPS (desc)
    results.sort(key=lambda x: (x["metric"], x["n"], x["d"], -x["qps"]))
    
    for r in results:
        # Find MLPack QPS for this group to calculate speedup
        mlpack_qps = next((x["qps"] for x in results if x["metric"] == r["metric"] and x["n"] == r["n"] and x["d"] == r["d"] and x["impl"] == "mlpack"), None)
        
        speedup = "-"
        if mlpack_qps and mlpack_qps > 0 and r["qps"] > 0:
            ratio = r["qps"] / mlpack_qps
            speedup = f"{ratio:.2f}x"
            
        print(f"| {r['metric']} | {r['n']} | {r['d']} | **{r['impl']}** | {r['build_s']:.4f} | {r['qps']:,.1f} | {speedup} |")

if __name__ == "__main__":
    main()
