# Covertreex

A high-performance parallel compressed cover tree (PCCT) library optimized for Vecchia-style Gaussian process pipelines.

[![PyPI version](https://badge.fury.io/py/covertreex.svg)](https://pypi.org/project/covertreex/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Features

- **Hybrid Python/Numba + Rust implementation** for maximum performance
- **~170x faster** than GPBoost for residual correlation k-NN queries
- **AVX2 SIMD optimized** dot products in the Rust backend
- **Hilbert curve ordering** for cache-efficient tree construction
- **Residual correlation metric** with RBF and Matérn 5/2 kernels
- **Batch insert** with parallel MIS-based conflict resolution

## Performance

Benchmark on AMD Ryzen 9 9950X (N=32k points, D=3, k=50 neighbors):

| Engine | Build Time | Query Throughput | vs GPBoost |
|--------|------------|------------------|------------|
| python-numba | 7.2s | 42,000 q/s | 154x faster |
| **rust-hilbert** | **0.85s** | **47,000 q/s** | **170x faster** |

## Installation

```bash
pip install covertreex
```

For development with Rust backend:
```bash
pip install -e ".[dev]"
maturin develop --release
```

## Quick Start

```python
from covertreex.api import PCCT, Runtime

# Configure runtime
runtime = Runtime(metric="euclidean", enable_numba=True)

# Build tree and query
with runtime.activate() as ctx:
    tree = PCCT(runtime).fit(points)
    indices, distances = tree.knn(queries, k=10, return_distances=True)
```

## Residual Correlation Metric

For Gaussian process applications with Vecchia approximations:

```python
from covertreex.api import PCCT, Runtime, Residual

# Configure residual metric
residual = Residual(
    v_matrix=V,           # Inducing point matrix
    p_diag=p_diag,        # Diagonal of precision matrix
    coords=coordinates,   # Spatial coordinates
    kernel_type=0,        # 0=RBF, 1=Matérn 5/2
)

runtime = Runtime(metric="residual", residual=residual)
with runtime.activate() as ctx:
    tree = PCCT(runtime).fit(points)
    neighbors, distances = tree.knn(queries, k=50, return_distances=True)
```

## CLI Usage

```bash
# List available profiles
python -m cli.pcct profile list

# Run k-NN benchmark
python -m cli.pcct query --dimension 8 --tree-points 8192 --queries 512 --k 8

# Run residual benchmark with Rust engine
python -m cli.pcct query --metric residual --engine rust-hilbert \
    --tree-points 32768 --dimension 3 --queries 1024 --k 50

# Environment/dependency check
python -m cli.pcct doctor --profile default
```

## Execution Engines

| Engine | Description | Use Case |
|--------|-------------|----------|
| `python-numba` | Reference implementation with full telemetry | Debugging, validation |
| `rust-natural` | Rust backend, natural point order | General use |
| `rust-hilbert` | Rust backend, Hilbert curve order | **Production (fastest)** |

The Rust backend is enabled by default. Disable with `COVERTREEX_ENABLE_RUST=0`.

## Benchmark Suite

```bash
# Gold standard residual benchmark (N=32k, D=3)
./benchmarks/run_residual_gold_standard.sh

# Comprehensive Rust vs Python comparison
python benchmarks/comprehensive_residual_benchmark.py

# CI reference benchmarks
python tools/run_reference_benchmarks.py
```

## Documentation

- [CLAUDE.md](CLAUDE.md) — Development guide and commands
- [CHANGELOG.md](CHANGELOG.md) — Release history
- [BENCHMARKS.md](BENCHMARKS.md) — Performance results
- [docs/](docs/) — API reference, architecture notes, and examples

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.
