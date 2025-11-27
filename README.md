# Covertreex

A parallel compressed cover tree (PCCT) library optimized for Vecchia-style Gaussian process pipelines. Features a hybrid Python/Numba + Rust implementation for high-performance CPU workloads.

[![PyPI version](https://badge.fury.io/py/covertreex.svg)](https://pypi.org/project/covertreex/)

## Installation

```bash
pip install covertreex
```

For development:
```bash
pip install -e ".[dev]"
maturin develop --release  # Build Rust backend
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

## CLI Usage

```bash
# List available profiles
python -m cli.pcct profile list

# Run k-NN benchmark
python -m cli.pcct query --dimension 8 --tree-points 8192 --queries 512 --k 8

# Build tree only
python -m cli.pcct build --dimension 8 --tree-points 65536 --batch-size 1024

# Environment check
python -m cli.pcct doctor --profile default
```

## Execution Engines

Three engines are available via `--engine`:

| Engine | Description |
|--------|-------------|
| `python-numba` | Reference Python/Numba implementation with full telemetry |
| `rust-natural` | Rust backend with natural point ordering |
| `rust-hilbert` | Rust backend with Hilbert curve reordering (fastest builds) |

The Rust backend is enabled by default. Disable with `COVERTREEX_ENABLE_RUST=0`.

## Residual Correlation Metric

For Gaussian process applications, the library supports residual correlation with configurable kernels:

```bash
python -m cli.pcct query \
    --metric residual \
    --residual-kernel-type 1 \  # 0=RBF, 1=Matern52
    --engine rust-hilbert
```

## Benchmark Suite

```bash
# Gold standard benchmark (N=32k, D=3, residual)
./benchmarks/run_residual_gold_standard.sh

# Rust vs Python comparison
python benchmarks/comprehensive_residual_benchmark.py

# Reference benchmarks for CI
python tools/run_reference_benchmarks.py
```

## Documentation

- `CLAUDE.md` — Development guide and commands
- `CHANGELOG.md` — Release notes
- `docs/` — API reference and examples

## License

MIT
