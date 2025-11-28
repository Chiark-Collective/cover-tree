# Covertreex

High-performance cover tree library for k-NN queries, optimized for Vecchia-style Gaussian process pipelines.

[![PyPI version](https://badge.fury.io/py/covertreex.svg)](https://pypi.org/project/covertreex/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Features

- **Very fast**, parallel implementation following [Parallel Cover Trees and their Applications](https://cs.ucr.edu/~ygu/papers/SPAA22/covertree.pdf)
- **Hybrid Python/Numba + Rust** implementation
- **AVX2 SIMD optimized** dot products in Rust backend
- **Residual correlation metric** with RBF and Matérn 5/2 kernels
- **Hilbert curve ordering** for cache-efficient tree construction

## Installation

```bash
pip install covertreex
```

## Quick Start

### Basic Euclidean k-NN

```python
import numpy as np
from covertreex import cover_tree

# Build tree and query
coords = np.random.randn(10000, 3).astype(np.float32)
tree = cover_tree(coords)
neighbors = tree.knn(k=10)

# With distances
neighbors, distances = tree.knn(k=10, return_distances=True)
```

### Residual Correlation Metric (Vecchia GP)

For Gaussian process applications with Vecchia approximations:

```python
import numpy as np
from covertreex import cover_tree
from covertreex.kernels import Matern52

coords = np.random.randn(10000, 3).astype(np.float32)

# Option 1: Provide a kernel (V-matrix built internally)
tree = cover_tree(coords, kernel=Matern52(lengthscale=1.0, variance=1.0))
neighbors = tree.knn(k=50)

# Option 2: Provide pre-computed V-matrix (from your GP)
# tree = cover_tree(coords, v_matrix=V, p_diag=p_diag)

# Predecessor constraint (for Vecchia): neighbor j must have j < query i
neighbors = tree.knn(k=50, predecessor_mode=True)
```

### Engine Selection

```python
from covertreex import cover_tree
from covertreex.kernels import Matern52

# cover_tree defaults to rust-hilbert (fastest)
tree = cover_tree(coords, kernel=Matern52(lengthscale=1.0), engine="rust-hilbert")
tree = cover_tree(coords, kernel=Matern52(lengthscale=1.0), engine="rust-natural")
tree = cover_tree(coords, kernel=Matern52(lengthscale=1.0), engine="python-numba")
```

### Profile Presets

```python
from covertreex import Runtime

# Load predefined configuration
runtime = Runtime.from_profile("residual-gold")

# With overrides
runtime = Runtime.from_profile("residual-gold", overrides=["seeds.global_seed=42"])
```

Available profiles: `default`, `residual-gold`, `residual-fast`, `residual-audit`, `cpu-debug`

## Performance

Residual correlation k-NN benchmark (AMD Ryzen 9 9950X, N=32k, D=3, k=50):

| Engine | Build Time | Query Throughput |
|--------|------------|------------------|
| python-numba | 7.0s | 40,000 q/s |
| **rust-hilbert** | **0.9s** | **44,000 q/s** |

## API Reference

### cover_tree (recommended)

Factory function for building cover trees. Handles all configuration internally.

```python
from covertreex import cover_tree
from covertreex.kernels import Matern52, RBF

# Euclidean distance (default)
tree = cover_tree(coords)

# Residual correlation with kernel
tree = cover_tree(coords, kernel=Matern52(lengthscale=1.0, variance=1.0))
tree = cover_tree(coords, kernel=RBF(lengthscale=2.0))

# Residual correlation with pre-computed V-matrix
tree = cover_tree(coords, v_matrix=V, p_diag=p_diag, kernel_diag=k_diag)

# Query
neighbors = tree.knn(k=10)
neighbors = tree.knn(k=50, predecessor_mode=True)  # Vecchia constraint
neighbors, distances = tree.knn(k=10, return_distances=True)
```

### Kernel Classes

GP kernels for residual correlation metric:

```python
from covertreex.kernels import Matern52, RBF

# Matérn 5/2 kernel (recommended for GP)
kernel = Matern52(lengthscale=1.0, variance=1.0)

# RBF (squared exponential) kernel
kernel = RBF(lengthscale=2.0, variance=1.0)
```

### CoverTree (advanced)

Lower-level interface with explicit runtime configuration:

```python
from covertreex import CoverTree, Runtime

runtime = Runtime(engine="rust-hilbert", metric="euclidean")
tree = CoverTree(runtime).fit(points)
neighbors = tree.knn(query_points, k=10)
```

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Build Rust backend
maturin develop --release

# Run tests
pytest

# Lint
ruff check covertreex
```

## CLI (Testing)

A CLI is included for benchmarking and testing:

```bash
python -m cli.pcct --help
python -m cli.pcct query --engine rust-hilbert --tree-points 32768 --k 50
```

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.
