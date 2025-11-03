# Covertreex

Parallel compressed cover tree (PCCT) library engineered for Vecchia-style Gaussian process pipelines on GPUs. This is the implementation companion to the plan captured in `PARALLEL_COMPRESSED_PLAN.md`.

> Status: scaffolding in progress. Expect rapid iteration across backends, persistence utilities, and traversal/insertion kernels.

## Getting Started

Within a Python 3.12 environment:

```bash
pip install -e ".[dev]"
```

The default backend is `jax.numpy` (`jnp`). Optional acceleration hooks leverage `numba` when the `numba` extra is installed.

## Reference Material

- `PARALLEL_COMPRESSED_PLAN.md` &mdash; architecture, milestones, and testing ladder.
- `notes/` &mdash; upstream context and domain-specific constraints gathered during planning.
