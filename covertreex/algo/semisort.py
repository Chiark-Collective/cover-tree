from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from covertreex.core.tree import TreeBackend


@dataclass(frozen=True)
class GroupByResult:
    keys: Any
    indptr: Any
    values: Any


def group_by_int(
    keys: Any,
    values: Any,
    *,
    backend: TreeBackend,
    stable: bool = True,
) -> GroupByResult:
    """Group `values` by integer `keys` and return CSR-style buffers.

    Parameters
    ----------
    keys:
        1-D array of integer keys.
    values:
        Array whose first dimension matches `keys`.
    backend:
        Tree backend providing array helpers.
    stable:
        Whether to preserve the relative order of equal keys (default True).
    """

    keys_np = np.asarray(backend.to_numpy(keys), dtype=np.int64)
    values_np = np.asarray(backend.to_numpy(values))

    if keys_np.ndim != 1:
        raise ValueError("group_by_int expects 1-D `keys`.")
    if values_np.shape[0] != keys_np.shape[0]:
        raise ValueError("`values` must align with `keys` in the first dimension.")

    if keys_np.size == 0:
        empty = backend.asarray([], dtype=backend.default_int)
        return GroupByResult(keys=empty, indptr=backend.asarray([0], dtype=backend.default_int), values=backend.asarray(values_np))

    order = np.argsort(keys_np, kind="stable" if stable else "quicksort")
    sorted_keys = keys_np[order]
    sorted_values = values_np[order]

    unique_keys, counts = np.unique(sorted_keys, return_counts=True)
    indptr = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])

    return GroupByResult(
        keys=backend.asarray(unique_keys, dtype=backend.default_int),
        indptr=backend.asarray(indptr, dtype=backend.default_int),
        values=backend.asarray(sorted_values),
    )

