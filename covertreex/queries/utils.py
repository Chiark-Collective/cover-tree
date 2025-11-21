from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from covertreex.core.tree import TreeBackend


def to_numpy_array(backend: TreeBackend, array: Any, dtype: Any) -> np.ndarray:
    """Materialise a backend array as a NumPy array with the desired dtype."""

    return np.asarray(backend.to_numpy(array), dtype=dtype)


class ChildChainCache:
    """Memoise decoded child chains from the compressed representation."""

    __slots__ = ("_children", "_next", "_cache", "_empty")

    def __init__(self, children: np.ndarray, next_cache: np.ndarray) -> None:
        self._children = children
        self._next = next_cache
        self._cache: Dict[int, np.ndarray] = {}
        self._empty = np.empty(0, dtype=np.int64)

    def get(self, parent: int) -> np.ndarray:
        if parent < 0 or parent >= self._children.shape[0]:
            return self._empty
        cached = self._cache.get(parent)
        if cached is not None:
            return cached

        head = int(self._children[parent])
        if head < 0 or head >= self._next.shape[0]:
            self._cache[parent] = self._empty
            return self._empty

        chain: List[int] = []
        seen: set[int] = set()
        current = head
        while 0 <= current < self._next.shape[0] and current not in seen:
            chain.append(current)
            seen.add(current)
            nxt = int(self._next[current])
            if nxt < 0 or nxt == current:
                break
            current = nxt

        arr = np.asarray(chain, dtype=np.int64) if chain else self._empty
        self._cache[parent] = arr
        return arr
