from __future__ import annotations

import numpy as np

try:  # pragma: no cover - optional dependency
    from numba import njit  # type: ignore

    NUMBA_RESIDUAL_SCOPE_AVAILABLE = True
except Exception:  # pragma: no cover - when numba unavailable
    njit = None  # type: ignore
    NUMBA_RESIDUAL_SCOPE_AVAILABLE = False


if NUMBA_RESIDUAL_SCOPE_AVAILABLE:

    @njit(cache=True)
    def _append_positions_impl(
        flags: np.ndarray,
        positions: np.ndarray,
        buffer: np.ndarray,
        count: int,
        limit: int,
        respect_limit: bool,
    ) -> tuple[int, int, int]:
        dedupe = 0
        saturated = 0
        num_flags = flags.shape[0]
        capacity = buffer.shape[0]
        limit_enabled = respect_limit and limit > 0

        for idx in range(positions.shape[0]):
            pos = int(positions[idx])
            if pos < 0 or pos >= num_flags:
                continue
            if flags[pos] != 0:
                dedupe += 1
                continue
            flags[pos] = 1
            if count < capacity:
                buffer[count] = pos
            count += 1
            if capacity > 0 and count >= capacity:
                saturated = 1
                break
            if limit_enabled and count >= limit:
                saturated = 1
                break

        return count, dedupe, saturated

    @njit(cache=True)
    def _reset_flags_impl(flags: np.ndarray, buffer: np.ndarray, count: int) -> None:
        num_flags = flags.shape[0]
        total = buffer.shape[0]
        limit = count if count < total else total
        for idx in range(limit):
            pos = int(buffer[idx])
            if 0 <= pos < num_flags:
                flags[pos] = 0

else:  # pragma: no cover - executed when numba missing

    def _append_positions_impl(flags, positions, buffer, count, limit, respect_limit):
        dedupe = 0
        saturated = 0
        num_flags = flags.shape[0]
        capacity = buffer.shape[0]
        limit_enabled = respect_limit and limit > 0

        for pos in positions:
            pos_int = int(pos)
            if pos_int < 0 or pos_int >= num_flags:
                continue
            if flags[pos_int] != 0:
                dedupe += 1
                continue
            flags[pos_int] = 1
            if count < capacity:
                buffer[count] = pos_int
            count += 1
            if capacity > 0 and count >= capacity:
                saturated = 1
                break
            if limit_enabled and count >= limit:
                saturated = 1
                break

        return count, dedupe, saturated

    def _reset_flags_impl(flags, buffer, count):
        num_flags = flags.shape[0]
        total = buffer.shape[0]
        limit = count if count < total else total
        for idx in range(limit):
            pos = int(buffer[idx])
            if 0 <= pos < num_flags:
                flags[pos] = 0


def residual_scope_append(
    flags: np.ndarray,
    positions: np.ndarray,
    buffer: np.ndarray,
    count: int,
    limit: int,
    *,
    respect_limit: bool = True,
) -> tuple[int, int, bool]:
    """Append unique tree positions into the per-query buffer.

    Returns (new_count, dedupe_hits, hit_limit).
    """

    new_count, dedupe_hits, saturated = _append_positions_impl(
        flags,
        positions,
        buffer,
        int(count),
        int(limit),
        bool(respect_limit),
    )
    return int(new_count), int(dedupe_hits), bool(saturated)


def residual_scope_reset(flags: np.ndarray, buffer: np.ndarray, count: int) -> None:
    """Clear flag entries for the members stored in the buffer."""

    _reset_flags_impl(flags, buffer, int(count))


__all__ = [
    "NUMBA_RESIDUAL_SCOPE_AVAILABLE",
    "residual_scope_append",
    "residual_scope_reset",
]
