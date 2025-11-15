from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List

from covertreex.logging import get_logger

from .base import ConflictGraphContext, ConflictGraphStrategy
from .builders import (
    AdjacencyBuild,
    build_dense_adjacency,
    build_grid_adjacency,
    build_residual_adjacency,
    build_segmented_adjacency,
)


class _DenseConflictStrategy(ConflictGraphStrategy):
    def build(self, ctx: ConflictGraphContext) -> AdjacencyBuild:
        return build_dense_adjacency(
            backend=ctx.backend,
            batch_size=ctx.batch_size,
            scope_indptr=ctx.scope_indptr,
            scope_indices=ctx.scope_indices,
            pairwise=ctx.pairwise,
            radii=ctx.radii_np,
            chunk_target_override=ctx.chunk_target_override,
            runtime=ctx.runtime,
        )


class _SegmentedConflictStrategy(ConflictGraphStrategy):
    def build(self, ctx: ConflictGraphContext) -> AdjacencyBuild:
        return build_segmented_adjacency(
            backend=ctx.backend,
            scope_indices=ctx.scope_indices,
            point_ids=ctx.point_ids,
            pairwise_np=ctx.pairwise_np,
            radii_np=ctx.radii_np,
        )


class _GridConflictStrategy(ConflictGraphStrategy):
    def build(self, ctx: ConflictGraphContext) -> AdjacencyBuild:
        return build_grid_adjacency(
            backend=ctx.backend,
            batch_points=ctx.grid_points if ctx.grid_points is not None else ctx.batch,
            batch_levels=ctx.levels,
            radii=ctx.radii,
            scope_indptr=ctx.scope_indptr,
            scope_indices=ctx.scope_indices,
            runtime=ctx.runtime,
        )


class _ResidualGridConflictStrategy(ConflictGraphStrategy):
    def build(self, ctx: ConflictGraphContext) -> AdjacencyBuild:
        return build_grid_adjacency(
            backend=ctx.backend,
            batch_points=ctx.grid_points if ctx.grid_points is not None else ctx.batch,
            batch_levels=ctx.levels,
            radii=ctx.radii,
            scope_indptr=ctx.scope_indptr,
            scope_indices=ctx.scope_indices,
            runtime=ctx.runtime,
        )


class _ResidualConflictStrategy(ConflictGraphStrategy):
    def build(self, ctx: ConflictGraphContext) -> AdjacencyBuild:
        return build_residual_adjacency(
            backend=ctx.backend,
            batch_size=ctx.batch_size,
            scope_indptr=ctx.scope_indptr,
            scope_indices=ctx.scope_indices,
            pairwise=ctx.pairwise,
            radii=ctx.radii_np,
            residual_pairwise=ctx.residual_pairwise_np,
            chunk_target_override=ctx.chunk_target_override,
            runtime=ctx.runtime,
        )


@dataclass(frozen=True)
class _ConflictStrategySpec:
    name: str
    predicate: Callable[[Any, bool, bool], bool]
    factory: Callable[[], ConflictGraphStrategy]
    origin: str
    predicate_label: str


LOGGER = get_logger("algo.conflict.registry")


_CONFLICT_REGISTRY: list[_ConflictStrategySpec] = []


def register_conflict_strategy(
    name: str,
    *,
    predicate: Callable[[Any, bool, bool], bool],
    factory: Callable[[], ConflictGraphStrategy],
    origin: str | None = None,
) -> None:
    global _CONFLICT_REGISTRY
    origin_label = origin or getattr(factory, "__module__", "<unknown>")
    predicate_label = getattr(predicate, "__qualname__", repr(predicate))
    _CONFLICT_REGISTRY = [spec for spec in _CONFLICT_REGISTRY if spec.name != name]
    _CONFLICT_REGISTRY.append(
        _ConflictStrategySpec(
            name=name,
            predicate=predicate,
            factory=factory,
            origin=origin_label,
            predicate_label=predicate_label,
        )
    )
    LOGGER.debug("Registered conflict strategy: %%s", name)


def deregister_conflict_strategy(name: str) -> None:
    global _CONFLICT_REGISTRY
    before = len(_CONFLICT_REGISTRY)
    _CONFLICT_REGISTRY = [spec for spec in _CONFLICT_REGISTRY if spec.name != name]
    if before != len(_CONFLICT_REGISTRY):
        LOGGER.debug("Deregistered conflict strategy: %%s", name)


def registered_conflict_strategies() -> tuple[str, ...]:
    return tuple(spec.name for spec in _CONFLICT_REGISTRY)


register_conflict_strategy(
    "residual_grid",
    predicate=lambda runtime, residual_mode, *_: (
        residual_mode and getattr(runtime, "conflict_graph_impl", "") == "grid"
    ),
    factory=_ResidualGridConflictStrategy,
)

register_conflict_strategy(
    "residual",
    predicate=lambda runtime, residual_mode, has_residual: residual_mode and has_residual,
    factory=_ResidualConflictStrategy,
)

register_conflict_strategy(
    "segmented",
    predicate=lambda runtime, *_: getattr(runtime, "conflict_graph_impl", "") == "segmented",
    factory=_SegmentedConflictStrategy,
)

register_conflict_strategy(
    "grid",
    predicate=lambda runtime, *_: getattr(runtime, "conflict_graph_impl", "") == "grid",
    factory=_GridConflictStrategy,
)

register_conflict_strategy(
    "dense",
    predicate=lambda *_: True,
    factory=_DenseConflictStrategy,
)


def select_conflict_strategy(
    runtime: Any,
    *,
    residual_mode: bool,
    has_residual_distances: bool,
) -> ConflictGraphStrategy:
    for spec in _CONFLICT_REGISTRY:
        try:
            if spec.predicate(runtime, residual_mode, has_residual_distances):
                return spec.factory()
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.exception("Conflict strategy '%%s' predicate failed.", spec.name)
            continue
    raise RuntimeError("No conflict strategy registered for the current runtime.")


def describe_conflict_strategies() -> List[dict[str, str]]:
    return [
        {
            "name": spec.name,
            "module": spec.origin,
            "predicate": spec.predicate_label,
            "factory": f"{spec.factory.__module__}.{spec.factory.__qualname__}",
        }
        for spec in _CONFLICT_REGISTRY
    ]
