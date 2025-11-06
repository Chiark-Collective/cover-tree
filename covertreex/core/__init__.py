"""Core data structures and persistence primitives for the PCCT."""

from .metrics import (
    Metric,
    MetricRegistry,
    available_metrics,
    configure_residual_metric,
    get_metric,
    reset_residual_metric,
)
from .persistence import (
    DEFAULT_JOURNAL_POOL,
    JournalScratchPool,
    PersistenceJournal,
    SliceUpdate,
    apply_persistence_journal,
    build_persistence_journal,
    clone_array_segment,
    clone_tree_with_updates,
)
from .tree import PCCTree, TreeBackend, TreeLogStats, get_runtime_backend

__all__ = [
    "PCCTree",
    "TreeBackend",
    "TreeLogStats",
    "SliceUpdate",
    "JournalScratchPool",
    "PersistenceJournal",
    "clone_array_segment",
    "clone_tree_with_updates",
    "build_persistence_journal",
    "apply_persistence_journal",
    "DEFAULT_JOURNAL_POOL",
    "Metric",
    "MetricRegistry",
    "available_metrics",
    "configure_residual_metric",
    "get_metric",
    "reset_residual_metric",
    "get_runtime_backend",
]
