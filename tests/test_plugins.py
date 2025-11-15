from __future__ import annotations

from types import SimpleNamespace

from typer.testing import CliRunner

from cli.pcct import app as pcct_app
from covertreex.algo.conflict.strategies import (
    register_conflict_strategy,
    deregister_conflict_strategy,
    select_conflict_strategy,
)
from covertreex.algo.traverse.base import TraversalStrategy
from covertreex.algo.traverse.strategies.registry import (
    deregister_traversal_strategy,
    registered_traversal_strategies,
)
from covertreex.plugins import traversal as traversal_plugins
from covertreex.plugins import conflict as conflict_plugins
from covertreex.plugins import metrics as metrics_plugins
from covertreex.core.metrics import Metric, _REGISTRY, get_metric
from covertreex.algo.conflict.base import ConflictGraphContext, ConflictGraphStrategy


class FakeEntryPoint:
    def __init__(self, name: str, payload):
        self.name = name
        self._payload = payload

    def load(self):
        return self._payload


class _DummyTraversal(TraversalStrategy):
    def collect(self, tree, batch, *, backend, runtime):
        raise AssertionError("not used")


class _DummyConflict(ConflictGraphStrategy):
    def build(self, ctx: ConflictGraphContext):
        raise AssertionError("not used")


def test_traversal_entrypoint_loader(monkeypatch) -> None:
    def fake_entrypoints(group: str):
        assert group == "covertreex.traversal"
        return [
            FakeEntryPoint(
                "extra_traversal",
                lambda: {
                    "name": "external_traversal",
                    "predicate": lambda runtime, backend: False,
                    "factory": _DummyTraversal,
                },
            )
        ]

    monkeypatch.setattr("covertreex.plugins._loader._select_entry_points", fake_entrypoints)
    traversal_plugins.load_entrypoints()
    assert "external_traversal" in registered_traversal_strategies()
    deregister_traversal_strategy("external_traversal")


def test_conflict_predicate_exception_falls_back(monkeypatch) -> None:
    def bad_predicate(*args, **kwargs):
        raise RuntimeError("boom")

    register_conflict_strategy(
        "failing_conflict",
        predicate=bad_predicate,
        factory=_DummyConflict,
    )
    runtime = SimpleNamespace(conflict_graph_impl="dense")
    strategy = select_conflict_strategy(runtime, residual_mode=False, has_residual_distances=False)
    assert not isinstance(strategy, _DummyConflict)
    deregister_conflict_strategy("failing_conflict")


def test_metric_entrypoint_loader(monkeypatch) -> None:
    def fake_entrypoints(group: str):
        assert group == "covertreex.metrics"

        def build_metric():
            def _pairwise(backend, lhs, rhs):
                return backend.asarray(lhs)

            def _pointwise(backend, lhs, rhs):
                return backend.asarray(lhs)

            return Metric(name="mock_metric", pairwise_kernel=_pairwise, pointwise_kernel=_pointwise)

        return [FakeEntryPoint("mock_metric", build_metric)]

    monkeypatch.setattr("covertreex.plugins._loader._select_entry_points", fake_entrypoints)
    metrics_plugins.load_entrypoints()
    metric = get_metric("mock_metric")
    assert metric.name == "mock_metric"
    _REGISTRY.unregister("mock_metric")


def test_cli_plugins_list_reports_entries() -> None:
    runner = CliRunner()
    result = runner.invoke(pcct_app, ["plugins"])
    assert result.exit_code == 0
    assert "traversal" in result.stdout
