from __future__ import annotations

import typer

from covertreex.plugins import conflict as conflict_plugins
from covertreex.plugins import metrics as metric_plugins
from covertreex.plugins import traversal as traversal_plugins

plugins_app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    invoke_without_command=True,
    help="Inspect registered traversal/conflict/metric plugins.",
)


def _format_rows() -> list[str]:
    entries: list[tuple[str, dict[str, str]]] = []
    entries += [("traversal", item) for item in traversal_plugins.list_plugins()]
    entries += [("conflict", item) for item in conflict_plugins.list_plugins()]
    entries += [("metric", item) for item in metric_plugins.list_metrics()]
    rows: list[str] = []
    if not entries:
        return rows
    header = f"{'type':<10} {'name':<24} {'module':<40} predicate/factory"
    rows.append(header)
    rows.append("-" * len(header))
    for plugin_type, item in entries:
        module = item.get("module", "-")
        predicate = item.get("predicate") or item.get("factory", "")
        rows.append(f"{plugin_type:<10} {item['name']:<24} {module:<40} {predicate}")
    return rows


@plugins_app.callback()
def plugins() -> None:
    rows = _format_rows()
    if not rows:
        typer.echo("No plugins registered.")
        return
    for line in rows:
        typer.echo(line)


__all__ = ["plugins_app"]
