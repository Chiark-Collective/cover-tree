from __future__ import annotations

import json
from typing import Iterable, Sequence

import typer
import yaml

from profiles.loader import (
    ProfileError,
    ProfileMetadata,
    available_profiles,
    load_profile_definition,
)


app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    help="Profile utilities for listing and describing runtime presets.",
)


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Return a monospaced table for CLI output."""

    if not rows:
        return ""
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    fmt = "  ".join("{:<" + str(width) + "}" for width in widths)
    lines = [fmt.format(*headers)]
    lines.append(fmt.format(*("-" * width for width in widths)))
    for row in rows:
        lines.append(fmt.format(*row))
    return "\n".join(lines)


def _format_tags(tags: Iterable[str]) -> str:
    entries = tuple(tag for tag in tags if tag)
    return ", ".join(entries) if entries else "-"


def _condense_text(value: str | None) -> str:
    if not value:
        return "-"
    return " ".join(value.split())


@app.command("list")
def list_profiles() -> None:
    """Display available profile definitions and their metadata."""

    names = available_profiles()
    if not names:
        typer.echo("No profiles were found under profiles/.")
        raise typer.Exit(code=1)
    rows = []
    for slug in names:
        definition = load_profile_definition(slug)
        metadata: ProfileMetadata = definition.metadata
        rows.append(
            (
                slug,
                metadata.workload or "-",
                _format_tags(metadata.tags),
                _condense_text(metadata.description),
            )
        )
    headers = ("Profile", "Workload", "Tags", "Description")
    typer.echo(_render_table(headers, rows))


@app.command("describe")
def describe_profile(
    name: str,
    output_format: str = typer.Option(
        "yaml",
        "--format",
        "-f",
        case_sensitive=False,
        help="Output format (yaml or json).",
    ),
) -> None:
    """Describe a specific profile definition."""

    try:
        definition = load_profile_definition(name)
    except ProfileError as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = {
        "metadata": {
            "name": definition.metadata.name,
            "description": definition.metadata.description,
            "workload": definition.metadata.workload,
            "tags": list(definition.metadata.tags),
            "path": str(definition.path),
        },
        "runtime": definition.model.model_dump(),
    }
    normalized_format = output_format.lower()
    if normalized_format == "json":
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    elif normalized_format == "yaml":
        typer.echo(yaml.safe_dump(payload, sort_keys=False))
    else:
        raise typer.BadParameter(f"Unsupported format '{output_format}'. Expected yaml or json.")
