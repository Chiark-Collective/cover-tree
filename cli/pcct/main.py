from __future__ import annotations

import typer

from cli.queries.app import app as legacy_query_app, disable_legacy_entrypoint_warning

from .benchmark_cli import benchmark_app
from .build_cli import build_app
from .doctor import app as doctor_app
from .profile import app as profile_app
from .plugins_cli import plugins_app
from .query_cli import query_app as new_query_app
from .telemetry_cli import telemetry_app


_HELP = """Parallel compressed cover tree (PCCT) command line interface.

Subcommands cover benchmarking, profiles, diagnostics, and compatibility
with the legacy `python -m cli.queries` entrypoint."""

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
    help=_HELP,
)


@app.callback()
def pcct_callback() -> None:
    """Root callback reserved for shared options (none yet)."""

    disable_legacy_entrypoint_warning()


# Register Typer subcommands
app.add_typer(profile_app, name="profile", help="Inspect configuration profiles.")
app.add_typer(new_query_app, name="query", help="Run PCCT benchmarks with profiles.")
app.add_typer(build_app, name="build", help="Construct trees with telemetry summaries.")
app.add_typer(benchmark_app, name="benchmark", help="Repeat query runs and aggregate metrics.")
app.add_typer(plugins_app, name="plugins", help="Inspect registered plugins.")
app.add_typer(telemetry_app, name="telemetry", help="Inspect telemetry artifacts.")
disable_legacy_entrypoint_warning()
app.add_typer(
    legacy_query_app,
    name="legacy-query",
    help="Legacy benchmark CLI (deprecated; use `pcct query`).",
)
app.add_typer(doctor_app, name="doctor", help="Run preflight environment checks.")


def main() -> None:
    app()


__all__ = ["app", "main"]
