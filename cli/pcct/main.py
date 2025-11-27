from __future__ import annotations

import typer

from .benchmark_cli import benchmark_app
from .build_cli import build_app
from .doctor import app as doctor_app
from .profile import app as profile_app
from .plugins_cli import plugins_app
from .query_cli import query_app as new_query_app
from .telemetry_cli import telemetry_app
from .breakdown_cli import breakdown_app


_HELP = """[bold]Covertreex[/bold] — Parallel compressed cover tree (PCCT) CLI.

High-performance k-NN queries with residual correlation for Gaussian process pipelines.

[bold cyan]Quick Start[/bold cyan]
  [dim]#[/dim] Run a basic benchmark
  python -m cli.pcct query --dimension 3 --tree-points 8192 --k 10

  [dim]#[/dim] Run with Rust backend (fastest)
  python -m cli.pcct query --engine rust-hilbert --tree-points 32768

  [dim]#[/dim] Run residual metric benchmark
  python -m cli.pcct query --metric residual --engine rust-hilbert

  [dim]#[/dim] Use a profile preset
  python -m cli.pcct query --profile residual-gold

[bold cyan]Common Workflows[/bold cyan]
  • [bold]query[/bold]     — Run k-NN benchmarks (most common)
  • [bold]build[/bold]     — Build tree only, measure construction time
  • [bold]profile[/bold]   — List/describe available profile presets
  • [bold]doctor[/bold]    — Check environment and dependencies

[bold cyan]Profiles[/bold cyan]
  Profiles are YAML presets in profiles/. List with: profile list
  Available: default, residual-gold, residual-fast, residual-audit, cpu-debug

[bold cyan]Engines[/bold cyan]
  • python-numba  — Reference impl with full telemetry
  • rust-natural  — Rust backend, natural point order
  • rust-hilbert  — Rust + Hilbert ordering [bold green](fastest)[/bold green]"""

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
    pass


# Register Typer subcommands
app.add_typer(profile_app, name="profile", help="List/describe profile presets from profiles/")
app.add_typer(new_query_app, name="query", help="Run k-NN benchmark with tree build + queries")
app.add_typer(build_app, name="build", help="Build tree only (no queries), measure construction")
app.add_typer(benchmark_app, name="benchmark", help="Run query benchmark multiple times, aggregate stats")
app.add_typer(breakdown_app, name="breakdown", help="Generate per-phase timing breakdown plots")
app.add_typer(plugins_app, name="plugins", help="List registered traversal/metric plugins")
app.add_typer(telemetry_app, name="telemetry", help="Inspect/export JSONL telemetry artifacts")
app.add_typer(doctor_app, name="doctor", help="Check Numba/JAX availability, verify environment")


def main() -> None:
    app()


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
