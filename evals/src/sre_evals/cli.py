"""`sre-eval` command line. Subcommands are added by the eval runner and reports."""

from __future__ import annotations

import typer

app = typer.Typer(help="Run evals and build results reports.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Run evals and build results reports."""
