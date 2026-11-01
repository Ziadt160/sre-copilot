"""`sre-golden` command line. Subcommands are added by the golden-set tooling."""

from __future__ import annotations

import typer

app = typer.Typer(help="Build, review and validate the golden set.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Build, review and validate the golden set."""
