"""`sre-ingest` command line. Subcommands are added by the ingestion pipeline."""

from __future__ import annotations

import typer

app = typer.Typer(help="Ingest versioned Kubernetes docs into Postgres.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Ingest versioned Kubernetes docs into Postgres."""
