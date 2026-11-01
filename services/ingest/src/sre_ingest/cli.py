"""`sre-ingest` command line."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from sre_ingest import report as report_mod
from sre_ingest import snapshot as snapshot_mod
from sre_ingest import sources, store
from sre_shared.config import load_corpus_config
from sre_shared.models import EmbeddingModelName

app = typer.Typer(help="Ingest versioned Kubernetes docs into Postgres.", no_args_is_help=True)
snapshot_app = typer.Typer(help="Export or load a corpus snapshot (Parquet).", no_args_is_help=True)
app.add_typer(snapshot_app, name="snapshot")

DsnOpt = Annotated[
    str | None, typer.Option("--dsn", help="Postgres DSN (default: DATABASE_URL / lab stack)")
]
VALID_MODELS = ("gte-small", "bge-large", "bge-m3")


def _models(raw: str) -> list[EmbeddingModelName]:
    names = [m.strip() for m in raw.split(",") if m.strip()]
    bad = [m for m in names if m not in VALID_MODELS]
    if bad:
        raise typer.BadParameter(f"unknown model(s) {bad}; choose from {VALID_MODELS}")
    return [m for m in VALID_MODELS if m in names]  # type: ignore[misc]


@app.callback()
def main() -> None:
    """Ingest versioned Kubernetes docs into Postgres."""


@app.command()
def pin(count: Annotated[int, typer.Option(help="Newest N minor versions")] = 3) -> None:
    """Resolve the newest N versions from main's hugo.toml and write config/corpus.yaml."""
    cfg = sources.pin(count)
    for s in cfg.versions:
        typer.echo(f"{s.k8s_version}  {s.branch}  {s.commit_sha}  {s.url_base}")


@app.command()
def fetch(
    versions: Annotated[
        str | None, typer.Option(help="Comma-separated, default all pinned")
    ] = None,
) -> None:
    """Sparse, shallow checkout of each pinned commit under data/sources/<version>/."""
    cfg = load_corpus_config()
    wanted = versions.split(",") if versions else None
    for s in cfg.versions:
        if wanted is None or s.k8s_version in wanted:
            typer.echo(f"{s.k8s_version}: {sources.fetch(cfg, s)}")


@app.command()
def run(
    versions: Annotated[
        str | None, typer.Option(help="Comma-separated, default all pinned")
    ] = None,
    models: Annotated[
        str, typer.Option(help="Comma-separated: gte-small,bge-large,bge-m3")
    ] = "gte-small",
    limit_docs: Annotated[int | None, typer.Option(help="Per version, evenly sampled")] = None,
    activate: Annotated[bool, typer.Option("--activate")] = False,
    no_fetch: Annotated[bool, typer.Option("--no-fetch", help="Use existing checkouts")] = False,
    dsn: DsnOpt = None,
) -> None:
    """fetch -> parse -> chunk -> embed -> upsert -> ready (-> activate)."""
    from sre_ingest.pipeline import run_ingest

    cfg = load_corpus_config()
    res = run_ingest(
        cfg,
        dsn=dsn,
        versions=versions.split(",") if versions else None,
        models=_models(models),
        limit_docs=limit_docs,
        activate=activate,
        do_fetch=not no_fetch,
        log=typer.echo,
    )
    with store.open_conn(dsn) as conn:
        rep = report_mod.build_report(conn, res.corpus_version_id)
    path = report_mod.write_report(rep)
    typer.echo(
        f"corpus_version={res.corpus_version_id} created={res.created_version} "
        f"docs={res.documents_seen} chunks_inserted={res.chunks_inserted} "
        f"chunks_total={res.chunks_total} report={path}"
    )


@app.command()
def embed(
    corpus_version: Annotated[str, typer.Option(help="id or 'building'-state version id")],
    models: Annotated[str, typer.Option()] = "gte-small",
    dsn: DsnOpt = None,
) -> None:
    """Fill missing embeddings of a corpus version for the given models."""
    from sre_ingest.embedding import embed_pending
    from sre_ingest.pipeline import load_embedders

    embedders, _ = load_embedders(_models(models), typer.echo)
    with store.open_conn(dsn) as conn:
        cv = store.resolve_version(conn, corpus_version)
        for name, emb in embedders.items():
            typer.echo(json.dumps(embed_pending(conn, cv, name, emb).as_dict()))


@app.command()
def activate(corpus_version_id: int, dsn: DsnOpt = None) -> None:
    """ready -> active; the previous active version becomes retired."""
    with store.open_conn(dsn) as conn:
        store.activate(conn, corpus_version_id)
    typer.echo(f"activated {corpus_version_id}")


@app.command()
def prune(dsn: DsnOpt = None) -> None:
    """Delete retired corpus versions (documents and chunks cascade)."""
    with store.open_conn(dsn) as conn:
        typer.echo(f"deleted {store.prune(conn)} retired version(s)")


@app.command()
def report(
    corpus_version: Annotated[str, typer.Option(help="id or 'active'")] = "active",
    dsn: DsnOpt = None,
) -> None:
    """Write a JSON report under data/reports/ and print its path."""
    with store.open_conn(dsn) as conn:
        cv = store.resolve_version(conn, corpus_version)
        rep = report_mod.build_report(conn, cv)
    typer.echo(str(report_mod.write_report(rep)))


@snapshot_app.command("export")
def snapshot_export(
    corpus_version: Annotated[str, typer.Option("--corpus-version", help="id or 'active'")],
    out: Annotated[Path, typer.Option("--out")],
    models: Annotated[
        str | None, typer.Option(help="Comma-separated models to keep (default: all held)")
    ] = None,
    dsn: DsnOpt = None,
) -> None:
    keep = [m.strip() for m in models.split(",") if m.strip()] if models else None
    with store.open_conn(dsn) as conn:
        cv = store.resolve_version(conn, corpus_version)
        counts = snapshot_mod.export_snapshot(conn, cv, out, keep)
    typer.echo(f"exported corpus_version={cv} {counts} -> {out}")


@snapshot_app.command("load")
def snapshot_load(
    path: Annotated[Path, typer.Option("--in")],
    activate: Annotated[bool, typer.Option("--activate")] = False,
    dsn: DsnOpt = None,
) -> None:
    with store.open_conn(dsn) as conn:
        cv, counts = snapshot_mod.load_snapshot(conn, path, activate=activate)
    typer.echo(f"loaded corpus_version={cv} {counts}")
