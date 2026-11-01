"""Results reports: README table and docs/results.md."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from psycopg import AsyncConnection

from sre_evals.report.details import (
    latest_runs,
    latest_sweep_runs,
    render_details,
    render_sweeps,
    storage_stats,
)
from sre_evals.report.readme import END, START, render_block, replace_block


def _rewrite_readme(readme: Path, runs: list[dict[str, Any]]) -> None:
    text = readme.read_bytes().decode("utf-8")
    new = replace_block(text, render_block(runs))
    if new != text:
        readme.write_bytes(new.encode("utf-8"))


def _write_details(details: Path, content: str) -> None:
    details.parent.mkdir(parents=True, exist_ok=True)
    details.write_bytes(content.encode("utf-8"))


async def build_reports(
    conn: AsyncConnection[Any], readme: Path, details: Path | None
) -> list[dict[str, Any]]:
    """Rewrite only the README results block; write the details page. Returns the runs used."""
    runs = await latest_runs(conn)
    await asyncio.to_thread(_rewrite_readme, readme, runs)
    if details is not None:
        corpus_ids = [r["corpus_version_id"] for r in runs if r["corpus_version_id"] is not None]
        storage = await storage_stats(conn, max(corpus_ids)) if corpus_ids else {}
        sweeps = await latest_sweep_runs(conn)
        await asyncio.to_thread(_write_details, details, render_details(runs, storage, sweeps))
    return runs


__all__ = [
    "END",
    "START",
    "build_reports",
    "latest_runs",
    "latest_sweep_runs",
    "render_block",
    "render_details",
    "render_sweeps",
    "replace_block",
    "storage_stats",
]
