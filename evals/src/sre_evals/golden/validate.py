"""Validate a golden file against the corpus: gold quotes exist, ids are unique, strings present."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from psycopg import AsyncConnection

from sre_shared.models import GoldenItem

EXACT_TAG = "exact:"
_WS = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def exact_string_of(item: GoldenItem) -> str | None:
    """The string an exact_string question must contain, carried in the `exact:<string>` tag."""
    for t in item.tags:
        if t.startswith(EXACT_TAG):
            return t[len(EXACT_TAG) :]
    return None


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    counts: dict[str, Counter[str]] = field(default_factory=dict)  # stratum -> status counts

    @property
    def ok(self) -> bool:
        return not self.errors


def read_items_lenient(path: Path) -> tuple[list[GoldenItem], list[str]]:
    """Parse every line; unlike `load_golden` this reports bad lines and duplicates as errors."""
    items: list[GoldenItem] = []
    errors: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = GoldenItem.model_validate(json.loads(line))
            except ValueError as e:
                errors.append(f"line {lineno}: not a valid GoldenItem ({str(e)[:120]})")
                continue
            if item.id in seen:
                errors.append(f"line {lineno}: duplicate id {item.id}")
            seen.add(item.id)
            items.append(item)
    return items, errors


async def _fetch_documents(
    conn: AsyncConnection[Any], pairs: set[tuple[str, str]]
) -> dict[tuple[str, str], list[str]]:
    if not pairs:
        return {}
    paths = [p for p, _ in pairs]
    versions = [v for _, v in pairs]
    cur = await conn.execute(
        "select d.source_path, d.k8s_version, d.parsed_text from public.documents d"
        " join unnest(%s::text[], %s::text[]) as w(p, v)"
        "   on d.source_path = w.p and d.k8s_version = w.v",
        [paths, versions],
    )
    docs: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in await cur.fetchall():
        docs[(row["source_path"], row["k8s_version"])].append(normalize_ws(row["parsed_text"]))
    return docs


async def validate_items(
    conn: AsyncConnection[Any], items: list[GoldenItem], errors: list[str] | None = None
) -> ValidationReport:
    report = ValidationReport(errors=list(errors or []))
    accepted = [i for i in items if i.status == "accepted"]
    pairs = {(p.source_path, p.k8s_version) for i in accepted for p in i.gold_passages}
    docs = await _fetch_documents(conn, pairs)
    for i in items:
        report.counts.setdefault(i.stratum, Counter())[i.status] += 1
    for i in accepted:
        if not i.gold_passages:
            report.errors.append(f"{i.id}: accepted item has no gold passage")
        for n, p in enumerate(i.gold_passages):
            texts = docs.get((p.source_path, p.k8s_version))
            where = f"{i.id}: passage {n} ({p.source_path} v{p.k8s_version})"
            if not texts:
                report.errors.append(f"{where}: no such document in the database")
            elif not any(normalize_ws(p.quote) in t for t in texts):
                report.errors.append(f"{where}: quote not found in parsed_text")
        if i.stratum == "exact_string":
            s = exact_string_of(i)
            if s is None:
                report.errors.append(f"{i.id}: exact_string item has no 'exact:<string>' tag")
            elif s not in i.question:
                report.errors.append(f"{i.id}: question does not contain {s!r}")
    return report


async def validate_file(conn: AsyncConnection[Any], path: Path) -> ValidationReport:
    items, errors = read_items_lenient(path)
    return await validate_items(conn, items, errors)
