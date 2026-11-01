"""Picking context chunks for golden-question synthesis.

Factual contexts are concept/task chunks spread across documents. Exact-string contexts are
chunks that contain a literal worth asking about (an error name, a CLI flag or a field path),
found by regex; the matched string is recorded so the question can be forced to contain it.
Selection is deterministic for a seed.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from psycopg import AsyncConnection
from pydantic import BaseModel

PickStratum = Literal["factual", "exact_string"]

MIN_BODY_CHARS = 200
MAX_BODY_CHARS = 2500

# Well-known pod/container status and event reasons; extended by the generic Err* pattern.
_ERROR_NAMES = (
    "CrashLoopBackOff",
    "ImagePullBackOff",
    "ErrImagePull",
    "ErrImageNeverPull",
    "OOMKilled",
    "CreateContainerConfigError",
    "CreateContainerError",
    "RunContainerError",
    "ContainerCannotRun",
    "FailedScheduling",
    "FailedMount",
    "NodeNotReady",
    "DeadlineExceeded",
    "Unschedulable",
    "InvalidImageName",
    "ContainerStatusUnknown",
)
ERROR_RE = re.compile(
    r"(?<![\w.-])(?:" + "|".join(_ERROR_NAMES) + r"|Err[A-Z][A-Za-z]{3,})(?![\w-])"
)
# Multi-word long flags such as --max-surge (single-word flags are too generic to be a test).
FLAG_RE = re.compile(r"(?<![\w-])--[a-z][a-z0-9]*(?:-[a-z0-9]+)+(?![\w-])")
FIELD_RE = re.compile(
    r"(?<![\w.])(?:spec|status|metadata)(?:\.[a-z][A-Za-z0-9]*(?:\[\])?){2,}(?![\w.])"
)
_KINDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("error", ERROR_RE),
    ("flag", FLAG_RE),
    ("field", FIELD_RE),
)


class ContextRow(BaseModel):
    """One picked chunk; the unit synthesize works on (one golden per row)."""

    chunk_key: str
    source_path: str
    k8s_version: str
    url: str
    title: str
    body: str
    stratum: PickStratum
    matched: str | None = None
    match_kind: str | None = None


def find_exact_strings(text: str) -> list[tuple[str, str]]:
    """All (kind, string) candidates in text, in order of first appearance, deduplicated."""
    found: dict[str, tuple[int, str]] = {}
    for kind, pattern in _KINDS:
        for m in pattern.finditer(text):
            s = m.group(0)
            if s not in found or m.start() < found[s][0]:
                found[s] = (m.start(), kind)
    return [(kind, s) for s, (_, kind) in sorted(found.items(), key=lambda kv: kv[1][0])]


def read_contexts(path: Path) -> list[ContextRow]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if line.strip():
                try:
                    rows.append(ContextRow.model_validate_json(line))
                except ValueError as e:
                    raise ValueError(f"{path}:{lineno}: {e}") from e
    return rows


def write_contexts(path: Path, rows: Iterable[ContextRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(r.model_dump_json(exclude_none=True) + "\n")


async def active_corpus_version(conn: AsyncConnection[Any]) -> int:
    row = await (
        await conn.execute(
            "select id from public.corpus_versions"
            " order by (status = 'active') desc, id desc limit 1"
        )
    ).fetchone()
    if row is None:
        raise RuntimeError("no corpus version in the database; run ingest first")
    return int(row["id"])


async def fetch_chunks(
    conn: AsyncConnection[Any],
    *,
    versions: Sequence[str] | None,
    doc_types: Sequence[str] | None,
    corpus_version_id: int | None = None,
) -> list[dict[str, Any]]:
    cv = corpus_version_id or await active_corpus_version(conn)
    vs = list(versions) if versions else None
    dts = list(doc_types) if doc_types else None
    cur = await conn.execute(
        "select chunk_key, source_path, k8s_version, url, title, body, doc_type"
        " from public.chunks"
        " where corpus_version_id = %s and access_group = 'public' and language = 'en'"
        "   and (%s::text[] is null or k8s_version = any(%s::text[]))"
        "   and (%s::text[] is null or doc_type = any(%s::text[]))"
        "   and char_length(body) between %s and %s"
        " order by source_path, k8s_version, ordinal, chunk_key",
        [cv, vs, vs, dts, dts, MIN_BODY_CHARS, MAX_BODY_CHARS],
    )
    return list(await cur.fetchall())


def _spread(
    candidates: list[ContextRow], n: int, rng: random.Random, *, distinct_matches: bool
) -> list[ContextRow]:
    """Round-robin across documents (order shuffled by the seed) until n rows are chosen."""
    by_doc: dict[tuple[str, str], list[ContextRow]] = defaultdict(list)
    for c in candidates:
        by_doc[(c.source_path, c.k8s_version)].append(c)
    docs = sorted(by_doc)
    rng.shuffle(docs)
    for d in docs:
        rng.shuffle(by_doc[d])
    chosen: list[ContextRow] = []
    seen_matches: set[str] = set()
    leftovers: list[ContextRow] = []
    round_no = 0
    while len(chosen) < n and any(len(by_doc[d]) > round_no for d in docs):
        for d in docs:
            if len(chosen) >= n:
                break
            if len(by_doc[d]) <= round_no:
                continue
            c = by_doc[d][round_no]
            if distinct_matches and c.matched in seen_matches:
                leftovers.append(c)
                continue
            if c.matched:
                seen_matches.add(c.matched)
            chosen.append(c)
        round_no += 1
    chosen.extend(leftovers[: n - len(chosen)])
    return chosen[:n]


def select_contexts(
    chunks: Sequence[dict[str, Any]], stratum: PickStratum, n: int, seed: int
) -> list[ContextRow]:
    """Pure selection step, deterministic for (chunks, stratum, n, seed)."""
    rng = random.Random(seed)  # noqa: S311  (selection, not security)
    candidates: list[ContextRow] = []
    for ch in chunks:
        base = {k: ch[k] for k in ("chunk_key", "source_path", "k8s_version", "url", "title")}
        if stratum == "factual":
            if ch["doc_type"] not in ("concept", "task"):
                continue
            candidates.append(ContextRow(body=ch["body"], stratum="factual", **base))
        else:
            hits = find_exact_strings(ch["body"])
            if not hits:
                continue
            kind, s = hits[rng.randrange(len(hits))]
            candidates.append(
                ContextRow(
                    body=ch["body"], stratum="exact_string", matched=s, match_kind=kind, **base
                )
            )
    picked = _spread(candidates, n, rng, distinct_matches=stratum == "exact_string")
    return sorted(picked, key=lambda r: (r.source_path, r.k8s_version, r.chunk_key))


async def pick_contexts(
    conn: AsyncConnection[Any],
    stratum: PickStratum,
    n: int,
    versions: Sequence[str] | None,
    seed: int,
    corpus_version_id: int | None = None,
) -> list[ContextRow]:
    doc_types = ["concept", "task"] if stratum == "factual" else None
    chunks = await fetch_chunks(
        conn, versions=versions, doc_types=doc_types, corpus_version_id=corpus_version_id
    )
    return select_contexts(chunks, stratum, n, seed)
