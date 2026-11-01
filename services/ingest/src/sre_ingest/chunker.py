"""Heading-based chunker.

Algorithm
1. Scan `parsed_text` into lines (fenced blocks tracked) and cut it into sections at ATX headings.
2. A section is a list of blocks: paragraphs (split at blank lines) and whole fenced code blocks.
3. Blocks are packed greedily into chunks while `header + "\\n\\n" + body` stays within the token
   budget (gte-small tokens, 510). A block that cannot fit alone is split at line boundaries
   (code, flagged) or at line/word boundaries (prose); the pieces are never packed or merged.
4. Merge forward only, inside one document: a chunk whose body is tiny is joined with the next
   chunk (keeping the first section's path) when the result fits and neither side is a split
   piece. Bodies are always `parsed_text[char_start:char_end]`.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from sre_ingest.textutil import Line, scan_lines, slugify
from sre_shared.models import ChunkRecord, DocumentRecord

CHUNKER_VERSION = "1"
MAX_TOKENS = 510
MIN_BODY_TOKENS = 48
MAX_HEADER_PART = 120

TokenCounter = Callable[[str], int]


@dataclass
class ChunkStats:
    code_block_splits: int = 0
    """Fenced blocks too large for one chunk, split at line boundaries."""
    prose_splits: int = 0
    merged: int = 0
    chunks: int = 0
    anchors: dict[str, int] = field(default_factory=dict)


@dataclass
class _Piece:
    start: int
    end: int
    path: list[str]
    anchor: str | None
    split: bool = False


def make_header(version: str, title: str, section_path: list[str]) -> str:
    parts = [p[:MAX_HEADER_PART] for p in [title, *section_path]]
    return f"Kubernetes v{version} | " + " > ".join(parts)


def _normalize_for_key(body: str) -> str:
    return " ".join(body.split())


def chunk_key(source_path: str, k8s_version: str, body: str, dup_index: int = 0) -> str:
    material = f"{source_path}|{k8s_version}|{_normalize_for_key(body)}"
    if dup_index:
        material += f"|dup{dup_index}"
    return hashlib.sha256(material.encode()).hexdigest()


def _sections(text: str, lines: list[Line]) -> list[tuple[list[str], str | None, int, int]]:
    """(heading trail, anchor of the nearest heading, start, end) per section."""
    out: list[tuple[list[str], str | None, int, int]] = []
    stack: list[tuple[int, str]] = []
    anchor: str | None = None
    cur_start = 0
    for ln in lines:
        if not ln.level:
            continue
        out.append(([h for _, h in stack], anchor, cur_start, ln.start))
        while stack and stack[-1][0] >= ln.level:
            stack.pop()
        stack.append((ln.level, ln.heading))
        anchor = ln.anchor or slugify(ln.heading)
        cur_start = ln.next
    out.append(([h for _, h in stack], anchor, cur_start, len(text)))
    return out


def _blocks(text: str, lines: list[Line], start: int, end: int) -> list[tuple[int, int, bool]]:
    """(start, end, is_code) blocks of the trimmed region [start, end)."""
    blocks: list[tuple[int, int, bool]] = []
    cur: tuple[int, int, int | None] | None = None  # start, end, fence_id
    for ln in lines:
        if ln.start < start or ln.end > end:
            continue
        blank = not text[ln.start : ln.end].strip()
        if cur is not None:
            same_fence = ln.fence_id is not None and ln.fence_id == cur[2]
            if same_fence or (cur[2] is None and ln.fence_id is None and not blank):
                cur = (cur[0], ln.end, cur[2])
                continue
            blocks.append((cur[0], cur[1], cur[2] is not None))
            cur = None
        if blank and ln.fence_id is None:
            continue
        cur = (ln.start, ln.end, ln.fence_id)
    if cur is not None:
        blocks.append((cur[0], cur[1], cur[2] is not None))
    return blocks


def chunk_document(
    doc: DocumentRecord,
    count_tokens: TokenCounter,
    max_tokens: int = MAX_TOKENS,
    min_body_tokens: int = MIN_BODY_TOKENS,
) -> tuple[list[ChunkRecord], ChunkStats]:
    text = doc.parsed_text
    lines = scan_lines(text)
    stats = ChunkStats()
    pieces: list[_Piece] = []

    for path, anchor, s0, e0 in _sections(text, lines):
        region = text[s0:e0]
        if not region.strip():
            continue
        s = s0 + (len(region) - len(region.lstrip()))
        e = s0 + len(region.rstrip())
        header = make_header(doc.k8s_version, doc.title, path)

        def fits(a: int, b: int, header: str = header) -> bool:
            return count_tokens(f"{header}\n\n{text[a:b]}") <= max_tokens

        cur: tuple[int, int] | None = None
        for bs, be, is_code in _blocks(text, lines, s, e):
            if not fits(bs, be):
                if cur:
                    pieces.append(_Piece(*cur, path, anchor))
                    cur = None
                for ps, pe in _split_span(text, bs, be, fits, by_lines=is_code):
                    pieces.append(_Piece(ps, pe, path, anchor, split=True))
                if is_code:
                    stats.code_block_splits += 1
                else:
                    stats.prose_splits += 1
                continue
            if cur is not None and fits(cur[0], be):
                cur = (cur[0], be)
            else:
                if cur:
                    pieces.append(_Piece(*cur, path, anchor))
                cur = (bs, be)
        if cur:
            pieces.append(_Piece(*cur, path, anchor))

    pieces = _merge_forward(doc, text, pieces, count_tokens, max_tokens, min_body_tokens, stats)

    chunks: list[ChunkRecord] = []
    seen: dict[str, int] = {}
    for i, p in enumerate(pieces):
        body = text[p.start : p.end]
        header = make_header(doc.k8s_version, doc.title, p.path)
        base_key = chunk_key(doc.source_path, doc.k8s_version, body)
        dup = seen.get(base_key, 0)
        seen[base_key] = dup + 1
        key = chunk_key(doc.source_path, doc.k8s_version, body, dup) if dup else base_key
        url = doc.url + (f"#{p.anchor}" if p.anchor and p.path else "")
        chunks.append(
            ChunkRecord(
                chunk_key=key,
                source_path=doc.source_path,
                k8s_version=doc.k8s_version,
                ordinal=i,
                project=doc.project,
                doc_type=doc.doc_type,
                title=doc.title,
                section_path=p.path,
                url=url,
                language=doc.language,
                access_group=doc.access_group,
                header=header,
                body=body,
                char_start=p.start,
                char_end=p.end,
                token_count=count_tokens(f"{header}\n\n{body}"),
            )
        )
    stats.chunks = len(chunks)
    return chunks, stats


def _merge_forward(
    doc: DocumentRecord,
    text: str,
    pieces: list[_Piece],
    count_tokens: TokenCounter,
    max_tokens: int,
    min_body_tokens: int,
    stats: ChunkStats,
) -> list[_Piece]:
    out: list[_Piece] = []
    i = 0
    while i < len(pieces):
        cur = pieces[i]
        while (
            i + 1 < len(pieces)
            and not cur.split
            and not pieces[i + 1].split
            and count_tokens(text[cur.start : cur.end]) < min_body_tokens
        ):
            nxt = pieces[i + 1]
            header = make_header(doc.k8s_version, doc.title, cur.path)
            if count_tokens(f"{header}\n\n{text[cur.start : nxt.end]}") > max_tokens:
                break
            cur = _Piece(cur.start, nxt.end, cur.path, cur.anchor)
            stats.merged += 1
            i += 1
        out.append(cur)
        i += 1
    return out


def _split_span(
    text: str, start: int, end: int, fits: Callable[[int, int], bool], by_lines: bool
) -> list[tuple[int, int]]:
    """Cut [start, end) into pieces that fit: at line ends when `by_lines`, else at whitespace."""
    out: list[tuple[int, int]] = []
    pos = start
    while pos < end:
        if fits(pos, end):
            out.append((pos, end))
            break
        cuts = [m.start() for m in re.finditer(r"\n", text[pos:end])]
        cuts = [pos + c for c in cuts if pos + c > pos]
        if not by_lines or not cuts or not any(fits(pos, c) for c in cuts[:1]):
            cuts = [pos + m.start() for m in re.finditer(r"\s+", text[pos:end]) if m.start() > 0]
        best = pos
        lo, hi = 0, len(cuts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if fits(pos, cuts[mid]):
                best = cuts[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        if best == pos:  # one unbreakable token: hard cut by characters
            best = _hard_cut(pos, end, fits)
        out.append((pos, best))
        pos = best
        while pos < end and (text[pos] in "\r\n" if by_lines else text[pos].isspace()):
            pos += 1
    return out


def _hard_cut(pos: int, end: int, fits: Callable[[int, int], bool]) -> int:
    lo, hi, best = pos + 1, end, pos + 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if fits(pos, mid):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best
