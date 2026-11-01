"""Parse `[n]` markers in a model answer and map them onto the sources it was given."""

from __future__ import annotations

import re
from collections.abc import Sequence

from sre_shared.models import Citation, RetrievedChunk

_NUM = r"\d+"
_MARK = rf"\[\s*{_NUM}(?:\s*,\s*{_NUM})*\s*\]"
# A run of markers like `[1][2]`; the first `[` must not follow an identifier character, `.`,
# `]` or `}` (that is an index such as `items[0]` or `{.a[0]}`, not a citation).
_RUN = re.compile(rf"([ \t]?)(?<![\w.\]}}])((?:{_MARK})+)")
_MARK_RE = re.compile(_MARK)
_PROTECTED = re.compile(r"(```.*?(?:```|\Z)|`[^`\n]*`)", re.DOTALL)
MAX_DIGITS = 6  # a longer number cannot be a source index; it is invalid without int()


def _clip(num: str) -> str:
    num = num.lstrip("0") or "0"
    return num[:12]  # long runs are recorded by their first digits


def resolve_citations(
    text: str,
    sources: Sequence[RetrievedChunk],
    groups: Sequence[Sequence[RetrievedChunk]] | None = None,
) -> tuple[str, list[Citation], list[str]]:
    """Return (clean text, citations, invalid markers).

    `[n]` refers to the n-th source (1-based); a source is one chunk, or a group of same-section
    chunks (`groups`), in which case it expands to one citation per member in rank order. Numbers
    outside 1..len(sources) are removed from the text and listed (first-mention order, no
    duplicates). Markers inside inline code or fenced blocks are left alone. Valid ones become
    citations, de-duplicated, in first-mention order.
    """
    if groups is None:
        groups = [[c] for c in sources]
    valid: list[int] = []
    invalid: list[str] = []

    def one(m: re.Match[str]) -> str:
        nums = re.findall(_NUM, m.group(0))
        keep: list[str] = []
        for raw in nums:
            n = int(raw) if len(raw) <= MAX_DIGITS else -1
            if 1 <= n <= len(groups):
                keep.append(str(n))
                if n not in valid:
                    valid.append(n)
            elif _clip(raw) not in invalid:
                invalid.append(_clip(raw))
        if not keep:
            return ""
        return m.group(0) if len(keep) == len(nums) else f"[{', '.join(keep)}]"

    def run(m: re.Match[str]) -> str:
        body = _MARK_RE.sub(one, m.group(2))
        return m.group(1) + body if body else ""

    parts = _PROTECTED.split(text)
    clean = "".join(p if i % 2 else _RUN.sub(run, p) for i, p in enumerate(parts))
    cites: list[Citation] = []
    seen: set[int] = set()
    for n in valid:
        for c in groups[n - 1]:
            if c.chunk_id in seen:
                continue
            seen.add(c.chunk_id)
            cites.append(
                Citation(
                    chunk_id=c.chunk_id,
                    source_path=c.source_path,
                    k8s_version=c.k8s_version,
                    url=c.url,
                    title=c.title,
                )
            )
    return clean, cites, invalid
