"""Parse `[n]` markers in a model answer and map them onto the sources it was given."""

from __future__ import annotations

import re
from collections.abc import Sequence

from sre_shared.models import Citation, RetrievedChunk

_GROUP = re.compile(r"([ \t]?)\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


def resolve_citations(
    text: str, sources: Sequence[RetrievedChunk]
) -> tuple[str, list[Citation], list[str]]:
    """Return (clean text, citations, invalid markers).

    `[n]` refers to the n-th source (1-based). Numbers outside 1..len(sources) are removed from
    the text and listed (first-mention order, no duplicates). Valid ones become citations,
    de-duplicated, in first-mention order.
    """
    valid: list[int] = []
    invalid: list[str] = []

    def sub(m: re.Match[str]) -> str:
        nums = [int(x) for x in re.findall(r"\d+", m.group(2))]
        keep: list[int] = []
        for n in nums:
            if 1 <= n <= len(sources):
                keep.append(n)
                if n not in valid:
                    valid.append(n)
            elif str(n) not in invalid:
                invalid.append(str(n))
        if not keep:
            return ""
        if len(keep) == len(nums):
            return m.group(0)
        return f"{m.group(1)}[{', '.join(map(str, keep))}]"

    clean = _GROUP.sub(sub, text)
    cites = [
        Citation(
            chunk_id=sources[n - 1].chunk_id,
            source_path=sources[n - 1].source_path,
            k8s_version=sources[n - 1].k8s_version,
            url=sources[n - 1].url,
            title=sources[n - 1].title,
        )
        for n in valid
    ]
    return clean, cites, invalid
