"""Line scanning shared by the parser and the chunker: headings and fenced code blocks."""

from __future__ import annotations

import re
from dataclasses import dataclass

FENCE_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
HEADING_RE = re.compile(r"^ {0,3}(?P<hashes>#{1,6})[ \t]+(?P<text>.*?)(?:[ \t]+#+)?[ \t]*$")
ANCHOR_RE = re.compile(r"\s*\{#(?P<id>[\w.-]+)\}\s*$")
LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")


@dataclass(frozen=True)
class Line:
    start: int
    end: int
    """End offset excluding the line break."""
    next: int
    """Offset of the following line."""
    fence_id: int | None
    """Same value for every line of one fenced block (fence lines included)."""
    level: int = 0
    """Heading level (1-6) for ATX headings outside fences, else 0."""
    heading: str = ""
    anchor: str | None = None


def scan_lines(text: str) -> list[Line]:
    out: list[Line] = []
    pos = 0
    fence_id = -1
    open_fence: tuple[str, int] | None = None
    for raw in text.splitlines(keepends=True):
        stripped = raw.rstrip("\r\n")
        start, end, nxt = pos, pos + len(stripped), pos + len(raw)
        pos = nxt
        m = FENCE_RE.match(stripped)
        if open_fence is not None:
            ch, n = open_fence
            closes = (
                m is not None
                and m["fence"][0] == ch
                and len(m["fence"]) >= n
                and not m["info"].strip()
            )
            out.append(Line(start, end, nxt, fence_id))
            if closes:
                open_fence = None
            continue
        if m is not None:
            fence_id += 1
            open_fence = (m["fence"][0], len(m["fence"]))
            out.append(Line(start, end, nxt, fence_id))
            continue
        h = HEADING_RE.match(stripped)
        if h is not None and h["text"].strip():
            title = h["text"].strip()
            am = ANCHOR_RE.search(title)
            anchor = am["id"] if am else None
            if am:
                title = title[: am.start()].strip()
            title = LINK_RE.sub(r"\1", title).replace("`", "").replace("**", "").strip()
            out.append(Line(start, end, nxt, None, len(h["hashes"]), title, anchor))
        else:
            out.append(Line(start, end, nxt, None))
    return out


def slugify(heading: str) -> str:
    """Approximation of Hugo's (Goldmark) auto heading ids."""
    s = heading.strip().lower().replace(" ", "-")
    return re.sub(r"[^\w-]", "", s)
