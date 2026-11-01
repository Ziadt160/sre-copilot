"""Hugo Markdown -> DocumentRecord: front matter, shortcode handling, normalization.

Shortcode policy (surveyed on release-1.36, see the T01 bucket entries): every shortcode name is
in exactly one table below. Handled ones are rendered to text, GENERATED ones are dropped because
Hugo builds them from data we do not fetch, anything else is "unknown": its tags are stripped,
inner text is kept and the name is counted in `ParseStats.unknown`.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from sre_ingest.textutil import scan_lines
from sre_shared.models import DocType, DocumentRecord

TAG_RE = re.compile(r"\{\{([<%])\s*(/)?\s*([A-Za-z_][\w-]*(?:/[\w-]+)*)?(.*?)([>%])\}\}", re.S)
ESC_RE = re.compile(r"\{\{([<%])/\*(.*?)\*/([>%])\}\}", re.S)
ARG_RE = re.compile(r"""(?:([\w-]+)=)?(?:"([^"]*)"|'([^']*)'|`([^`]*)`|([^\s"'`=]+))""", re.S)
FRONT_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
TAB_MARK = re.compile(r"^\x01TAB (.*)\x01$")
ESC_OPEN, ESC_CLOSE = "", ""

BLOCK = {
    "note", "caution", "warning", "alert", "comment", "mermaid", "highlight", "tabs", "tab",
    "table", "details", "example", "pageinfo", "tutorials/module", "tutorials/modules",
    "tutorials/carousel", "tutorials/carousel-item",
}  # fmt: skip
INLINE_HANDLED = {
    "glossary_tooltip", "glossary_definition", "heading", "feature-state", "skew", "param",
    "latest-version", "release-branch", "include", "code_sample", "codenew", "ref", "relref",
    "figure", "version-check", "latest-semver",
}  # fmt: skip
GENERATED = {
    "api-reference", "page-api-reference", "feature-gate-table", "feature-gate-list",
    "feature-gate-description", "cve-feed", "cncf-landscape", "doc-versions-list",
    "latest-release-notes", "legacy-repos-deprecation", "dockershim-removal", "kat-button",
    "thirdparty-content", "link", "eol-releases", "release-binaries", "release-data",
    "upcoming-releases", "upcoming-events", "kubecon-events", "kubeweekly", "social-media-feed",
    "toc", "case-studies", "katacoda-tutorial", "blocks", "release-branches",
}  # fmt: skip
CALLOUT_DEFAULT = {"note": "Note:", "caution": "Caution:", "warning": "Warning:"}
EXT_LANG = {
    "yaml": "yaml", "yml": "yaml", "json": "json", "sh": "shell", "go": "go", "py": "python",
    "toml": "toml", "ini": "ini", "html": "html", "js": "javascript",
}  # fmt: skip
SKEW_SHIFT = {
    "currentVersion": 0, "latestVersion": 0, "prevMinorVersion": -1,
    "nextMinorVersion": 1, "oldestMinorVersion": -3,
}  # fmt: skip
MAX_INCLUDE_DEPTH = 5


@dataclass
class ParseStats:
    handled: Counter[str] = field(default_factory=Counter)
    generated: Counter[str] = field(default_factory=Counter)
    unknown: Counter[str] = field(default_factory=Counter)
    problems: Counter[str] = field(default_factory=Counter)
    """Missing include/example/glossary/i18n lookups, stray closing tags."""

    def merge(self, other: ParseStats) -> None:
        self.handled.update(other.handled)
        self.generated.update(other.generated)
        self.unknown.update(other.unknown)
        self.problems.update(other.problems)


@dataclass
class ParseContext:
    root: Path
    """Checkout root of one pinned version (holds content/ and i18n/)."""
    minor: str
    """e.g. "1.36"."""
    url_base: str
    stats: ParseStats = field(default_factory=ParseStats)
    i18n: dict[str, str] = field(default_factory=dict)
    lang: str = "en"

    @classmethod
    def load(cls, root: Path, minor: str, url_base: str) -> ParseContext:
        ctx = cls(root=root, minor=minor, url_base=url_base.rstrip("/"))
        path = root / "i18n" / "en" / "en.toml"
        if path.is_file():
            with path.open("rb") as f:
                data = tomllib.load(f)
            ctx.i18n = {
                k: str(v["other"]) for k, v in data.items() if isinstance(v, dict) and "other" in v
            }
        return ctx


def split_front_matter(text: str) -> tuple[dict[str, Any], str, bool]:
    text = text.lstrip("﻿").replace("\r\n", "\n")
    m = FRONT_RE.match(text)
    if not m:
        return {}, text, True
    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return {}, text[m.end() :], False
    return (data if isinstance(data, dict) else {}), text[m.end() :], True


def parse_args(raw: str) -> tuple[list[str], dict[str, str]]:
    pos: list[str] = []
    named: dict[str, str] = {}
    raw = raw.strip().rstrip("/").strip()
    for m in ARG_RE.finditer(raw):
        val = next(g for g in m.groups()[1:] if g is not None)
        if m.group(1):
            named[m.group(1)] = val
        else:
            pos.append(val)
    return pos, named


def _find_close(text: str, start: int, name: str) -> tuple[int, int] | None:
    depth = 1
    for m in TAG_RE.finditer(text, start):
        if m.group(3) != name:
            continue
        depth += -1 if m.group(2) else 1
        if depth == 0:
            return m.start(), m.end()
    return None


def _has_close(text: str, start: int, name: str) -> bool:
    return any(m.group(2) and m.group(3) == name for m in TAG_RE.finditer(text, start))


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").replace("\r\n", "\n")
    except (OSError, UnicodeDecodeError):
        return None


def _minor_shift(minor: str, delta: int) -> str:
    major, _, mn = minor.partition(".")
    return f"{major}.{int(mn) + delta}"


def _fence(body: str, lang: str = "") -> str:
    body = body.strip("\n")
    ticks = "```"
    while ticks in body:
        ticks += "`"
    return f"\n\n{ticks}{lang}\n{body}\n{ticks}\n\n"


def _glossary(ctx: ParseContext, term_id: str) -> tuple[dict[str, Any], str] | None:
    path = ctx.root / "content" / ctx.lang / "docs" / "reference" / "glossary" / f"{term_id}.md"
    raw = _read(path)
    if raw is None:
        ctx.stats.problems[f"missing_glossary:{term_id}"] += 1
        return None
    fm, body, _ = split_front_matter(raw)
    return fm, body


def _include(fname: str, ctx: ParseContext, depth: int) -> str:
    raw = _read(ctx.root / "content" / ctx.lang / "includes" / fname)
    if raw is None:
        ctx.stats.problems[f"missing_include:{fname}"] += 1
        return ""
    _, body, _ = split_front_matter(raw)
    return "\n\n" + render(body, ctx, depth + 1).strip() + "\n\n"


def _glossary_definition(named: dict[str, str], ctx: ParseContext, depth: int) -> str:
    g = _glossary(ctx, named.get("term_id", ""))
    if g is None:
        return ""
    fm, body = g
    if named.get("length") == "all":
        text = render(COMMENT_RE.sub("", body.replace("<!--more-->", "")).strip(), ctx, depth + 1)
    else:
        text = " ".join(str(fm.get("short_description", "")).split())
    prepend = named.get("prepend")
    return f"{prepend} {text}".strip() if prepend else text


def _inline(name: str, args: str, ctx: ParseContext, depth: int) -> str:
    pos, named = parse_args(args)
    st = ctx.stats
    if name in GENERATED:
        st.generated[name] += 1
        return ""
    if name not in INLINE_HANDLED:
        st.unknown[name] += 1
        return ""
    st.handled[name] += 1
    if name == "glossary_tooltip":
        if "text" in named:
            return named["text"]
        term = named.get("term_id", pos[0] if pos else "")
        g = _glossary(ctx, term)
        return str(g[0].get("title", term)) if g else term
    if name == "glossary_definition":
        return _glossary_definition(named, ctx, depth)
    if name == "heading":
        key = pos[0] if pos else ""
        text = ctx.i18n.get(f"{key}_heading")
        if text is None:
            st.problems[f"missing_i18n_heading:{key}"] += 1
            text = key.replace("_", " ").capitalize()
        return f"\n\n## {text}\n\n"
    if name == "feature-state":
        label = ctx.i18n.get("feature_state", "FEATURE STATE:")
        ver = named.get("for_k8s_version", f"v{ctx.minor}")
        gate = (
            f" Feature gate: {named['feature_gate_name']}." if "feature_gate_name" in named else ""
        )
        return f"\n\n{label} Kubernetes {ver} [{named.get('state', '')}]{gate}\n\n"
    if name == "skew":
        key = pos[0] if pos else ""
        if key == "currentPatchVersion":
            return f"{ctx.minor}.x"
        if key.startswith("currentVersionAddMinor"):
            return _minor_shift(ctx.minor, int(pos[1]) if len(pos) > 1 else 0)
        if key in SKEW_SHIFT:
            return _minor_shift(ctx.minor, SKEW_SHIFT[key])
        st.problems[f"skew_key:{key}"] += 1
        return ctx.minor
    if name == "param":
        if pos and pos[0] == "version":
            return f"v{ctx.minor}"
        st.problems[f"param:{pos[0] if pos else ''}"] += 1
        return ""
    if name in {"latest-version", "latest-semver", "release-branch"}:
        return f"release-{ctx.minor}" if name == "release-branch" else f"v{ctx.minor}"
    if name == "version-check":
        return ""
    if name == "include":
        return _include(pos[0] if pos else named.get("file", ""), ctx, depth)
    if name in {"code_sample", "codenew"}:
        fname = named.get("file", pos[0] if pos else "").lstrip("/")
        raw = _read(ctx.root / "content" / ctx.lang / "examples" / fname)
        if raw is None:
            st.problems[f"missing_example:{fname}"] += 1
            return ""
        lang = named.get("language") or EXT_LANG.get(fname.rsplit(".", 1)[-1], "")
        return _fence(raw, lang)
    if name in {"ref", "relref"}:
        return pos[0] if pos else ""
    if name == "figure":
        return (
            "\n\n" + (named.get("caption") or named.get("title") or named.get("alt", "")) + "\n\n"
        )
    return ""


def _block(name: str, args: str, inner: str, ctx: ParseContext, depth: int) -> str:
    pos, named = parse_args(args)
    st = ctx.stats
    if name == "comment":
        st.handled[name] += 1
        return ""
    if name == "mermaid":
        st.handled[name] += 1
        return _fence(inner, "mermaid")
    if name == "highlight":
        st.handled[name] += 1
        return _fence(inner, pos[0] if pos else "")
    body = render(inner, ctx, depth).strip("\n")
    if name in {"note", "caution", "warning"}:
        st.handled[name] += 1
        label = ctx.i18n.get(name, CALLOUT_DEFAULT[name])
        return f"\n\n{label} {body.strip()}\n\n"
    if name == "alert":
        st.handled[name] += 1
        return f"\n\n{named.get('title', 'Note')}: {body.strip()}\n\n"
    if name == "tabs":
        st.handled[name] += 1
        return f"\n\n{body}\n\n"
    if name == "tab":
        st.handled[name] += 1
        title = named.get("name", "").strip() or "Tab"
        if not body.strip() and named.get("include"):
            body = _include(named["include"].strip(), ctx, depth).strip()
        if named.get("codelang"):
            body = _fence(body, named["codelang"]).strip("\n")
        return f"\n\n\x01TAB {title}\x01\n\n{body}\n\n"
    if name == "details":
        st.handled[name] += 1
        return f"\n\n**{named.get('summary', 'Details')}**\n\n{body}\n\n"
    if name == "table":
        st.handled[name] += 1
        cap = f"Table: {named['caption']}\n\n" if named.get("caption") else ""
        return f"\n\n{cap}{body}\n\n"
    if name in BLOCK:  # example, pageinfo, tutorials/*: tags dropped, content kept
        st.handled[name] += 1
        return f"\n\n{body}\n\n"
    st.unknown[name] += 1
    return f"\n\n{body}\n\n"


def render(text: str, ctx: ParseContext, depth: int = 0) -> str:
    """Resolve shortcodes in `text`. Escaped shortcodes (`{{</* x */>}}`) come out literally."""
    if depth > MAX_INCLUDE_DEPTH:
        ctx.stats.problems["include_depth"] += 1
        return ""
    text = ESC_RE.sub(lambda m: f"{ESC_OPEN}{m[1]} {m[2].strip()} {m[3]}{ESC_CLOSE}", text)
    out: list[str] = []
    pos = 0
    while True:
        m = TAG_RE.search(text, pos)
        if m is None:
            out.append(text[pos:])
            break
        out.append(text[pos : m.start()])
        name = m.group(3) or ""
        if m.group(2):  # closing tag without opener
            ctx.stats.problems[f"stray_close:{name}"] += 1
            pos = m.end()
            continue
        is_block = name in BLOCK or (
            name not in INLINE_HANDLED and name not in GENERATED and _has_close(text, m.end(), name)
        )
        close = _find_close(text, m.end(), name) if is_block else None
        if close is not None:
            out.append(_block(name, m.group(4), text[m.end() : close[0]], ctx, depth))
            pos = close[1]
        else:
            out.append(_inline(name, m.group(4), ctx, depth))
            pos = m.end()
    return "".join(out).replace(ESC_OPEN, "{{").replace(ESC_CLOSE, "}}")


def normalize(text: str) -> str:
    """Strip HTML comments outside code, turn tab markers into headings, tidy whitespace."""
    kept: list[str] = []
    level = 0
    in_comment = False
    for ln in scan_lines(text):
        raw = text[ln.start : ln.end].rstrip()
        if ln.fence_id is not None:
            kept.append(raw)
            continue
        if ln.level:
            level = ln.level
        raw = COMMENT_RE.sub("", raw)
        if in_comment:
            if "-->" not in raw:
                continue
            raw = raw.split("-->", 1)[1]
            in_comment = False
        elif "<!--" in raw:
            raw = raw.split("<!--", 1)[0]
            in_comment = True
        m = TAB_MARK.match(raw.strip())
        if m:
            raw = "#" * min(level + 1, 6) + " " + m.group(1)
        kept.append(raw.rstrip())
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(kept))
    return out.replace("\x01", "").strip("\n")


def doc_type_for(front: dict[str, Any], rel_to_docs: PurePosixPath) -> DocType:
    by_name = {
        "concept": DocType.CONCEPT,
        "task": DocType.TASK,
        "reference": DocType.REFERENCE,
        "tutorial": DocType.TUTORIAL,
    }
    ct = str(front.get("content_type", "")).strip().lower()
    if ct in by_name:
        return by_name[ct]
    by_dir = {
        "concepts": DocType.CONCEPT,
        "tasks": DocType.TASK,
        "reference": DocType.REFERENCE,
        "tutorials": DocType.TUTORIAL,
    }
    top = rel_to_docs.parts[0] if rel_to_docs.parts else ""
    return by_dir.get(top, DocType.OTHER)


def url_for(source_path: str, docs_root: str, url_base: str, front: dict[str, Any]) -> str:
    base = url_base.rstrip("/")
    if front.get("url"):
        return base + "/" + str(front["url"]).strip("/") + "/"
    rel = PurePosixPath(source_path).relative_to(PurePosixPath(docs_root).parent)
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] in {"_index", "index"}:
        parts.pop()
    elif front.get("slug") and parts:
        parts[-1] = str(front["slug"])
    return base + "/" + "/".join(parts) + ("/" if parts else "")


def parse_document(
    source_path: str, raw: str, ctx: ParseContext, docs_root: str = "content/en/docs"
) -> DocumentRecord | None:
    """None when the page is a draft or has no text left."""
    front, body, ok = split_front_matter(raw)
    if not ok:
        ctx.stats.problems["bad_front_matter"] += 1
    if front.get("draft") is True:
        return None
    text = normalize(render(body, ctx))
    if not text.strip():
        return None
    rel = PurePosixPath(source_path).relative_to(PurePosixPath(docs_root))
    title = " ".join(str(front.get("title") or "").split())
    if not title:
        first = next((ln.heading for ln in scan_lines(text) if ln.level), "")
        title = first or (rel.parent.name if rel.stem in {"_index", "index"} else rel.stem)
    url = url_for(source_path, docs_root, ctx.url_base, front)
    doc_type = doc_type_for(front, rel)
    digest = hashlib.sha256("\n".join([title, doc_type.value, url, text]).encode()).hexdigest()
    return DocumentRecord(
        k8s_version=ctx.minor,
        source_path=source_path,
        url=url,
        title=title,
        doc_type=doc_type,
        content_hash=digest,
        parsed_text=text,
    )
