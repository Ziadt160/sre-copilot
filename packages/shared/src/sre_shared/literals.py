"""Literal strings worth matching exactly in Kubernetes questions: error and status names,
multi-word CLI flags, and API field paths.

One definition serves the golden-set picker (an exact-string item must contain such a
literal) and the query router (a query with one leans on the keyword leg). Note the
circularity when measuring the router on golden v1: its exact-string stratum is defined by
these same patterns, so the router is also measured on factual items that contain literals.
"""

from __future__ import annotations

import re

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


def find_exact_strings(text: str) -> list[tuple[str, str]]:
    """All (kind, string) candidates in text, in order of first appearance, deduplicated."""
    found: dict[str, tuple[int, str]] = {}
    for kind, pattern in _KINDS:
        for m in pattern.finditer(text):
            s = m.group(0)
            if s not in found or m.start() < found[s][0]:
                found[s] = (m.start(), kind)
    return [(kind, s) for s, (_, kind) in sorted(found.items(), key=lambda kv: kv[1][0])]


def has_literal(text: str) -> bool:
    """True when the text contains at least one error name, multi-word flag or field path."""
    return any(p.search(text) for _, p in _KINDS)
