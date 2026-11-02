"""Names T14's `sre_core.agent.models` exports, with a stand-in until that module merges.

The graph needs `AllProvidersFailed` (the fallback chain is exhausted) and `extractive_answer`
(the last resort: quoted passages, no model). T14 owns both; importing them dynamically keeps
this package importable and type-checked before T14 lands, and the stand-ins follow T14's
published interface (bucket T14-1) so behaviour does not change when the real ones appear.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from typing import Any

from sre_core.answer.prompts import group_sources
from sre_shared.models import RetrievedChunk


class _StandInAllProvidersFailed(RuntimeError):
    """Every model in the chain failed (stand-in for sre_core.agent.models.AllProvidersFailed)."""


def _stand_in_extractive(question: str, context: Sequence[RetrievedChunk]) -> tuple[str, list[int]]:
    groups = group_sources(context)[:3]
    parts = []
    for n, members in enumerate(groups, start=1):
        text = " ".join(m.body for m in members if m.body).strip()[:600]
        if text:
            parts.append(f'"{text}" [{n}]')
    return ("\n\n".join(parts), list(range(1, len(parts) + 1)))


def _load() -> tuple[type[Exception], Callable[[str, Sequence[RetrievedChunk]], Any]]:
    try:
        mod = importlib.import_module("sre_core.agent.models")
    except ImportError:
        return _StandInAllProvidersFailed, _stand_in_extractive
    return mod.AllProvidersFailed, mod.extractive_answer


AllProvidersFailed, _extractive = _load()


def extractive_answer(question: str, context: Sequence[RetrievedChunk]) -> tuple[str, list[int]]:
    """Quoted passages with `[n]` markers numbered like `group_sources(context)`."""
    text, used = _extractive(question, context)
    return str(text), list(used)
