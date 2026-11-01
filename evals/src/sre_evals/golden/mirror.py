"""Mirror accepted golden items into a LangSmith dataset (one-way, idempotent by example id)."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sre_shared.golden_io import golden_version, load_golden
from sre_shared.models import GoldenItem
from sre_shared.settings import get_settings

_NAMESPACE = uuid.UUID("5f0a6c3e-2b7d-4a55-9a43-6a1d6b1f9c11")


def example_id(dataset: str, item_id: str) -> str:
    """Stable id so re-running the mirror updates rather than duplicates."""
    return str(uuid.uuid5(_NAMESPACE, f"{dataset}:{item_id}"))


def to_example(dataset: str, item: GoldenItem, version: str) -> dict[str, Any]:
    return {
        "id": example_id(dataset, item.id),
        "inputs": {
            "question": item.question,
            "filters": item.filters.model_dump(mode="json") if item.filters else None,
        },
        "outputs": {
            "expected_answer": item.expected_answer,
            "gold_passages": [p.model_dump(mode="json") for p in item.gold_passages],
        },
        "metadata": {
            "item_id": item.id,
            "stratum": item.stratum,
            "origin": item.origin,
            "reviewer": item.reviewer,
            "golden_version": version,
        },
    }


@dataclass
class MirrorResult:
    skipped_reason: str | None = None
    mirrored: int = 0


def _default_client(api_key: str) -> Any:
    from langsmith import Client

    return Client(api_key=api_key)


def mirror_dataset(
    path: Path,
    dataset: str,
    *,
    api_key: str | None = None,
    client_factory: Callable[[str], Any] = _default_client,
) -> MirrorResult:
    """Upsert accepted items. With no API key nothing is sent and the reason is returned."""
    if api_key is None:
        secret = get_settings().langsmith_api_key
        api_key = secret.get_secret_value() if secret else None
    if not api_key:
        return MirrorResult(skipped_reason="no LangSmith API key set (LANGSMITH_API_KEY); skipped")
    items = load_golden(path, accepted_only=True)
    if not items:
        return MirrorResult(skipped_reason="no accepted items in the file; nothing to mirror")
    version = golden_version(path)
    client = client_factory(api_key)
    if not client.has_dataset(dataset_name=dataset):
        client.create_dataset(
            dataset,
            description=f"SRE Copilot golden set, mirrored from {path.parent.name} ({version})",
        )
    client.create_examples(
        dataset_name=dataset, examples=[to_example(dataset, i, version) for i in items]
    )
    return MirrorResult(mirrored=len(items))
