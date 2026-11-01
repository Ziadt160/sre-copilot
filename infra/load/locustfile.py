"""Locust users for the core API: golden questions against GET /search (and optionally
POST /answer).

    uv run locust -f infra/load/locustfile.py --host http://127.0.0.1:8000

Environment:
    LOAD_GOLDEN        golden items file (default evals/golden/v1/items.jsonl; accepted items only)
    LOAD_MODE          retrieval mode sent as `mode` (hybrid | dense | keyword; default hybrid)
    LOAD_K             `k` for /search (default 5)
    LOAD_THINK_S       fixed think time between requests in seconds (default 0 = closed loop)
    LOAD_ANSWER_WEIGHT weight of POST /answer next to GET /search's 10 (default 0 = search only).
                       /answer calls a model: run it against a stub or a local model, never a
                       hosted one.
    LOAD_TOKEN         optional bearer token (omit for the anonymous, public-docs caller)

`sweep.py` drives this file headless across concurrency levels.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from locust import HttpUser, constant, task

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(os.environ.get("LOAD_GOLDEN", REPO_ROOT / "evals" / "golden" / "v1" / "items.jsonl"))
MODE = os.environ.get("LOAD_MODE", "hybrid")
K = int(os.environ.get("LOAD_K", "5"))
THINK_S = float(os.environ.get("LOAD_THINK_S", "0"))
ANSWER_WEIGHT = int(os.environ.get("LOAD_ANSWER_WEIGHT", "0"))
TOKEN = os.environ.get("LOAD_TOKEN")


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Accepted golden items as {question, versions, doc_types}; filters come with the item."""
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("status") != "accepted":
                continue
            filters = item.get("filters") or {}
            out.append(
                {
                    "question": item["question"],
                    "versions": filters.get("k8s_versions") or [],
                    "doc_types": filters.get("doc_types") or [],
                }
            )
    if not out:
        raise SystemExit(f"{path}: no accepted golden items")
    return out


QUESTIONS = load_questions(GOLDEN)
HEADERS = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}


class CoreUser(HttpUser):
    wait_time = constant(THINK_S)

    @task(10)
    def search(self) -> None:
        q = random.choice(QUESTIONS)  # noqa: S311 - load shape, not security
        params: list[tuple[str, str]] = [("q", q["question"]), ("k", str(K)), ("mode", MODE)]
        params += [("version", v) for v in q["versions"]]
        params += [("doc_type", d) for d in q["doc_types"]]
        self.client.get("/search", params=params, headers=HEADERS, name="GET /search")

    @task(ANSWER_WEIGHT)
    def answer(self) -> None:
        q = random.choice(QUESTIONS)  # noqa: S311
        body = {
            "question": q["question"],
            "filters": {
                "k8s_versions": q["versions"] or None,
                "doc_types": q["doc_types"] or None,
            },
        }
        self.client.post("/answer", json=body, headers=HEADERS, name="POST /answer")
