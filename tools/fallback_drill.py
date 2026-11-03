"""Fallback drill on the real model router: take the first provider away and watch the chain.

The routing tests prove the chain on fake transports; this drill runs the production router
(`make_model_router`, config/agent.yaml + config/limits.yaml, real keys) against the real
providers, three ways:

    exhausted    the first provider's daily quota is spent (an in-memory ledger says so), so it
                 is refused locally, with no call made and no quota used, and the next provider
                 must serve.
    unreachable  the first provider's connection fails (a transport that refuses), as when it is
                 down or blocked.
    all-down     every provider refuses: the router raises AllProvidersFailed and the answer is
                 the extractive one (quoted passages with citations, no model).

    uv run python tools/fallback_drill.py --role answer --questions 3 --out docs/fallback-drill.md

Each line records which model served, how long it took and what failed first. Hosted providers
further down the chain are called for real, so their own limits apply.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx2

from sre_core.agent.models import AllProvidersFailed, extractive_answer, make_model_router
from sre_core.agent.ports import ModelRole
from sre_shared.aio import run
from sre_shared.config import load_limits
from sre_shared.llm import MemoryLedger, ModelSpec
from sre_shared.models import DocType, RetrievedChunk


def _refusing_client(spec: ModelSpec) -> httpx2.AsyncClient:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("drill: provider unreachable", request=request)

    return httpx2.AsyncClient(transport=httpx2.MockTransport(refuse))


def _context(quote: str, source: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=1, chunk_key="drill", source_path=source, k8s_version="1.35",
            doc_type=DocType.CONCEPT, title="t", section_path=[], url="https://kubernetes.io/",
            header="h", body=quote, score=1.0,
        )
    ]  # fmt: skip


async def drill(role: ModelRole, n_questions: int) -> list[dict[str, Any]]:
    from sre_shared.golden_io import golden_path, load_golden

    golden = [g for g in load_golden(golden_path("v1")) if g.status == "accepted"][:n_questions]
    limits = load_limits()
    probe = make_model_router(ledger=MemoryLedger())
    chain = probe.chain(role)
    first = chain[0]
    rows: list[dict[str, Any]] = []

    for scenario in ("exhausted", "unreachable", "all-down"):
        ledger = MemoryLedger()
        if scenario == "exhausted":
            provider, model = first.split(":", 1)
            tpd = limits.model_limits(provider, model).tpd or 0
            ledger.add(first, tpd)

        def factory(spec: ModelSpec, scenario: str = scenario) -> httpx2.AsyncClient | None:
            if scenario == "all-down" or (scenario == "unreachable" and spec.label == first):
                return _refusing_client(spec)
            return None

        router = make_model_router(ledger=ledger, http_client_factory=factory)
        for g in golden:
            messages = [{"role": "user", "content": f"Answer briefly: {g.question}"}]
            started = time.perf_counter()
            try:
                res = await router.complete(role, messages, max_tokens=256)
                served, note = res.model, ""
            except AllProvidersFailed as exc:
                quote = g.gold_passages[0].quote if g.gold_passages else ""
                src = g.gold_passages[0].source_path if g.gold_passages else ""
                text, _ = extractive_answer(g.question, _context(quote, src))
                served = "extractive" if text else "none"
                note = "; ".join(f"{label} {why}" for label, why in exc.attempts)
            rows.append({
                "scenario": scenario, "item": g.id, "served_by": served,
                "ms": round((time.perf_counter() - started) * 1000), "failed": note,
            })  # fmt: skip
    return [{"chain": chain}, *rows]


def render(rows: Sequence[dict[str, Any]]) -> str:
    head, *body = rows
    lines = [
        f"Chain for this role: {' -> '.join(head['chain'])} -> extractive answer",
        "",
        "| scenario | item | served by | ms | attempts that failed (all-down) |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['scenario']} | {r['item']} | {r['served_by']} | {r['ms']} | {r['failed'] or '-'} |"
        for r in body
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fallback_drill")
    parser.add_argument("--role", default="answer")
    parser.add_argument("--questions", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = render(run(drill(args.role, args.questions)))
    print(report)
    if args.out is not None:
        args.out.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
