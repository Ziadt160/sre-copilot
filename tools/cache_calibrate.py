"""Calibrate the semantic answer cache's similarity threshold (`cache.semantic_min_similarity`).

A semantic hit serves a stored answer to a different question, so the threshold must let real
paraphrases through and keep every question with a different answer out. The decision depends
only on the cosine similarity of the two questions' embeddings and on the literal guard (the same
error names, flags and versions on both sides), so it is measured offline with the production
embedder and the cache's own `normalise_question` / `literal_tokens`; no database is needed.

    generate  a local model writes, per golden v1 question, two paraphrases (same answer) and one
              contrast question (same topic, different answer) -> a pairs file of drafts.
    sweep     with a review file (one model pass per pair, as for the golden sets), keeps the
              accepted pairs and reports, per threshold, the share of paraphrases that would hit
              and the number of different questions that would wrongly hit. Negatives are the
              accepted contrast questions plus every pair of distinct golden questions.

    uv run python tools/cache_calibrate.py generate --out evals/cache/pairs-v1.jsonl
    uv run python tools/cache_calibrate.py sweep --pairs evals/cache/pairs-v1.jsonl \
        --review evals/cache/pairs-v1.review.jsonl --out docs/cache-calibration.md

The sweep's arithmetic is a pure function (tested in evals/tests/tools/test_cache_calibrate.py).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

THRESHOLDS = [round(0.80 + 0.01 * i, 2) for i in range(19)]  # 0.80 .. 0.98

GENERATE_PROMPT = """\
You help test a cache of answers to Kubernetes questions. For the question below, write:
- "paraphrases": two rewordings an engineer might type that ask exactly the same thing and need
  exactly the same answer. Keep every literal (flags, field names, error strings, versions)
  character for character.
- "contrast": one question on the same topic that needs a DIFFERENT answer (another object,
  another field, the opposite operation, another version).
Reply with one JSON object: {"paraphrases": ["...", "..."], "contrast": "..."}

Question: """


@dataclass(frozen=True)
class Pair:
    pair_id: str
    kind: str  # paraphrase (should hit) | contrast or cross (must not hit)
    stored: str  # the question whose answer is in the cache
    asked: str  # the new question looked up


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def accepted_pairs(rows: Sequence[Mapping[str, Any]], review: Mapping[str, bool]) -> list[Pair]:
    """Generated pairs the review kept; a pair the review did not cover is dropped."""
    return [
        Pair(str(r["id"]), str(r["kind"]), str(r["base"]), str(r["text"]))
        for r in rows
        if review.get(str(r["id"])) is True
    ]


def cross_pairs(questions: Mapping[str, str]) -> list[Pair]:
    """Every pair of distinct golden questions: different answers, both directions alike."""
    return [
        Pair(f"{a}|{b}", "cross", questions[a], questions[b])
        for a, b in itertools.combinations(sorted(questions), 2)
    ]


def sweep(
    pairs: Sequence[Pair],
    similarity: Callable[[Pair], float],
    same_literals: Callable[[Pair], bool],
    thresholds: Sequence[float] = THRESHOLDS,
) -> list[dict[str, Any]]:
    """Per threshold: paraphrase hit rate, wrong hits (a negative served), and which ones."""
    sims = {p.pair_id: similarity(p) for p in pairs}
    lits = {p.pair_id: same_literals(p) for p in pairs}
    positives = [p for p in pairs if p.kind == "paraphrase"]
    negatives = [p for p in pairs if p.kind != "paraphrase"]
    rows = []
    for t in thresholds:
        hits = [p for p in positives if lits[p.pair_id] and sims[p.pair_id] >= t]
        wrong = [p for p in negatives if lits[p.pair_id] and sims[p.pair_id] >= t]
        rows.append({
            "threshold": t,
            "hit_rate": len(hits) / len(positives) if positives else 0.0,
            "hits": len(hits),
            "positives": len(positives),
            "wrong_hits": len(wrong),
            "negatives": len(negatives),
            "wrong_ids": [p.pair_id for p in wrong],
        })  # fmt: skip
    return rows


def recommend(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The lowest threshold with no wrong hit: the most paraphrase hits that stay safe."""
    safe = [r for r in rows if r["wrong_hits"] == 0]
    return min(safe, key=lambda r: r["threshold"]) if safe else None


def render(
    rows: Sequence[Mapping[str, Any]], current: float, best: Mapping[str, Any] | None
) -> str:
    lines = [
        "| threshold | paraphrase hit rate | wrong hits |",
        "|---|---|---|",
    ]
    for r in rows:
        mark = " (current)" if abs(r["threshold"] - current) < 1e-9 else ""
        lines.append(
            f"| {r['threshold']:.2f}{mark} | {r['hit_rate']:.3f} ({r['hits']}/{r['positives']}) "
            f"| {r['wrong_hits']} / {r['negatives']} |"
        )
    pick = (
        f"Lowest safe threshold: **{best['threshold']:.2f}** (paraphrase hit rate "
        f"{best['hit_rate']:.3f}, 0 wrong hits)."
        if best
        else "No threshold in the range avoids every wrong hit."
    )
    return "\n".join([*lines, "", pick])


# --- I/O ------------------------------------------------------------------------------------


def _golden_questions() -> dict[str, str]:
    from sre_shared.golden_io import golden_path, load_golden

    return {g.id: g.question for g in load_golden(golden_path("v1")) if g.status == "accepted"}


def generate(out: Path, model: str) -> None:
    from sre_shared.llm import ChatClient, parse_model_spec

    client = ChatClient(parse_model_spec(model))
    rows: list[dict[str, Any]] = []
    for gid, question in sorted(_golden_questions().items()):
        res = client.complete(
            [{"role": "user", "content": GENERATE_PROMPT + question}],
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        try:
            data = json.loads(res.text)
            paraphrases = [str(x) for x in data["paraphrases"]][:2]
            contrast = str(data["contrast"])
        except (ValueError, KeyError, TypeError):
            print(f"{gid}: unparseable reply, skipped", file=sys.stderr)
            continue
        for i, text in enumerate(paraphrases, start=1):
            rows.append({"id": f"{gid}-p{i}", "kind": "paraphrase", "base_id": gid,
                         "base": question, "text": text, "generator": model})  # fmt: skip
        rows.append({"id": f"{gid}-c", "kind": "contrast", "base_id": gid, "base": question,
                     "text": contrast, "generator": model})  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} draft pairs -> {out}")


def run_sweep(pairs_path: Path, review_path: Path, out: Path | None) -> None:
    from sre_core.cache.factory import literal_tokens, normalise_question
    from sre_shared.config import load_agent_config
    from sre_shared.embed import get_embedder

    review = {str(r["id"]): bool(r["keep"]) for r in read_jsonl(review_path)}
    pairs = accepted_pairs(read_jsonl(pairs_path), review) + cross_pairs(_golden_questions())
    embedder = get_embedder("gte-small")
    # Stored rows are embedded from the normalised question, lookups from the question as asked.
    texts = sorted({normalise_question(p.stored) for p in pairs} | {p.asked for p in pairs})
    vecs = np.asarray(embedder.embed_queries(texts), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index = {t: vecs[i] for i, t in enumerate(texts)}

    def similarity(p: Pair) -> float:
        return float(index[normalise_question(p.stored)] @ index[p.asked])

    def same_literals(p: Pair) -> bool:
        return literal_tokens(normalise_question(p.stored)) == literal_tokens(p.asked)

    rows = sweep(pairs, similarity, same_literals)
    current = load_agent_config().cache.semantic_min_similarity
    report = render(rows, current, recommend(rows))
    print(report)
    if out is not None:
        out.write_text(report + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cache_calibrate")
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--model", default="ollama:qwen2.5:7b")
    s = sub.add_parser("sweep")
    s.add_argument("--pairs", type=Path, required=True)
    s.add_argument("--review", type=Path, required=True)
    s.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "generate":
        generate(args.out, args.model)
    else:
        run_sweep(args.pairs, args.review, args.out)


if __name__ == "__main__":
    main()
