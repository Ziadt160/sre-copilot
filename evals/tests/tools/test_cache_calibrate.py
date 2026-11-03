"""The threshold sweep's arithmetic, on hand-made similarities."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "cache_calibrate", ROOT / "tools" / "cache_calibrate.py"
)
assert spec is not None and spec.loader is not None
cc = importlib.util.module_from_spec(spec)
sys.modules["cache_calibrate"] = cc
spec.loader.exec_module(cc)

SIM = {"p1": 0.95, "p2": 0.88, "c1": 0.90, "x1": 0.70, "lit": 0.99}
PAIRS = [
    cc.Pair("p1", "paraphrase", "a", "a'"),
    cc.Pair("p2", "paraphrase", "b", "b'"),
    cc.Pair("c1", "contrast", "c", "c-other"),
    cc.Pair("x1", "cross", "d", "e"),
    cc.Pair("lit", "contrast", "f --flag-a", "f --flag-b"),  # near-identical, literals differ
]


def run(thresholds: list[float]) -> list[dict[str, object]]:
    return cc.sweep(PAIRS, lambda p: SIM[p.pair_id], lambda p: p.pair_id != "lit", thresholds)


def test_hits_and_wrong_hits_per_threshold() -> None:
    rows = run([0.85, 0.90, 0.91])
    assert [r["hits"] for r in rows] == [2, 1, 1]
    assert [r["wrong_hits"] for r in rows] == [1, 1, 0]
    assert rows[0]["wrong_ids"] == ["c1"]
    assert rows[2]["hit_rate"] == 0.5


def test_the_literal_guard_keeps_a_near_identical_question_out() -> None:
    (row,) = run([0.98])
    assert row["wrong_hits"] == 0  # 'lit' is at 0.99 but its flags differ


def test_recommend_picks_the_lowest_safe_threshold() -> None:
    best = cc.recommend(run([0.85, 0.90, 0.91, 0.96]))
    assert best is not None and best["threshold"] == 0.91
    assert cc.recommend(run([0.85])) is None


def test_only_reviewed_pairs_are_kept() -> None:
    rows = [{"id": "q-p1", "kind": "paraphrase", "base": "q", "text": "q'"},
            {"id": "q-c", "kind": "contrast", "base": "q", "text": "r"}]  # fmt: skip
    kept = cc.accepted_pairs(rows, {"q-p1": True, "q-c": False})
    assert [p.pair_id for p in kept] == ["q-p1"]
    assert len(cc.cross_pairs({"a": "x", "b": "y", "c": "z"})) == 3
