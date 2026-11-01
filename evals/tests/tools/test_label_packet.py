"""Merge rules of tools/label_packet.py (no DB, no model)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[3] / "tools" / "label_packet.py"
_spec = importlib.util.spec_from_file_location("label_packet", _PATH)
assert _spec and _spec.loader
lp = importlib.util.module_from_spec(_spec)
sys.modules["label_packet"] = lp
_spec.loader.exec_module(lp)

IDS = ["q1", "q2", "q3"]
A = {"q1": "supported", "q2": "partly", "q3": "supported"}
B = {"q1": "supported", "q2": "supported", "q3": "supported"}


def test_agreed_items_keep_the_passes_label_and_disagreements_take_the_settlement() -> None:
    got = lp.final_labels(A, B, IDS, {"q2": "partly"})
    assert got == [
        ("q1", "supported", lp.AGREED),
        ("q2", "partly", lp.SETTLED),
        ("q3", "supported", lp.AGREED),
    ]


def test_every_disagreement_must_be_settled() -> None:
    with pytest.raises(ValueError, match="q2"):
        lp.final_labels(A, B, IDS, {})


def test_settling_an_agreed_item_is_refused() -> None:
    # It would hide a lead override behind "settled".
    with pytest.raises(ValueError, match="q1"):
        lp.final_labels(A, B, IDS, {"q1": "partly", "q2": "partly"})


def test_a_pass_must_label_every_packet_item_once(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    p.write_text(
        "\n".join(json.dumps({"item_id": i, "label": "supported"}) for i in ["q1", "q2"]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="q3"):
        lp.read_pass(p, IDS)
    p.write_text(json.dumps({"item_id": "q1", "label": "mostly"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown label"):
        lp.read_pass(p, ["q1"])


def test_merge_keeps_other_runs_and_replaces_this_packet(tmp_path: Path) -> None:
    index = tmp_path / "packet.index.json"
    index.write_text(
        json.dumps([{"n": n, "run_id": 11, "item_id": i, "stratum": "factual"}
                    for n, i in enumerate(IDS, 1)]),
        encoding="utf-8",
    )  # fmt: skip
    for name, labels in (("a", A), ("b", B)):
        (tmp_path / f"{name}.jsonl").write_text(
            "\n".join(json.dumps({"item_id": i, "label": v}) for i, v in labels.items()),
            encoding="utf-8",
        )
    out = tmp_path / "faithfulness.jsonl"
    # A label for the same item id from another run must survive.
    out.write_text(
        json.dumps({"run_id": 10, "item_id": "q1", "stratum": "factual", "label": "partly",
                    "labeller": "x", "labelled_at": "t"}) + "\n",
        encoding="utf-8",
    )  # fmt: skip
    args = (index, tmp_path / "a.jsonl", tmp_path / "b.jsonl", out, ["q2=partly"])
    assert lp.merge(*args) == 0
    assert lp.merge(*args) == 0  # idempotent: a second merge replaces, never duplicates
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert sorted((r["run_id"], r["item_id"]) for r in rows) == [
        (10, "q1"), (11, "q1"), (11, "q2"), (11, "q3"),
    ]  # fmt: skip
    assert len(out.with_suffix(".pass-a.jsonl").read_text(encoding="utf-8").splitlines()) == 3
