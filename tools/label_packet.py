"""Model labelling of stored answers: export a blind packet, then merge two passes.

The owner cannot judge Kubernetes answers, so faithfulness labels come from two independent
model passes under evals/labels/faithfulness-rubric.md, with disagreements settled by the
lead against the sources (the procedure of decision 27). This tool holds the two mechanical
steps:

    export   the labelling sample of a stored answer run as a packet: question, answer and
             the sources exactly as the model saw them (same grouping and numbering as the
             prompt). No judge scores, no gold passages, no strata.
    merge    two passes (JSON Lines: item_id, label, ...) into the repo's label file, after
             printing their agreement and disagreements. Each disagreement needs a
             `--settle item_id=label`; the passes are also kept in label format beside the
             output (`<out>.pass-a.jsonl`, `.pass-b.jsonl`).

    uv run python tools/label_packet.py export --database-url ... --run 10 --out packet.md
    uv run python tools/label_packet.py merge --index packet.index.json --pass-a a.jsonl \
        --pass-b b.jsonl --out evals/labels/faithfulness-v1.jsonl --settle fac-0017=partly

The merge rules are pure functions (tested in evals/tests/tools/test_label_packet.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PASS_LABELLERS = {"pass-a": "model:claude-opus-5.5", "pass-b": "model:claude-sonnet-5.5"}
AGREED = "model:claude-opus-5.5+claude-sonnet-5.5 (agreed)"
SETTLED = "model:claude-opus-5.5 (lead, settled)"


def read_pass(path: Path, ids: Sequence[str]) -> dict[str, str]:
    """item_id -> label of one pass; every packet item exactly once, known labels only."""
    from sre_evals.answers.stats import LABELS

    got: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["label"] not in LABELS:
                raise ValueError(f"{path}: unknown label {row['label']!r} for {row['item_id']}")
            if row["item_id"] in got:
                raise ValueError(f"{path}: {row['item_id']} labelled twice")
            got[row["item_id"]] = row["label"]
    if set(got) != set(ids):
        raise ValueError(f"{path}: items differ from the packet: {sorted(set(got) ^ set(ids))}")
    return got


def disagreements(a: Mapping[str, str], b: Mapping[str, str], ids: Sequence[str]) -> list[str]:
    return [i for i in ids if a[i] != b[i]]


def final_labels(
    a: Mapping[str, str], b: Mapping[str, str], ids: Sequence[str], settled: Mapping[str, str]
) -> list[tuple[str, str, str]]:
    """(item_id, label, labeller): the passes' label where they agree, the settled one where
    they do not. A settlement for an item the passes agree on is refused (it would hide a
    lead override); a missing settlement is refused too."""
    dis = set(disagreements(a, b, ids))
    extra = sorted(set(settled) - dis)
    if extra:
        raise ValueError(f"settled items the passes agree on: {extra}")
    missing = sorted(dis - set(settled))
    if missing:
        raise ValueError(f"settle these disagreements first: {missing}")
    return [(i, settled[i], SETTLED) if i in dis else (i, a[i], AGREED) for i in ids]


def agreement_line(a: Mapping[str, str], b: Mapping[str, str], ids: Sequence[str]) -> str:
    from sre_evals.answers.stats import accuracy, cohens_kappa, to_binary

    la, lb = [a[i] for i in ids], [b[i] for i in ids]
    ba, bb = [to_binary(x) for x in la], [to_binary(x) for x in lb]

    def f(x: float | None) -> str:
        return "-" if x is None else f"{x:.3f}"

    return (
        f"n={len(ids)}  3-class: accuracy {f(accuracy(la, lb))} kappa {f(cohens_kappa(la, lb))}"
        f" | binary: accuracy {f(accuracy(ba, bb))} kappa {f(cohens_kappa(ba, bb))}"
    )


def export(dsn: str, run_id: int, n: int, seed: int, out: Path) -> int:
    from sre_core.answer.prompts import group_sources, render_source
    from sre_evals.answers.label import LabelItem, labellable, stratified_sample
    from sre_shared.db import connect_sync
    from sre_shared.models import AnswerResult

    with connect_sync(dsn) as conn:
        rows = conn.execute(
            "select item_id, details from public.eval_scores where run_id = %s"
            " and metric = 'answer_detail' order by item_id",
            (run_id,),
        ).fetchall()
    if not rows:
        print(f"run {run_id} has no stored answers", file=sys.stderr)
        return 2
    items = [LabelItem.from_detail(run_id, r["item_id"], r["details"]) for r in rows]
    results = {r["item_id"]: AnswerResult.model_validate(r["details"]["result"]) for r in rows}
    chosen = stratified_sample(labellable(items), n, seed)
    parts: list[str] = []
    index: list[dict[str, Any]] = []
    for pos, it in enumerate(chosen, 1):
        groups = group_sources(results[it.item_id].context)
        sources = "\n\n".join(render_source(i, g) for i, g in enumerate(groups, 1))
        parts.append(
            f"======== ITEM {pos} of {len(chosen)} | id: {it.item_id} ========\n\n"
            f"QUESTION:\n{it.question}\n\nANSWER:\n{it.answer}\n\n"
            f"SOURCES (exactly as the model saw them):\n{sources}\n"
        )
        index.append({"n": pos, "run_id": run_id, "item_id": it.item_id, "stratum": it.stratum})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(parts), encoding="utf-8", newline="\n")
    out.with_suffix(".index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"{len(chosen)} items -> {out}")
    return 0


def merge(
    index_path: Path, pass_a: Path, pass_b: Path, out: Path | None, settle: Sequence[str]
) -> int:
    from sre_evals.answers.label import Label, read_labels, write_labels

    index = {r["item_id"]: r for r in json.loads(index_path.read_text(encoding="utf-8"))}
    ids = sorted(index, key=lambda i: index[i]["n"])
    a, b = read_pass(pass_a, ids), read_pass(pass_b, ids)
    print(agreement_line(a, b, ids))
    for i in disagreements(a, b, ids):
        print(f"  {i} ({index[i]['stratum']}): A={a[i]} B={b[i]}")
    if out is None:
        return 0
    settled = dict(s.split("=", 1) for s in settle)
    now = datetime.now(UTC).isoformat(timespec="seconds")

    def lab(i: str, label: str, who: str) -> Label:
        return Label(index[i]["run_id"], i, index[i]["stratum"], label, who, now)

    keys = {(index[i]["run_id"], i) for i in ids}

    def replace(path: Path, new: list[Label]) -> int:
        """Rewrite `path` with this packet's labels; other runs' lines stay. Returns kept."""
        kept = [x for x in read_labels(path) if (x.run_id, x.item_id) not in keys]
        write_labels(path, [*kept, *new])
        return len(kept)

    for name, p in (("pass-a", a), ("pass-b", b)):
        replace(
            out.with_suffix(f".{name}.jsonl"), [lab(i, p[i], PASS_LABELLERS[name]) for i in ids]
        )
    final = [lab(i, label, who) for i, label, who in final_labels(a, b, ids, settled)]
    kept = replace(out, final)
    print(f"wrote {len(final)} labels to {out} (kept {kept} from other runs)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="write the blind labelling packet of a stored run")
    ex.add_argument("--database-url", required=True)
    ex.add_argument("--run", type=int, required=True)
    ex.add_argument("--sample", type=int, default=30)
    ex.add_argument("--seed", type=int, default=1)
    ex.add_argument("--out", type=Path, required=True)
    mg = sub.add_parser("merge", help="compare two passes; with --out, write the labels")
    mg.add_argument("--index", type=Path, required=True)
    mg.add_argument("--pass-a", type=Path, required=True)
    mg.add_argument("--pass-b", type=Path, required=True)
    mg.add_argument("--out", type=Path, default=None)
    mg.add_argument("--settle", nargs="*", default=[], help="item_id=label for each disagreement")
    args = ap.parse_args(argv)
    if args.cmd == "export":
        return export(args.database_url, args.run, args.sample, args.seed, args.out)
    return merge(args.index, args.pass_a, args.pass_b, args.out, args.settle)


if __name__ == "__main__":
    raise SystemExit(main())
