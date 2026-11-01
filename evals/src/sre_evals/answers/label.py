"""Human faithfulness labelling of stored answers: a stratified sample, a resumable terminal UI.

The labeller reads the question, the answer and exactly the passages the model was given, and
presses `s` (every claim supported by the passages), `p` (partly), `n` (not supported). No
Kubernetes knowledge is needed: it is a reading task.

Label file (JSON Lines, one decision per line, rewritten atomically after every decision):

    {"run_id": 12, "item_id": "fac-0003", "stratum": "factual", "label": "supported",
     "labeller": "ziad", "labelled_at": "2026-11-03T10:00:00+00:00"}

`label` is one of supported / partly / not_supported. `calibrate` joins these lines to the
stored `answer_detail` rows by (run_id, item_id), so the file stays small.
"""

from __future__ import annotations

import json
import os
import random
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from psycopg import AsyncConnection
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from sre_evals.answers.stats import KEY_TO_LABEL, LABELS
from sre_evals.golden.review import Terminal

KEYS_HELP = "[s]upported [p]artly [n]ot supported [u]ndo e[x]it and save"
MARKER = re.compile(r"\[\d+\]")


@dataclass(frozen=True)
class Passage:
    chunk_id: int
    source_path: str
    k8s_version: str
    title: str
    url: str
    body: str
    header: str = ""

    @property
    def text(self) -> str:
        """The passage as the model saw it: header, blank line, body."""
        return f"{self.header}\n\n{self.body}" if self.header else self.body


@dataclass(frozen=True)
class LabelItem:
    run_id: int
    item_id: str
    stratum: str
    question: str
    answer: str
    passages: tuple[Passage, ...]
    abstained: bool = False

    @property
    def key(self) -> tuple[int, str]:
        return (self.run_id, self.item_id)

    @classmethod
    def from_detail(cls, run_id: int, item_id: str, d: dict[str, Any]) -> LabelItem:
        return cls(
            run_id=run_id,
            item_id=item_id,
            stratum=str(d.get("stratum", item_id.split("-", 1)[0])),
            question=str(d["question"]),
            answer=str(d["answer"]),
            abstained=bool(d.get("abstained", False)),
            passages=tuple(
                Passage(
                    chunk_id=int(p["chunk_id"]),
                    source_path=p["source_path"],
                    k8s_version=p["k8s_version"],
                    title=p["title"],
                    url=p["url"],
                    body=p["body"],
                    header=p.get("header", ""),
                )
                for p in d["context"]
            ),
        )


async def load_details(
    conn: AsyncConnection[Any], run_id: int, item_ids: Sequence[str] | None = None
) -> list[LabelItem]:
    """Stored answers of a run (`answer_detail` rows), in item-id order."""
    cur = await conn.execute(
        "select item_id, details from public.eval_scores where run_id = %s"
        " and metric = 'answer_detail' order by item_id",
        [run_id],
    )
    wanted = set(item_ids) if item_ids is not None else None
    return [
        LabelItem.from_detail(run_id, r["item_id"], r["details"])
        for r in await cur.fetchall()
        if wanted is None or r["item_id"] in wanted
    ]


def labellable(items: Sequence[LabelItem]) -> list[LabelItem]:
    """Answers worth a faithfulness label: a non-empty answer given at least one passage
    (an abstention or an empty answer makes no claims to check)."""
    return [i for i in items if i.answer.strip() and i.passages and not i.abstained]


def stratified_sample(items: Sequence[LabelItem], n: int, seed: int) -> list[LabelItem]:
    """`n` items, allocated to strata in proportion to their size (largest remainder; every
    stratum gets at least one when n allows), random within a stratum, then shuffled so the
    labeller does not see one stratum at a time. Deterministic for a given pool, n and seed."""
    if n >= len(items):
        chosen = list(items)
    else:
        rng = random.Random(seed)  # noqa: S311
        by: dict[str, list[LabelItem]] = {}
        for it in sorted(items, key=lambda i: i.key):
            by.setdefault(it.stratum, []).append(it)
        strata = sorted(by)
        total = len(items)
        quota = {s: len(by[s]) * n / total for s in strata}
        alloc = {s: int(quota[s]) for s in strata}
        if n >= len(strata):
            for s in strata:
                alloc[s] = max(alloc[s], 1)
        while sum(alloc.values()) > n:  # the floor of 1 can overshoot: trim the largest
            big = max((s for s in strata if alloc[s] > 1), key=lambda s: (alloc[s] - quota[s], s))
            alloc[big] -= 1
        for s in sorted(strata, key=lambda s: (-(quota[s] - alloc[s]), s)):
            if sum(alloc.values()) >= n:
                break
            if alloc[s] < len(by[s]):
                alloc[s] += 1
        chosen = [it for s in strata for it in rng.sample(by[s], alloc[s])]
        rng.shuffle(chosen)
        return chosen
    random.Random(seed).shuffle(chosen)  # noqa: S311
    return chosen


# -- label file ----------------------------------------------------------------------------


@dataclass
class Label:
    run_id: int
    item_id: str
    stratum: str
    label: str
    labeller: str
    labelled_at: str


def read_labels(path: Path) -> list[Label]:
    if not path.exists():
        return []
    out: list[Label] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            if d["label"] not in LABELS:
                raise ValueError(f"{path}: unknown label {d['label']!r}")
            out.append(Label(**d))
    return out


def write_labels(path: Path, labels: Sequence[Label]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(lab.__dict__, sort_keys=True) + "\n" for lab in labels)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


# -- session -------------------------------------------------------------------------------


@dataclass
class LabelSession:
    items: list[LabelItem]
    out_path: Path
    labeller: str
    now: Callable[[], datetime] = lambda: datetime.now(UTC)
    decided: dict[tuple[int, str], Label] = field(default_factory=dict)
    queue: list[LabelItem] = field(default_factory=list)
    message: str = ""
    _order: list[tuple[int, str]] = field(default_factory=list)
    _kept: list[Label] = field(default_factory=list)  # labels outside this sample

    @classmethod
    def open(
        cls,
        items: list[LabelItem],
        out_path: Path,
        labeller: str,
        now: Callable[[], datetime] | None = None,
    ) -> LabelSession:
        s = cls(items=items, out_path=out_path, labeller=labeller)
        if now is not None:
            s.now = now
        keys = {i.key for i in items}
        for lab in read_labels(out_path):
            key = (lab.run_id, lab.item_id)
            if key not in keys:
                s._kept.append(lab)  # another run or sample: never dropped from the file
                continue
            if key in s.decided:  # a duplicated line: the last one wins, one undo entry
                s._order.remove(key)
            s.decided[key] = lab
            s._order.append(key)
        s.queue = [i for i in items if i.key not in s.decided]
        return s

    @property
    def current(self) -> LabelItem | None:
        return self.queue[0] if self.queue else None

    @property
    def finished(self) -> bool:
        return not self.queue

    def _save(self) -> None:
        write_labels(self.out_path, [*self._kept, *(self.decided[k] for k in self._order)])

    def label(self, label: str) -> None:
        if label not in LABELS:
            raise ValueError(f"unknown label {label!r}")
        item = self.current
        if item is None:
            raise RuntimeError("nothing left to label")
        self.message = ""
        self.queue.pop(0)
        self.decided[item.key] = Label(
            item.run_id, item.item_id, item.stratum, label, self.labeller, self.now().isoformat()
        )
        self._order.append(item.key)
        self._save()

    def undo(self) -> None:
        if not self._order:
            self.message = "nothing to undo"
            return
        key = self._order.pop()
        del self.decided[key]
        item = next(i for i in self.items if i.key == key)
        self.queue.insert(0, item)
        self._save()
        self.message = f"undid {key[1]}"

    def progress(self) -> dict[str, Counter[str]]:
        out: dict[str, Counter[str]] = {}
        for lab in self.decided.values():
            out.setdefault(lab.stratum, Counter())[lab.label] += 1
        for it in self.queue:
            out.setdefault(it.stratum, Counter())["pending"] += 1
        return out


def render(session: LabelSession, console: Console) -> None:
    done, total = len(session.decided), len(session.items)
    console.print(f"labelled {done} of {total}", highlight=False)
    it = session.current
    if it is None:
        console.print("[green]every sampled answer has a label[/green]")
        return
    body: list[Text] = [
        Text.assemble(("Question: ", "bold cyan"), it.question),
        Text(""),
        Text.assemble(("Answer: ", "bold green"), it.answer),
        Text(""),
        Text("Passages the model was given (judge the answer against ONLY these):", style="bold"),
    ]
    for pos, p in enumerate(it.passages, start=1):
        body.append(Text(""))
        body.append(
            Text.assemble(
                (f"[{pos}] ", "bold yellow"),
                f"chunk {p.chunk_id}  {p.title}  v{p.k8s_version}  {p.source_path}",
            )
        )
        body.append(Text(p.text))
    console.print(
        Panel(
            Group(*body), title=f"{it.item_id} [{it.stratum}] run {it.run_id}", subtitle=KEYS_HELP
        )
    )
    if session.message:
        console.print(f"[red]{session.message}[/red]")


def run_label(session: LabelSession, term: Terminal, console: Console) -> None:
    while not session.finished:
        render(session, console)
        key = term.read_key().lower()
        if key == "x":
            break
        if key in KEY_TO_LABEL:
            session.label(KEY_TO_LABEL[key])
        elif key == "u":
            session.undo()
        else:
            session.message = f"unknown key {key!r}"
    render(session, console)
