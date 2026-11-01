"""Terminal review of candidate golden items: accept, reject, rewrite, edit, trim, skip, undo.

`ReviewSession` holds the state and every decision; `run_review` is the thin key loop. Both
input and the clock are injectable so tests drive a whole session with scripted keys.
`apply_decisions` feeds a decisions file through the same session methods, so a
non-interactive review (for example by a model) obeys the same rules.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sre_evals.golden.validate import exact_string_of, normalize_ws
from sre_shared.golden_io import load_golden, write_golden
from sre_shared.models import GoldenItem

REWRITE_TARGET = 0.30
KEYS_HELP = "[a]ccept [r]eject [q]uestion rewrite [e]dit answer [t]rim quote [s]kip [u]ndo e[x]it"


class Terminal(Protocol):
    def read_key(self) -> str: ...
    def read_line(self, prompt: str) -> str: ...


def stdlib_read_key() -> str:
    """One keypress without Enter, using only the standard library."""
    if sys.platform == "win32":
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):  # function/arrow key prefix: swallow the second code
            msvcrt.getwch()
            return ""
        return ch
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class StdTerminal:
    def read_key(self) -> str:
        return stdlib_read_key()

    def read_line(self, prompt: str) -> str:
        return input(prompt)


@dataclass
class _Undo:
    item_id: str
    previous: GoldenItem | None  # the decided copy that was replaced, if any


@dataclass
class ReviewSession:
    candidates: list[GoldenItem]
    out_path: Path
    reviewer: str
    now: Callable[[], datetime] = lambda: datetime.now(UTC)
    decided: dict[str, GoldenItem] = field(default_factory=dict)
    queue: list[str] = field(default_factory=list)
    work: GoldenItem | None = None
    message: str = ""
    _undo: list[_Undo] = field(default_factory=list)
    _by_id: dict[str, GoldenItem] = field(default_factory=dict)

    @classmethod
    def open(
        cls,
        in_path: Path,
        out_path: Path,
        reviewer: str,
        now: Callable[[], datetime] | None = None,
    ) -> ReviewSession:
        candidates = load_golden(in_path)
        s = cls(candidates=candidates, out_path=out_path, reviewer=reviewer)
        if now is not None:
            s.now = now
        if out_path.exists():
            for it in load_golden(out_path):
                s.decided[it.id] = it
        s._by_id = {c.id: c for c in candidates}
        s.queue = [c.id for c in candidates if c.id not in s.decided]
        s._load_next()
        return s

    # -- queue ----------------------------------------------------------------------

    def _load_next(self) -> None:
        self.work = self._by_id[self.queue[0]].model_copy(deep=True) if self.queue else None

    @property
    def current(self) -> GoldenItem | None:
        return self.work

    @property
    def finished(self) -> bool:
        return self.work is None

    def _save(self) -> None:
        write_golden(self.out_path, self.decided.values())

    # -- decisions ------------------------------------------------------------------

    def _commit(self, status: str) -> None:
        assert self.work is not None
        item = self.work
        item.status = status  # type: ignore[assignment]
        item.reviewer = self.reviewer
        item.reviewed_at = self.now()
        self._undo.append(_Undo(item.id, self.decided.get(item.id)))
        self.decided[item.id] = item
        self.queue.pop(0)
        self._save()
        self._load_next()

    def accept(self) -> None:
        w = self._need()
        if not w.gold_passages:
            self.message = "cannot accept: no gold passage"
            return
        self._commit("accepted")

    def reject(self) -> None:
        self._need()
        self._commit("rejected")

    def skip(self) -> None:
        self._need()
        if len(self.queue) > 1:
            self.queue.append(self.queue.pop(0))
        self._load_next()
        self.message = "skipped (will come back at the end)"

    def undo(self) -> None:
        if not self._undo:
            self.message = "nothing to undo"
            return
        u = self._undo.pop()
        if u.previous is None:
            self.decided.pop(u.item_id, None)
        else:
            self.decided[u.item_id] = u.previous
        if u.item_id not in self.queue:
            self.queue.insert(0, u.item_id)
        else:
            self.queue.remove(u.item_id)
            self.queue.insert(0, u.item_id)
        self._save()
        self._load_next()
        self.message = f"undid decision on {u.item_id}"

    # -- edits to the working copy (committed by accept) ----------------------------------

    def set_question(self, text: str) -> bool:
        w = self._need()
        text = text.strip()
        if not text:
            self.message = "empty question ignored"
            return False
        s = exact_string_of(w)
        if w.stratum == "exact_string" and s and s not in text:
            self.message = f"rewrite must keep the exact string {s!r}"
            return False
        if text != w.question:
            w.question = text
            w.origin = "rewritten"
        return True

    def set_answer(self, text: str) -> bool:
        w = self._need()
        text = text.strip()
        if not text:
            self.message = "empty answer ignored"
            return False
        w.expected_answer = text
        return True

    def trim_quote(self, text: str, index: int = 0) -> bool:
        w = self._need()
        quote = w.gold_passages[index].quote
        span = _locate_span(quote, text)
        if span is None:
            self.message = "trim rejected: text is not a sub-span of the quote"
            return False
        w.gold_passages[index].quote = span
        return True

    def _need(self) -> GoldenItem:
        if self.work is None:
            raise RuntimeError("no item under review")
        self.message = ""
        return self.work

    # -- progress ---------------------------------------------------------------------

    def progress(self) -> dict[str, Counter[str]]:
        out: dict[str, Counter[str]] = {}
        for it in self.decided.values():
            c = out.setdefault(it.stratum, Counter())
            c[it.status] += 1
            if it.status == "accepted" and it.origin == "rewritten":
                c["rewritten"] += 1
        for iid in self.queue:
            out.setdefault(self._by_id[iid].stratum, Counter())["pending"] += 1
        return out

    def rewritten_share(self) -> float:
        acc = [i for i in self.decided.values() if i.status == "accepted"]
        return sum(i.origin == "rewritten" for i in acc) / len(acc) if acc else 0.0


class ReviewDecision(BaseModel):
    """One line of a decisions file: the same edits the interactive keys make."""

    id: str
    action: Literal["accept", "reject"]
    question: str | None = None
    expected_answer: str | None = None
    quote: str | None = None  # verbatim sub-span of the (single) gold quote to keep
    reason: str = ""


@dataclass
class ApplyReport:
    accepted: int = 0
    rejected: int = 0
    errors: list[str] = field(default_factory=list)


def apply_decisions(session: ReviewSession, decisions: list[ReviewDecision]) -> ApplyReport:
    """Apply decisions in order. An edit the session refuses (a rewrite without the exact
    string, a trim that is not a sub-span) leaves that item undecided and is reported."""
    report = ApplyReport()
    for d in decisions:
        if d.id not in session.queue:
            report.errors.append(f"{d.id}: not pending (unknown or already decided)")
            continue
        session.queue.remove(d.id)
        session.queue.insert(0, d.id)
        session._load_next()
        edits = [
            (d.question, session.set_question),
            (d.expected_answer, session.set_answer),
            (d.quote, session.trim_quote),
        ]
        refused = [
            session.message for value, edit in edits if value is not None and not edit(value)
        ]
        if refused:
            report.errors.append(f"{d.id}: {'; '.join(refused)}")
            session.queue.append(session.queue.pop(0))
            session._load_next()
            continue
        work = session._need()
        work.notes = d.reason or work.notes
        if d.action == "accept":
            session.accept()
            report.accepted += 1
        else:
            session.reject()
            report.rejected += 1
    return report


def _locate_span(quote: str, text: str) -> str | None:
    """`text` as a verbatim sub-span of `quote`, tolerating different whitespace; else None."""
    text = text.strip()
    if not text:
        return None
    if text in quote:
        return text
    words = normalize_ws(text).split(" ")
    m = re.search(r"\s+".join(re.escape(w) for w in words), quote)
    return m.group(0) if m else None


# --- rendering -------------------------------------------------------------------------


def render(session: ReviewSession, console: Console) -> None:
    w = session.current
    table = Table(title="progress", show_header=True, header_style="bold")
    for col in ("stratum", "accepted", "rejected", "pending", "rewritten"):
        table.add_column(col, justify="right" if col != "stratum" else "left")
    for stratum, c in sorted(session.progress().items()):
        table.add_row(
            stratum, str(c["accepted"]), str(c["rejected"]), str(c["pending"]), str(c["rewritten"])
        )
    share = session.rewritten_share()
    flag = "ok" if share >= REWRITE_TARGET else f"below {REWRITE_TARGET:.0%} target"
    console.print(table)
    console.print(f"rewritten by hand: {share:.0%} of accepted ({flag})")
    if w is None:
        console.print("[green]queue empty: every candidate has a decision[/green]")
        return
    p = w.gold_passages[0] if w.gold_passages else None
    exact = exact_string_of(w)
    body: list[Text] = [
        Text.assemble(("Q: ", "bold cyan"), w.question),
        Text.assemble(("A: ", "bold green"), w.expected_answer or "(none)"),
    ]
    if exact:
        body.append(Text.assemble(("exact string: ", "bold yellow"), exact))
    if p:
        url = next((t[4:] for t in w.tags if t.startswith("url:")), "-")
        body.append(
            Text.assemble(("source: ", "bold"), f"{p.source_path}  v{p.k8s_version}  {url}")
        )
        quote = Text(p.quote)
        if exact:
            quote.highlight_words([exact], style="bold yellow")
        body.append(quote)
    console.print(
        Panel(
            Group(*body),
            title=f"{w.id} [{w.stratum}] origin={w.origin} model={w.synth_model or '-'}",
            subtitle=KEYS_HELP,
        )
    )
    if session.message:
        console.print(f"[red]{session.message}[/red]")


def run_review(session: ReviewSession, term: Terminal, console: Console) -> None:
    handlers: dict[str, Callable[[], None]] = {
        "a": session.accept,
        "r": session.reject,
        "s": session.skip,
        "u": session.undo,
    }
    while not session.finished:
        render(session, console)
        key = term.read_key().lower()
        if key == "x":
            break
        if key in handlers:
            handlers[key]()
        elif key == "q":
            session.set_question(term.read_line("new question (empty keeps): "))
        elif key == "e":
            session.set_answer(term.read_line("new answer (empty keeps): "))
        elif key == "t":
            session.trim_quote(term.read_line("sub-span of the quote to keep: "))
        else:
            session.message = f"unknown key {key!r}"
    render(session, console)
