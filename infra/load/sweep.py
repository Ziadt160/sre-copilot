"""Concurrency sweep: run `locustfile.py` headless at 1/5/10/25/50 users and summarise.

    uv run python infra/load/sweep.py --host http://127.0.0.1:8000 --label primary-only

Writes `<out>/<label>.json` and `<out>/<label>.md` (p50/p95/p99, RPS and error rate per level,
plus the saturation point). Saturation is the first level where p95 is at least twice the p95 of
the lowest level, or where the error rate exceeds 1%.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

LOCUSTFILE = Path(__file__).with_name("locustfile.py")
DEFAULT_LEVELS = (1, 5, 10, 25, 50)
P95_FACTOR = 2.0
ERROR_RATE_LIMIT = 0.01


@dataclass(frozen=True)
class Level:
    users: int
    requests: int
    failures: int
    rps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    @property
    def error_rate(self) -> float:
        return self.failures / self.requests if self.requests else 0.0


def parse_stats_csv(users: int, text: str) -> Level:
    """The `Aggregated` row of Locust's `<prefix>_stats.csv`."""
    for row in csv.DictReader(io.StringIO(text)):
        if row.get("Name") == "Aggregated":
            return Level(
                users=users,
                requests=int(row["Request Count"]),
                failures=int(row["Failure Count"]),
                rps=float(row["Requests/s"]),
                p50_ms=float(row["50%"]),
                p95_ms=float(row["95%"]),
                p99_ms=float(row["99%"]),
            )
    raise ValueError("no Aggregated row in the Locust stats CSV")


def saturation_users(levels: list[Level]) -> int | None:
    """Users at the first level that doubled the baseline p95 or exceeded the error limit."""
    if not levels:
        return None
    base = levels[0].p95_ms
    for lv in levels:
        if lv.error_rate > ERROR_RATE_LIMIT:
            return lv.users
        if lv is not levels[0] and base > 0 and lv.p95_ms >= P95_FACTOR * base:
            return lv.users
    return None


def render_markdown(label: str, levels: list[Level], duration_s: int) -> str:
    sat = saturation_users(levels)
    lines = [
        f"### Load sweep: {label}",
        "",
        f"{duration_s} s per level, closed loop. Saturation = first level with p95 >= "
        f"{P95_FACTOR:g}x the lowest level's p95, or errors > {ERROR_RATE_LIMIT:.0%}.",
        "",
        "| users | requests | RPS | p50 ms | p95 ms | p99 ms | error rate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for lv in levels:
        lines.append(
            f"| {lv.users} | {lv.requests} | {lv.rps:.1f} | {lv.p50_ms:.0f} | {lv.p95_ms:.0f} |"
            f" {lv.p99_ms:.0f} | {lv.error_rate:.2%} |"
        )
    lines += ["", f"Saturation point: {sat} users" if sat else "Saturation point: not reached"]
    return "\n".join(lines) + "\n"


def run_level(host: str, users: int, duration_s: int, spawn_rate: float, workdir: Path) -> Level:
    prefix = workdir / f"u{users}"
    cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(LOCUSTFILE),
        "--headless",
        "--host",
        host,
        "-u",
        str(users),
        "-r",
        str(spawn_rate),
        "-t",
        f"{duration_s}s",
        "--csv",
        str(prefix),
        "--only-summary",
    ]
    # Locust exits non-zero when any request failed; the CSV still holds the numbers.
    subprocess.run(cmd, check=False)  # noqa: S603
    return parse_stats_csv(users, Path(f"{prefix}_stats.csv").read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--label", default="primary-only")
    ap.add_argument("--levels", default=",".join(map(str, DEFAULT_LEVELS)))
    ap.add_argument("--duration", type=int, default=30, help="seconds per level")
    ap.add_argument("--spawn-rate", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=Path("data/load"))
    args = ap.parse_args()

    users = [int(x) for x in args.levels.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    levels: list[Level] = []
    with tempfile.TemporaryDirectory() as tmp:
        for u in users:
            lv = run_level(args.host, u, args.duration, args.spawn_rate, Path(tmp))
            levels.append(lv)
            print(
                f"{u:>3} users: {lv.rps:.1f} rps, p95 {lv.p95_ms:.0f} ms, err {lv.error_rate:.2%}"
            )
    payload = {
        "label": args.label,
        "host": args.host,
        "duration_s": args.duration,
        "levels": [asdict(lv) | {"error_rate": lv.error_rate} for lv in levels],
        "saturation_users": saturation_users(levels),
    }
    (args.out / f"{args.label}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.out / f"{args.label}.md").write_text(
        render_markdown(args.label, levels, args.duration), encoding="utf-8"
    )
    print(f"wrote {args.out / (args.label + '.json')} and .md")


if __name__ == "__main__":
    main()
