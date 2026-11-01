"""The load sweep's summariser (infra/load/sweep.py): CSV parsing, saturation, markdown."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from sre_shared.settings import REPO_ROOT

_spec = importlib.util.spec_from_file_location("sweep", REPO_ROOT / "infra" / "load" / "sweep.py")
assert _spec and _spec.loader
sweep = importlib.util.module_from_spec(_spec)
sys.modules["sweep"] = sweep
_spec.loader.exec_module(sweep)

HEADER = "Type,Name,Request Count,Failure Count,Median Response Time,Average Response Time,"
HEADER += "Min Response Time,Max Response Time,Average Content Size,Requests/s,Failures/s,"
HEADER += "50%,66%,75%,80%,90%,95%,98%,99%,99.9%,99.99%,100%"


def _csv(name: str, reqs: int, fails: int, p50: int, p95: int, p99: int, rps: float) -> str:
    row = f"GET,{name},{reqs},{fails},{p50},{p50},1,{p99},100,{rps},0,"
    row += f"{p50},{p50},{p50},{p50},{p50},{p95},{p95},{p99},{p99},{p99},{p99}"
    agg = row.replace(f"GET,{name}", ",Aggregated", 1)
    return "\n".join([HEADER, row, agg]) + "\n"


def level(users: int, p95: float, reqs: int = 1000, fails: int = 0) -> object:
    return sweep.Level(users, reqs, fails, 50.0, p95 / 2, p95, p95 * 1.5)


def test_parse_stats_csv_reads_the_aggregated_row() -> None:
    lv = sweep.parse_stats_csv(10, _csv("GET /search", 500, 5, 40, 120, 200, 33.5))
    assert (lv.users, lv.requests, lv.failures) == (10, 500, 5)
    assert (lv.p50_ms, lv.p95_ms, lv.p99_ms, lv.rps) == (40, 120, 200, 33.5)
    assert lv.error_rate == pytest.approx(0.01)


def test_parse_stats_csv_without_aggregated_row_fails() -> None:
    with pytest.raises(ValueError):
        sweep.parse_stats_csv(1, HEADER + "\nGET,/x,1,0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1\n")


def test_saturation_is_first_level_where_p95_doubles() -> None:
    levels = [level(1, 50), level(5, 70), level(10, 99), level(25, 100), level(50, 400)]
    assert sweep.saturation_users(levels) == 25  # 100 >= 2 x 50


def test_saturation_on_error_rate_even_if_latency_is_flat() -> None:
    levels = [level(1, 50), level(5, 55), level(10, 60, reqs=1000, fails=11)]
    assert sweep.saturation_users(levels) == 10
    assert sweep.saturation_users([level(1, 50, fails=10)]) is None  # exactly 1%: not over


def test_saturation_not_reached() -> None:
    assert sweep.saturation_users([level(1, 50), level(5, 60), level(10, 70)]) is None
    assert sweep.saturation_users([]) is None


def test_render_markdown_has_a_row_per_level_and_the_saturation_line() -> None:
    md = sweep.render_markdown("with-replica", [level(1, 50), level(5, 200)], 30)
    assert "with-replica" in md and "| 1 |" in md and "| 5 |" in md
    assert "Saturation point: 5 users" in md


def test_locustfile_lists_its_tasks() -> None:
    import subprocess

    out = subprocess.run(
        [sys.executable, "-m", "locust", "-f", str(Path(sweep.LOCUSTFILE)), "--list"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert "CoreUser" in out.stdout or "search" in out.stdout
