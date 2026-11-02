"""Per-user daily token budget over fakeredis."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import fakeredis
import pytest

from sre_core.agent.ports import BudgetExceeded
from sre_core.budget.factory import RedisBudgetGuard, make_budget_guard
from sre_shared.models import Claims
from sre_shared.settings import Settings

ALICE = Claims(sub="alice")
BOB = Claims(sub="bob")


class Now:
    def __init__(self) -> None:
        self.t = datetime(2026, 11, 2, 10, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.t


def guard(daily: int = 1000, now: Now | None = None) -> RedisBudgetGuard:
    return RedisBudgetGuard(fakeredis.FakeAsyncRedis(), daily, clock=now or Now())


async def test_user_over_allowance_is_refused_before_any_call() -> None:
    g = guard(1000)
    await g.check(ALICE, 600)
    await g.record(ALICE, 600)
    await g.check(ALICE, 400)  # exactly to the limit is fine
    with pytest.raises(BudgetExceeded):
        await g.check(ALICE, 401)


async def test_other_users_are_unaffected() -> None:
    g = guard(1000)
    await g.record(ALICE, 1000)
    with pytest.raises(BudgetExceeded):
        await g.check(ALICE, 1)
    await g.check(BOB, 1000)


async def test_budget_is_shared_between_guards_on_one_redis() -> None:
    server = fakeredis.FakeServer()
    now = Now()
    a = RedisBudgetGuard(fakeredis.FakeAsyncRedis(server=server), 100, clock=now)
    b = RedisBudgetGuard(fakeredis.FakeAsyncRedis(server=server), 100, clock=now)
    await a.record(ALICE, 90)
    with pytest.raises(BudgetExceeded):
        await b.check(ALICE, 20)


async def test_new_utc_day_starts_fresh() -> None:
    now = Now()
    g = guard(100, now)
    await g.record(ALICE, 100)
    with pytest.raises(BudgetExceeded):
        await g.check(ALICE, 1)
    now.t += timedelta(days=1)
    await g.check(ALICE, 100)


async def test_single_call_larger_than_the_allowance_is_refused_and_zero_allowance_refuses() -> (
    None
):
    with pytest.raises(BudgetExceeded):
        await guard(100).check(ALICE, 101)
    with pytest.raises(BudgetExceeded):
        await guard(0).check(ALICE, 1)


async def test_nonpositive_records_are_ignored() -> None:
    g = guard(100)
    await g.record(ALICE, 0)
    await g.record(ALICE, -50)
    assert await g.used(ALICE) == 0


async def test_factory_uses_configured_allowance_and_requires_redis() -> None:
    g = make_budget_guard(
        Settings(_env_file=None, redis_url=None), redis=fakeredis.FakeAsyncRedis()
    )
    await g.check(ALICE, 20000)  # config/agent.yaml: user_daily_tokens 20000
    with pytest.raises(BudgetExceeded):
        await g.check(ALICE, 20001)
    with pytest.raises(RuntimeError, match="REDIS_URL"):
        make_budget_guard(Settings(_env_file=None, redis_url=None))
