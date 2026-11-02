"""Per-user daily token budget (Phase 3, T14).

Counted in Redis per user per UTC day so every API process shares it; a day is a calendar day
here (unlike the provider quota ledger, which is a rolling 24 h) because the allowance is our
own fairness rule, not a provider's limit. `reserve` holds a model call's planned tokens before
it runs (INCRBY, then give them back if that passed the allowance), and `record` swaps the hold
for the tokens really used, so parallel turns cannot all pass one check (V07-6). `check` is the
non-binding pre-check the API runs before a turn. An allowance of 0 refuses every call.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sre_core.ports import BudgetExceeded, BudgetGuard
from sre_shared.config import load_agent_config
from sre_shared.models import Claims
from sre_shared.settings import Settings, get_settings

KEY_PREFIX = "sre:budget:"
KEY_TTL_S = 2 * 86400  # outlives the day it counts, so a midnight check never sees a gap


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RedisBudgetGuard:
    def __init__(
        self,
        redis: Any,
        daily_tokens: int,
        *,
        clock: Callable[[], datetime] = _utcnow,
        prefix: str = KEY_PREFIX,
    ) -> None:
        self._redis = redis
        self._daily = daily_tokens
        self._clock = clock
        self._prefix = prefix

    def _key(self, claims: Claims) -> str:
        day = self._clock().astimezone(UTC).strftime("%Y-%m-%d")
        return f"{self._prefix}{day}:{claims.sub}"

    async def used(self, claims: Claims) -> int:
        return int(await self._redis.get(self._key(claims)) or 0)

    async def check(self, claims: Claims, planned_tokens: int) -> None:
        used = await self.used(claims)
        if used + max(planned_tokens, 0) > self._daily:
            raise BudgetExceeded(
                f"daily token allowance reached ({used} of {self._daily} used, "
                f"{planned_tokens} planned); it resets at 00:00 UTC"
            )

    async def reserve(self, claims: Claims, planned_tokens: int) -> None:
        planned = max(int(planned_tokens), 0)
        key = self._key(claims)
        total = int(await self._redis.incrby(key, planned))
        await self._redis.expire(key, KEY_TTL_S)
        if total > self._daily:
            await self._redis.decrby(key, planned)
            raise BudgetExceeded(
                f"daily token allowance reached ({total - planned} of {self._daily} used, "
                f"{planned} planned); it resets at 00:00 UTC"
            )

    async def record(self, claims: Claims, tokens: int, *, reserved: int = 0) -> None:
        delta = max(int(tokens), 0) - max(int(reserved), 0)
        if delta == 0:
            return
        key = self._key(claims)
        await self._redis.incrby(key, delta)
        await self._redis.expire(key, KEY_TTL_S)


def make_budget_guard(
    settings: Settings | None = None,
    *,
    redis: Any = None,
    clock: Callable[[], datetime] = _utcnow,
) -> BudgetGuard:
    """`redis` (an async client) is injected by tests; otherwise it comes from `redis_url`."""
    settings = settings or get_settings()
    daily = load_agent_config(settings.config_dir).budgets.user_daily_tokens
    if redis is None:
        if not settings.redis_url:
            raise RuntimeError(
                "budgets need Redis: set REDIS_URL (lab: redis://127.0.0.1:6379/0, compose "
                "profile agent)"
            )
        import redis.asyncio as aioredis

        redis = aioredis.Redis.from_url(settings.redis_url)
    return RedisBudgetGuard(redis, daily, clock=clock)
