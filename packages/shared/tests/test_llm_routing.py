"""RedisLedger (shared rolling-24 h counts) and the Workers AI base_url placeholder."""

from __future__ import annotations

import fakeredis

from sre_shared.llm import WINDOW_S, RedisLedger, parse_model_spec
from sre_shared.settings import Settings


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_counts_roll_off_after_24_hours() -> None:
    clock = Clock()
    ledger = RedisLedger(fakeredis.FakeRedis(), clock=clock)
    ledger.add("groq:m", 100)
    clock.t += 3600
    ledger.add("groq:m", 50)
    assert ledger.used("groq:m") == 150
    clock.t += WINDOW_S - 3600 + 1  # the first entry is now 24 h and a second old
    assert ledger.used("groq:m") == 50
    clock.t += 3600
    assert ledger.used("groq:m") == 0


def test_two_instances_share_counts_and_labels_are_separate() -> None:
    server = fakeredis.FakeServer()
    clock = Clock()
    a = RedisLedger(fakeredis.FakeRedis(server=server), clock=clock)
    b = RedisLedger(fakeredis.FakeRedis(server=server), clock=clock)
    a.add("groq:m", 10)
    b.add("groq:m", 5)
    b.add("nvidia:n", 7)
    assert a.used("groq:m") == b.used("groq:m") == 15
    assert a.used("nvidia:n") == 7
    assert a.used("never") == 0


def test_two_calls_in_the_same_instant_both_count() -> None:
    ledger = RedisLedger(fakeredis.FakeRedis(), clock=Clock())
    ledger.add("m", 3)
    ledger.add("m", 3)
    assert ledger.used("m") == 6


def test_cloudflare_account_id_placeholder() -> None:
    name = "@cf/openai/gpt-oss-20b"
    unset = Settings(_env_file=None, cloudflare_account_id=None)
    spec = parse_model_spec(f"cloudflare_workers_ai:{name}", unset)
    assert "{cloudflare_account_id}" in spec.base_url  # left visible: the router skips it
    s = Settings(_env_file=None, cloudflare_account_id="abc", cloudflare_api_token="tok")
    spec = parse_model_spec(f"cloudflare_workers_ai:{name}", s)
    assert spec.base_url == "https://api.cloudflare.com/client/v4/accounts/abc/ai/v1"
    assert spec.api_key == "tok"
    assert "tok" not in repr(spec)
