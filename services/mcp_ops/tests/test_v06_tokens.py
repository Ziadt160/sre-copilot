"""V06 (adversarial): tokens core would never mint, and one token's call budget under parallel use.

The T11 suite covers no token, expired, wrong secret, wrong audience, wrong issuer, a non-JWT,
alg none and a lifetime over 300 s. These are the variants it does not try: another HMAC
algorithm with the real secret, an `iat` in the future (which would stretch a 300 s token into a
long-lived one), an audience list, claims outside their bounds. Key confusion needs a public key
to confuse; mcp-ops accepts HS256 only (`verify_mcp_token`: algorithms=["HS256"]), so the HS384 and
HS512 cases are the whole of that attack here.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any

import jwt
import pytest
from fastmcp.exceptions import ToolError

if TYPE_CHECKING:
    from conftest import Rig  # type: ignore[attr-defined, unused-ignore]
    from sre_mcp_ops.backends.simulator import SimulatorBackend

# The test secret is shorter than PyJWT recommends for HS384/HS512; that is the point of signing
# with it, so the warning is noise here.
pytestmark = pytest.mark.filterwarnings("ignore::jwt.warnings.InsecureKeyLengthWarning")

SECRET = "mcp-test-secret-0123456789abcdef0123456789"
WEB = {"namespace": "shop", "deployment": "web"}


def signed(algorithm: str = "HS256", *, drop: tuple[str, ...] = (), **override: Any) -> str:
    """A token signed with the real secret, so only the claim or header under test is wrong."""
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": "sre-core", "aud": "mcp-ops", "sub": "alice", "scopes": ["viewer", "operator"],
        "namespaces": ["shop"], "iat": now, "exp": now + 100, "jti": uuid.uuid4().hex,
        "max_calls": 5,
    }  # fmt: skip
    claims.update(override)
    for name in drop:
        claims.pop(name)
    return jwt.encode(claims, SECRET, algorithm=algorithm)


def bad_tokens() -> dict[str, str]:
    now = int(time.time())
    return {
        "HS384 with the real secret": signed("HS384"),
        "HS512 with the real secret": signed("HS512"),
        # 300 s apart, but an hour from now: PyJWT refuses an iat beyond now + leeway
        # (jwt/api_jwt.py _validate_iat), so the lifetime cap cannot be dodged this way.
        "iat in the future": signed(iat=now + 3600, exp=now + 3900),
        "audience list naming mcp-ops": signed(aud=["mcp-ops", "somebody-else"]),
        "max_calls above the cap": signed(max_calls=11),
        "max_calls zero": signed(max_calls=0),
        "unknown scope": signed(scopes=["viewer", "root"]),
        "turn zero": signed(turn=0),
        "no jti": signed(drop=("jti",)),
        "no sub": signed(drop=("sub",)),
        "empty sub": signed(sub=""),
        "no exp": signed(drop=("exp",)),
        "exp before iat": signed(iat=now, exp=now - 1),
    }


BAD_TOKENS = [
    "HS384 with the real secret",
    "HS512 with the real secret",
    "iat in the future",
    "audience list naming mcp-ops",
    "max_calls above the cap",
    "max_calls zero",
    "unknown scope",
    "turn zero",
    "no jti",
    "no sub",
    "empty sub",
    "no exp",
    "exp before iat",
]


@pytest.mark.parametrize("name", BAD_TOKENS)
async def test_tokens_core_never_mints_are_refused_before_any_tool(rig: Rig, name: str) -> None:
    token = bad_tokens()[name]
    with pytest.raises(Exception):
        async with rig.client(token) as c:
            await c.call_tool("list_pods", {"namespace": "shop"})
    assert await rig.rows() == []


async def test_parallel_calls_on_one_token_share_one_budget(rig: Rig) -> None:
    token = rig.token(max_calls=1)

    async def one() -> Any:
        async with rig.client(token) as c:  # a connection of its own each time
            return await c.call_tool("list_pods", {"namespace": "shop"})

    out = await asyncio.gather(*(one() for _ in range(6)), return_exceptions=True)
    ok = [o for o in out if not isinstance(o, BaseException)]
    refused = [o for o in out if isinstance(o, ToolError) and "max_calls" in str(o)]
    assert len(ok) == 1 and len(refused) == 5
    assert len(await rig.rows("status = 'refused'")) == 5


async def test_a_one_call_write_token_cannot_be_raced_into_two_writes(
    rig: Rig, sim: SimulatorBackend
) -> None:
    t = await rig.thread()
    appr = await rig.approval(t, "restart_deployment", WEB)
    token = rig.token(thread_id=t, approval_id=appr, max_calls=1)
    out = await asyncio.gather(
        *(rig.write("restart_deployment", WEB, token) for _ in range(4)), return_exceptions=True
    )
    assert sum(1 for o in out if isinstance(o, dict)) == 1
    assert len(sim.mutation_log()) == 1
