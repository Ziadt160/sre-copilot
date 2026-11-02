"""Phase 3 contracts: tool arguments, idempotency, approval rules, the mcp-ops token, SSE events."""

from __future__ import annotations

import uuid

import jwt
import pydantic
import pytest

from sre_shared.contracts import agent as ag
from sre_shared.contracts import ops
from sre_shared.contracts.export import SCHEMA_PATH, render

POLICY = ops.NamespacePolicy(
    allowed=frozenset({"shop", "payments"}), protected=frozenset({"payments"})
)
THREAD = uuid.UUID("00000000-0000-0000-0000-000000000001")
SECRET = "test-secret-at-least-32-bytes-long!!"


def test_the_idempotency_key_ignores_argument_order_and_filled_defaults() -> None:
    a = ops.idempotency_key(THREAD, 3, "get_logs", {"namespace": "shop", "pod": "web-1"})
    b = ops.idempotency_key(
        THREAD, 3, "get_logs", {"pod": "web-1", "tail_lines": 200, "namespace": "shop"}
    )
    assert a == b and len(a) == 64
    assert a != ops.idempotency_key(THREAD, 4, "get_logs", {"namespace": "shop", "pod": "web-1"})
    assert a != ops.idempotency_key(None, 3, "get_logs", {"namespace": "shop", "pod": "web-1"})


@pytest.mark.parametrize(
    "args",
    [
        {"namespace": "Shop", "deployment": "web"},  # upper case is not a Kubernetes name
        {"namespace": "shop; rm -rf /", "deployment": "web"},
        {"namespace": "shop", "deployment": "web", "force": True},  # unknown field
        {"namespace": "shop"},
    ],
)
def test_arguments_that_are_not_kubernetes_names_or_unknown_are_refused(args: dict) -> None:
    with pytest.raises(pydantic.ValidationError):
        ops.parse_args("restart_deployment", args)


def test_unknown_tools_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown tool"):
        ops.tool_kind("delete_namespace")
    with pytest.raises(ValueError, match="unknown tool"):
        ops.parse_args("delete_namespace", {})


def test_who_must_approve() -> None:
    def need(tool: str, args: dict, route: str = "action") -> str:
        return ops.required_approval(tool, ops.parse_args(tool, args), POLICY, route=route)  # type: ignore[arg-type]

    assert need("list_pods", {"namespace": "payments"}) == "none"
    assert need("restart_deployment", {"namespace": "shop", "deployment": "web"}) == "operator"
    assert need("restart_deployment", {"namespace": "payments", "deployment": "api"}) == "admin"
    assert (
        need("scale_deployment", {"namespace": "shop", "deployment": "web", "replicas": 3})
        == "operator"
    )
    assert (
        need("scale_deployment", {"namespace": "shop", "deployment": "web", "replicas": 0})
        == "admin"
    )
    incident = {"severity": "sev2", "title": "checkout down", "summary": "5xx since 10:00"}
    assert need("create_incident", incident, route="escalate") == "none"
    assert need("create_incident", incident, route="action") == "operator"


def test_scopes_and_namespaces_are_checked_on_both_sides() -> None:
    restart = ops.parse_args("restart_deployment", {"namespace": "shop", "deployment": "web"})
    assert ops.allowed("restart_deployment", restart, ["viewer"], ["shop"], POLICY) is not None
    assert (
        ops.allowed("restart_deployment", restart, ["viewer", "operator"], ["shop"], POLICY) is None
    )
    assert "may not act" in (
        ops.allowed("restart_deployment", restart, ["operator"], ["payments"], POLICY) or ""
    )
    other = ops.parse_args("list_pods", {"namespace": "kube-system"})
    assert "allow-list" in (
        ops.allowed("list_pods", other, ["viewer"], ["kube-system"], POLICY) or ""
    )


def test_rollback_needs_an_explicit_revision_and_rates_high_risk() -> None:
    with pytest.raises(pydantic.ValidationError):
        ops.parse_args("rollback_deployment", {"namespace": "shop", "deployment": "web"})
    rb = ops.parse_args(
        "rollback_deployment", {"namespace": "shop", "deployment": "web", "to_revision": 3}
    )
    assert ops.risk_of("rollback_deployment", rb, POLICY) == "high"
    assert "revision 3" in ops.expected_effect("rollback_deployment", rb)


def _mint(**kw: object) -> str:
    base: dict = {"secret": SECRET, "sub": "alice", "scopes": ["viewer", "operator"], "namespaces": ["shop"],
                  "thread_id": THREAD, "turn": 2}  # fmt: skip
    base.update(kw)
    return ops.mint_mcp_token(**base)


def test_the_mcp_token_round_trips_and_carries_the_turn() -> None:
    claims = ops.verify_mcp_token(_mint(), SECRET)
    assert (claims.sub, claims.thread_id, claims.turn, claims.max_calls) == ("alice", THREAD, 2, 2)
    assert claims.aud == "mcp-ops" and claims.exp - claims.iat == 120


def test_a_forged_expired_or_misaddressed_token_is_refused() -> None:
    with pytest.raises(jwt.InvalidSignatureError):
        ops.verify_mcp_token(_mint(), "another-secret-also-32-bytes-long!!")
    with pytest.raises(jwt.ExpiredSignatureError):
        ops.verify_mcp_token(_mint(now=1_000_000), SECRET)
    wrong_aud = jwt.encode({"aud": "other", "iss": "sre-core", "sub": "a", "iat": 1, "exp": 2**31, "jti": "x"},
                           SECRET, algorithm="HS256")  # fmt: skip
    with pytest.raises(jwt.InvalidAudienceError):
        ops.verify_mcp_token(wrong_aud, SECRET)
    with pytest.raises(pydantic.ValidationError, match="lifetime"):
        _mint(ttl_s=3600)


def test_sse_events_frame_by_name_and_validate() -> None:
    name, data = ag.sse_payload(ag.TurnEvent(thread_id=THREAD, turn=1))
    assert name == "turn" and data == {"thread_id": str(THREAD), "turn": 1}
    parsed = ag.AGENT_EVENT.validate_python({"event": "token", "text": "hi"})
    assert isinstance(parsed, ag.TokenEvent)
    with pytest.raises(pydantic.ValidationError):
        ag.AGENT_EVENT.validate_python({"event": "token", "text": "hi", "extra": 1})


def test_the_web_schema_file_is_current() -> None:
    # Regenerate with: uv run python -m sre_shared.contracts.export
    assert SCHEMA_PATH.read_text(encoding="utf-8") == render()


def test_a_token_names_the_tools_it_may_call() -> None:
    token = ops.mint_mcp_token(
        secret="s" * 32, sub="alice", scopes=["viewer"], namespaces=["shop"],
        tools=["create_incident", "page_oncall"],
    )  # fmt: skip
    claims = ops.verify_mcp_token(token, "s" * 32)
    assert claims.tools == ["create_incident", "page_oncall"]
    unbound = ops.verify_mcp_token(
        ops.mint_mcp_token(secret="s" * 32, sub="alice", scopes=["viewer"], namespaces=[]), "s" * 32
    )
    assert unbound.tools is None


def test_get_logs_names_a_pod_or_a_deployment_but_not_both() -> None:
    assert ops.parse_args("get_logs", {"namespace": "shop", "deployment": "checkout"})
    assert ops.parse_args("get_logs", {"namespace": "shop", "pod": "checkout-abc-1"})
    for bad in ({"namespace": "shop"}, {"namespace": "shop", "pod": "p", "deployment": "d"}):
        with pytest.raises(pydantic.ValidationError):
            ops.parse_args("get_logs", bad)
