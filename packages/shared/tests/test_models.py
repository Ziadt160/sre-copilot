import pytest
from pydantic import ValidationError

from sre_shared.models import ANONYMOUS, Claims, RetrievalRequest


def test_request_needs_a_leg() -> None:
    with pytest.raises(ValidationError):
        RetrievalRequest(query="x", use_dense=False, use_keyword=False)
    assert RetrievalRequest(query="x", use_dense=False).use_keyword


def test_request_is_hashable_for_caching() -> None:
    a = RetrievalRequest(query="CrashLoopBackOff", k=8)
    b = RetrievalRequest(query="CrashLoopBackOff", k=8)
    assert a == b


def test_claims_shape_matches_supabase_jwt() -> None:
    c = Claims(sub="u1", groups=["team-a"])
    assert c.to_jwt_claims() == {
        "sub": "u1",
        "role": "authenticated",
        "app_metadata": {"groups": ["team-a"]},
    }
    assert ANONYMOUS.role == "anon"
