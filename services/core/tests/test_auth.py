"""Supabase access tokens signed with an asymmetric key (ES256 by default), verified against the
project's JWKS. Found wiring the web UI to a real local Supabase: every login was a 401 because
only HS256 was accepted."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException

from sre_core.api import auth

KID = "test-key"


@dataclass
class _Key:
    key: Any


class FakeJwks:
    def __init__(self, public: Any, *, down: bool = False) -> None:
        self.public, self.down = public, down

    def get_signing_key_from_jwt(self, token: str) -> _Key:
        if self.down:
            raise jwt.PyJWKClientConnectionError("unreachable")
        if jwt.get_unverified_header(token).get("kid") != KID:
            raise jwt.PyJWKClientError("unknown key id")
        return _Key(self.public)


def token(private: Any, **claims: Any) -> str:
    now = int(time.time())
    payload = {
        "sub": "user-1", "role": "authenticated", "exp": now + 300, "iat": now,
        "app_metadata": {"groups": ["public"], "scopes": ["viewer", "operator", "root"],
                         "namespaces": ["shop"]},
        **claims,
    }  # fmt: skip
    return jwt.encode(payload, private, algorithm="ES256", headers={"kid": KID})


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> Any:
    private = ec.generate_private_key(ec.SECP256R1())
    monkeypatch.setattr(auth, "jwks_client", lambda url: FakeJwks(private.public_key()))
    return private


def test_an_es256_token_from_the_projects_keys_is_accepted(keys: Any) -> None:
    claims = auth.claims_from_authorization(f"Bearer {token(keys)}")
    assert claims.sub == "user-1"
    assert claims.scopes == ["viewer", "operator"]  # unknown scope names are dropped
    assert claims.namespaces == ["shop"]


def test_a_token_signed_by_another_key_is_refused(keys: Any) -> None:
    other = ec.generate_private_key(ec.SECP256R1())
    with pytest.raises(HTTPException) as err:
        auth.claims_from_authorization(f"Bearer {token(other)}")
    assert err.value.status_code == 401


def test_an_unsigned_token_is_refused(keys: Any) -> None:
    unsigned = jwt.encode({"sub": "user-1", "role": "authenticated", "exp": 2**31}, None, "none")
    with pytest.raises(HTTPException) as err:
        auth.claims_from_authorization(f"Bearer {unsigned}")
    assert err.value.status_code == 401


def test_unreachable_signing_keys_are_a_503_not_anonymous(
    keys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(auth, "jwks_client", lambda url: FakeJwks(None, down=True))
    with pytest.raises(HTTPException) as err:
        auth.claims_from_authorization(f"Bearer {token(keys)}")
    assert err.value.status_code == 503
