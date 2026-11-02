"""Authentication: a Supabase Auth JWT in `Authorization: Bearer`, else anonymous.

Supabase signs access tokens with an asymmetric key (ES256 by default, RS256 possible) published
at `/auth/v1/.well-known/jwks.json`; projects still on the legacy shared secret sign HS256. Each
algorithm is verified only with its own kind of key, and `none` or anything else is refused.

Groups come from the token's `app_metadata.groups`; the access-group filter itself is RLS, so
this module only builds `Claims`. A token that is present but invalid is rejected, never
downgraded to anonymous.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import jwt
from fastapi import HTTPException
from jwt.types import Options

from sre_shared.models import ANONYMOUS, Claims
from sre_shared.settings import get_settings

_ASYMMETRIC = ("ES256", "RS256")
_OPTIONS: Options = {"verify_aud": False, "require": ["exp"]}


@lru_cache(maxsize=4)
def jwks_client(url: str) -> jwt.PyJWKClient:
    """Keys are fetched once and kept (a rotated key id triggers a refetch)."""
    return jwt.PyJWKClient(url, cache_keys=True, lifespan=600, timeout=5)


def _payload(token: str) -> dict[str, Any]:
    settings = get_settings()
    alg = jwt.get_unverified_header(token).get("alg")
    if alg in _ASYMMETRIC:
        url = f"{settings.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"
        try:
            key = jwks_client(url).get_signing_key_from_jwt(token).key
        except jwt.PyJWKClientConnectionError as exc:
            raise HTTPException(status_code=503, detail="signing keys unavailable") from exc
        return jwt.decode(token, key, algorithms=[alg], options=_OPTIONS)
    if alg == "HS256":
        secret = settings.supabase_jwt_secret
        if secret is None:
            raise HTTPException(status_code=503, detail="JWT verification is not configured")
        return jwt.decode(token, secret.get_secret_value(), algorithms=["HS256"], options=_OPTIONS)
    raise jwt.InvalidAlgorithmError(f"unsupported algorithm {alg!r}")


def claims_from_authorization(authorization: str | None) -> Claims:
    if not authorization:
        return ANONYMOUS
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="expected 'Authorization: Bearer <jwt>'")
    try:
        payload = _payload(token.strip())
    except jwt.PyJWTError as exc:  # bad signature, unknown key id, expired, malformed
        raise HTTPException(status_code=401, detail="invalid token") from exc

    role = payload.get("role")
    if role == "anon":
        return ANONYMOUS
    sub = payload.get("sub")
    if role != "authenticated" or not isinstance(sub, str) or not sub:
        raise HTTPException(status_code=401, detail="token is not a user token")
    app_metadata = payload.get("app_metadata") or {}
    if not isinstance(app_metadata, dict):
        raise HTTPException(status_code=401, detail="malformed app_metadata claim")

    def strings(key: str) -> list[str]:
        raw = app_metadata.get(key) or []
        return [x for x in raw if isinstance(x, str)] if isinstance(raw, list) else []

    # Unknown scope names are dropped, never widened: only an admin sets app_metadata.
    scopes = [s for s in strings("scopes") if s in ("viewer", "operator", "admin")]
    return Claims(
        sub=sub,
        role="authenticated",
        groups=strings("groups"),
        scopes=scopes,  # filtered to the Scope literals above
        namespaces=strings("namespaces"),
    )
