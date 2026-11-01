"""Phase 1 authentication: a Supabase Auth JWT (HS256) in `Authorization: Bearer`, else anonymous.

Groups come from the token's `app_metadata.groups`; the access-group filter itself is RLS, so
this module only builds `Claims`. A token that is present but invalid is rejected, never
downgraded to anonymous.
"""

from __future__ import annotations

from typing import Any

import jwt
from fastapi import HTTPException

from sre_shared.models import ANONYMOUS, Claims
from sre_shared.settings import get_settings


def claims_from_authorization(authorization: str | None) -> Claims:
    if not authorization:
        return ANONYMOUS
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="expected 'Authorization: Bearer <jwt>'")
    secret = get_settings().supabase_jwt_secret
    if secret is None:
        raise HTTPException(status_code=503, detail="JWT verification is not configured")
    try:
        payload: dict[str, Any] = jwt.decode(
            token.strip(),
            secret.get_secret_value(),
            algorithms=["HS256"],
            options={"verify_aud": False, "require": ["exp"]},
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="invalid token") from exc

    role = payload.get("role")
    if role == "anon":
        return ANONYMOUS
    sub = payload.get("sub")
    if role != "authenticated" or not isinstance(sub, str) or not sub:
        raise HTTPException(status_code=401, detail="token is not a user token")
    raw_groups = (payload.get("app_metadata") or {}).get("groups") or []
    groups = [g for g in raw_groups if isinstance(g, str)] if isinstance(raw_groups, list) else []
    return Claims(sub=sub, role="authenticated", groups=groups)
