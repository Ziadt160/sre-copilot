"""Bearer-token auth: core signs a short-lived token per turn, mcp-ops verifies it here.

Verification happens in FastMCP's HTTP auth layer, before a request is parsed as MCP: a missing,
expired, forged or wrong-audience token is a 401 and never reaches a tool. Nothing else in this
package reads the Authorization header. The verified claims travel as the request's access token,
which FastMCP also snapshots for background tasks, so a task body sees the same caller.
"""

from __future__ import annotations

import logging

import jwt
from fastmcp.server.auth import AccessToken, TokenVerifier
from fastmcp.server.dependencies import get_access_token
from pydantic import ValidationError

from sre_shared.contracts.ops import MCP_TOKEN_ISSUER, McpTokenClaims, verify_mcp_token

log = logging.getLogger(__name__)


class McpTokenVerifier(TokenVerifier):
    def __init__(self, secret: str) -> None:
        super().__init__()
        self._secret = secret

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            claims = verify_mcp_token(token, self._secret)
        except (jwt.InvalidTokenError, ValidationError) as exc:
            log.info("mcp token refused: %s", type(exc).__name__)  # never the token itself
            return None
        return AccessToken(
            # The bearer string is not kept: task snapshots persist the access token, and the
            # claims are all any tool needs.
            token="verified",  # noqa: S106
            client_id=MCP_TOKEN_ISSUER,
            scopes=[],
            expires_at=claims.exp,
            claims=claims.model_dump(mode="json"),
        )


def current_claims() -> McpTokenClaims | None:
    """The verified caller, or None when there is none (a restored task that lost its caller
    runs anonymously in fastmcp-tasks; every entry point treats None as a refusal)."""
    token = get_access_token()
    if token is None or not token.claims:
        return None
    try:
        return McpTokenClaims.model_validate(token.claims)
    except ValidationError:
        return None
