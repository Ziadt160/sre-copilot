"""Per-query-type retrieval: a query that contains a literal (error name, multi-word flag, field
path) uses the `router.literal` route from config/retrieval.yaml, anything else
`router.default`. Phase 1 found the keyword leg helps exact-string questions and hurts factual
ones; this sends each kind to the legs that suit it."""

from __future__ import annotations

from sre_shared.config import RouterConfig, load_retrieval_defaults
from sre_shared.literals import has_literal
from sre_shared.models import RetrievalRequest


def route_request(
    req: RetrievalRequest, router: RouterConfig | None = None
) -> tuple[RetrievalRequest, str]:
    """The request with the route's legs, weights and rrf_k applied, and the route name."""
    router = router or load_retrieval_defaults().router
    name = "literal" if has_literal(req.query) else "default"
    route = router.literal if name == "literal" else router.default
    routed = req.model_copy(
        update={
            "use_dense": route.use_dense,
            "use_keyword": route.use_keyword,
            "w_dense": route.w_dense,
            "w_keyword": route.w_keyword,
            "rrf_k": route.rrf_k,
            "route_by_query_type": False,  # applied; not re-applied downstream
        }
    )
    return routed, name
