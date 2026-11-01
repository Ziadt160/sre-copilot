# mcp-ops (Phase 3)

FastMCP server for cluster tools: read tools, approved writes run as MCP tasks, scopes checked
here (not only in the agent), idempotency keys and an audit log in Postgres. Two backends behind
one interface: a kind cluster (lab, CI) and a simulator with seeded failures (public demo).
