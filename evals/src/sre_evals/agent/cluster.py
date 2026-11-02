"""The cluster the agent acts on, as the eval runner sees it, and a fake of it.

`ClusterState` is the three-method surface the runner needs from the simulator (T11) or any other
backend: reset to a scenario, a state snapshot, and the log of mutations that really ran. The
runner and the metrics read only those; `FakeCluster` implements the scenario semantics from
`config/simulator/scenarios.yaml` (a failure stays until its `fixed_by` action succeeds) so the
task set's expected end states can be checked for reachability without the simulator.

Shapes the runner reads (T15-1 in the orchestra bucket):

    snapshot()  -> {"deployments": {"ns/name": {"replicas": int, "revision": int,
                                                 "healthy": bool}, ...}, ...}
    mutations() -> [{"tool", "namespace", "target", "args", "idempotency_key", "changed"}, ...]
                   one entry per executed non-read call, in execution order.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from sre_shared.contracts import ops


@runtime_checkable
class ClusterState(Protocol):
    def reset(self, scenario: str, seed: int) -> None: ...

    def snapshot(self) -> dict[str, Any]: ...

    def mutations(self) -> list[dict[str, Any]]: ...


class FakeCluster:
    """Scenario semantics only: no timing (`after_s` is ignored), outcomes never random."""

    def __init__(
        self,
        scenarios: Mapping[str, Any],
        policy: ops.NamespacePolicy,
        *,
        dedupe: bool = True,
    ) -> None:
        self._scenarios = scenarios
        self._policy = policy
        # dedupe=True mimics mcp-ops: a repeated idempotency key returns the stored result and
        # does not run again. dedupe=False is a broken backend, for tests of the metrics.
        self.dedupe = dedupe
        self._name = ""
        self._deployments: dict[str, dict[str, Any]] = {}
        self._mutations: list[dict[str, Any]] = []
        self._results: dict[str, dict[str, Any]] = {}
        self._planted: dict[str, list[str]] = {}

    # --- ClusterState -----------------------------------------------------------------

    def reset(self, scenario: str, seed: int) -> None:
        sc = self._scenarios[scenario]
        self._name = scenario
        self._deployments = {
            f"{ns}/{dep}": {
                "replicas": d.replicas,
                "revision": d.current_revision,
                "failure": d.failure,
                "revisions": dict(d.revisions),
            }
            for ns, deps in sc.namespaces.items()
            for dep, d in deps.items()
        }
        self._mutations = []
        self._results = {}
        self._planted = {}

    def snapshot(self) -> dict[str, Any]:
        return {
            "scenario": self._name,
            "deployments": {
                k: {
                    "replicas": d["replicas"],
                    "revision": d["revision"],
                    "healthy": d["failure"] is None,
                }
                for k, d in sorted(self._deployments.items())
            },
        }

    def mutations(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._mutations)

    # --- execution (what mcp-ops does behind a tool) --------------------------------------

    def execute(
        self, tool: str, args: Mapping[str, Any], idempotency_key: str | None = None
    ) -> dict[str, Any]:
        """Run a write or escalation tool. Raises ValueError for a call the cluster cannot run."""
        parsed = ops.parse_args(tool, args)
        if idempotency_key is not None and self.dedupe and idempotency_key in self._results:
            return self._results[idempotency_key]
        norm = ops.normalize_args(tool, args)
        ns, target = ops.target_of(tool, parsed)
        result: dict[str, Any] = {"ok": True}
        if tool in ops.WRITE_TOOLS:
            if ns not in self._policy.allowed:
                raise ValueError(f"namespace {ns} is not on the allow-list")
            dep = self._deployments.get(f"{ns}/{target}")
            if dep is None:
                raise ValueError(f"no deployment {ns}/{target}")
            before = (dep["replicas"], dep["revision"], dep["failure"])
            result = self._apply_write(tool, norm, dep)
            changed = before != (dep["replicas"], dep["revision"], dep["failure"])
        else:
            changed = True
        self._mutations.append(
            {
                "seq": len(self._mutations) + 1,
                "tool": tool,
                "namespace": ns,
                "target": target,
                "args": norm,
                "idempotency_key": idempotency_key,
                "changed": changed,
            }
        )
        if idempotency_key is not None:
            self._results[idempotency_key] = result
        return result

    def _failure_of(self, dep: dict[str, Any]) -> Any:
        return self._scenarios[self._name].failures[dep["failure"]]

    def _apply_write(self, tool: str, args: dict[str, Any], dep: dict[str, Any]) -> dict[str, Any]:
        if tool == "scale_deployment":
            dep["replicas"] = args["replicas"]  # scaling never fixes a failure
        elif tool == "restart_deployment":
            if dep["failure"] and self._failure_of(dep).fixed_by == "restart_deployment":
                dep["failure"] = None
        else:  # rollback_deployment
            rev = args["to_revision"]
            if rev not in dep["revisions"]:
                raise ValueError(f"no revision {rev}")
            dep["revision"] = rev
            if dep["failure"]:
                f = self._failure_of(dep)
                if f.fixed_by == "rollback_deployment" and rev in f.good_revisions:
                    dep["failure"] = None
        return {"ok": True, "replicas": dep["replicas"], "revision": dep["revision"]}

    # --- reads and planting ---------------------------------------------------------------

    def plant_log(self, namespace: str, deployment: str, text: str) -> None:
        self._planted.setdefault(f"{namespace}/{deployment}", []).append(text)

    def read(self, tool: str, args: Mapping[str, Any]) -> dict[str, Any]:
        """Small, deterministic read results (enough for an agent to have something to read)."""
        parsed = ops.parse_args(tool, args)
        ns, target = ops.target_of(tool, parsed)
        if tool == "list_pods":
            return {
                "items": [
                    {"name": f"{k.split('/')[1]}-0", "status": self._pod_status(d)}
                    for k, d in sorted(self._deployments.items())
                    if k.startswith(f"{ns}/")
                ]
            }
        if tool == "get_logs":
            pod = str(getattr(parsed, "pod", None) or "")
            dep_name = getattr(parsed, "deployment", None) or pod.rsplit("-", 1)[0]
            d = self._deployments.get(f"{ns}/{dep_name}")
            lines = [self._failure_of(d).log] if d and d["failure"] else []
            return {"lines": [*lines, *self._planted.get(f"{ns}/{dep_name}", [])]}
        if tool == "get_events":
            return {
                "events": [
                    self._failure_of(d).event
                    for k, d in sorted(self._deployments.items())
                    if k.startswith(f"{ns}/") and d["failure"]
                ]
            }
        d = self._deployments.get(f"{ns}/{target}")
        return {"found": d is not None, "healthy": d is not None and d["failure"] is None}

    def _pod_status(self, d: dict[str, Any]) -> str:
        return self._failure_of(d).pod_status if d["failure"] else "Running"


def simulate_expected(
    item: Any, scenarios: Mapping[str, Any], policy: ops.NamespacePolicy
) -> list[str]:
    """Run the item's expected action on a fresh cluster and compare with its end state.

    This is the reachability check: an end state nobody can reach (a rollback to a revision that
    does not exist, `healthy: true` after a restart on a crash that needs a rollback, an approver
    without the scope) is a task no agent can pass.
    """
    e = item.expected
    out: list[str] = []
    cluster = FakeCluster(scenarios, policy)
    cluster.reset(item.scenario, 0)
    # Expected args are a subset (a create_incident has a title the item does not fix), so the
    # arguments are checked by name and the call is only run when the item gives all of them.
    unknown = set(e.args) - set(ops.TOOL_ARGS[e.tool].model_fields)
    if unknown:
        return [f"{item.id}: {sorted(unknown)} are not arguments of {e.tool}"]
    parsed = ops.TOOL_ARGS[e.tool].model_construct(**e.args)
    kind = ops.tool_kind(e.tool)
    why = ops.allowed(
        e.tool, parsed, list(item.caller.scopes), list(item.caller.namespaces), policy
    )
    if why:
        out.append(f"{item.id}: the caller cannot make the expected call: {why}")
    if kind != "read":
        level = ops.required_approval(e.tool, parsed, policy, route="action")
        wanted = e.approval_scope or "operator"
        if level != wanted and level != "none":
            out.append(f"{item.id}: approval_scope should be {level}, item says {wanted}")
        if e.approval == "approve" and level == "admin" and "admin" not in item.decider().scopes:
            out.append(f"{item.id}: nobody with the admin scope approves it")
        if e.approval == "approve" and e.tool in ops.WRITE_TOOLS:
            try:
                cluster.execute(e.tool, e.args, "k")
            except ValueError as exc:
                out.append(f"{item.id}: the expected call cannot run: {exc}")
    snap = cluster.snapshot()["deployments"]
    for ds in e.end_state:
        got = snap.get(ds.key)
        if got is None:
            continue  # reported by validate_items
        for field in ("replicas", "revision", "healthy"):
            want = getattr(ds, field)
            if want is not None and got[field] != want:
                out.append(
                    f"{item.id}: end state {ds.key}.{field}={want!r} is unreachable"
                    f" (the expected action leaves {got[field]!r})"
                )
    return out
