"""A deterministic cluster for the lab, the public demo and the agent evals.

State comes from `config/simulator/scenarios.yaml`. Nothing here reads a clock or a random
source: pod names and injected failures come from `sha256(seed, ...)`, so the same seed and the
same idempotency key always give the same outcome, and a retry sees what the first attempt saw.

`reset`, `mutation_log`, `snapshot` and `set_log` are a Python API for tests and the eval rig;
they are deliberately not MCP tools, so the agent cannot reach them.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any

from sre_mcp_ops.backends.base import BackendError, LogChunk
from sre_shared.config import SimFailure, SimScenarios, load_simulator_scenarios

_HEALTHY_LOG = (
    'level=info msg="listening" port=8080\n'
    'level=info msg="health check ok"\n'
    'level=info msg="request served" status=200'
)
_REASONS = {
    "CrashLoopBackOff": "BackOff",
    "OOMKilled": "OOMKilling",
    "ImagePullBackOff": "Failed",
}


@dataclass
class _Dep:
    replicas: int
    revisions: dict[int, str]
    current_revision: int
    failure: str | None
    restarted_at: str | None = None
    log_override: str | None = None


@dataclass
class _State:
    deps: dict[str, dict[str, _Dep]] = field(default_factory=dict)
    failures: dict[str, SimFailure] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)


def _digest(*parts: object) -> str:
    return hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()


class SimulatorBackend:
    def __init__(
        self,
        scenario: str = "default",
        seed: int = 7,
        *,
        scenarios: SimScenarios | None = None,
        failure_rate: float = 0.0,
    ) -> None:
        self._scenarios = scenarios or load_simulator_scenarios()
        self._failure_rate = failure_rate
        self._log: list[dict[str, Any]] = []
        self._state = _State()
        self.scenario = scenario
        self.seed = seed
        self.reset(scenario, seed)

    # --- test and eval hooks (not MCP tools) -------------------------------------------------

    def reset(self, scenario: str | None = None, seed: int | None = None) -> None:
        self.scenario = scenario or self.scenario
        self.seed = self.seed if seed is None else seed
        sc = self._scenarios.scenarios.get(self.scenario)
        if sc is None:
            raise KeyError(f"unknown simulator scenario {self.scenario!r}")
        state = _State(failures=dict(sc.failures))
        for ns, deps in sc.namespaces.items():
            state.deps[ns] = {
                name: _Dep(
                    replicas=d.replicas,
                    revisions=dict(d.revisions),
                    current_revision=d.current_revision,
                    failure=d.failure,
                )
                for name, d in deps.items()
            }
        self._state = state
        self._log = []

    def mutation_log(self) -> list[dict[str, Any]]:
        """Every write application that reached the cluster, in order. `changed` says whether
        state moved; the duplicate count counts entries per idempotency key beyond the first."""
        return copy.deepcopy(self._log)

    def snapshot(self) -> dict[str, Any]:
        return {
            "deployments": {
                f"{ns}/{name}": {
                    "replicas": d.replicas,
                    "revision": d.current_revision,
                    "healthy": d.failure is None,
                    "restarted_at": d.restarted_at,
                }
                for ns, deps in self._state.deps.items()
                for name, d in deps.items()
            }
        }

    def set_log(self, namespace: str, deployment: str, text: str) -> None:
        """Replace what a deployment's pods log (the eval rig plants hostile log text here)."""
        self._dep(namespace, deployment).log_override = text

    # --- helpers ------------------------------------------------------------------------------

    def _dep(self, namespace: str, name: str) -> _Dep:
        try:
            return self._state.deps[namespace][name]
        except KeyError:
            raise BackendError("not_found", f"deployment {name} not found in {namespace}") from None

    def _failure(self, d: _Dep) -> SimFailure | None:
        return self._state.failures[d.failure] if d.failure else None

    def _unit(self, key: str) -> float:
        return int(_digest(self.seed, key)[:8], 16) / 2**32

    def _fail_injected(self, key: str) -> None:
        if self._failure_rate and self._unit(key) < self._failure_rate:
            raise BackendError("unavailable", "the cluster API did not answer; try again")

    def _pods(self, namespace: str, label_selector: str | None) -> list[dict[str, Any]]:
        wanted: dict[str, str] = {}
        if label_selector:
            for term in label_selector.split(","):
                k, sep, v = term.strip().replace("==", "=").partition("=")
                if not sep or not k:
                    raise BackendError("invalid_selector", "label_selector must be k=v[,k=v]")
                wanted[k] = v
        pods: list[dict[str, Any]] = []
        for name, d in self._state.deps.get(namespace, {}).items():
            if any(k != "app" or v != name for k, v in wanted.items()):
                continue
            f = self._failure(d)
            status = f.pod_status if f else "Running"
            ready = f is None
            rs = _digest(self.seed, namespace, name, d.current_revision, d.restarted_at)[:9]
            for i in range(d.replicas):
                pods.append(
                    {
                        "name": f"{name}-{rs}-{_digest(self.seed, rs, i)[:5]}",
                        "namespace": namespace,
                        "deployment": name,
                        "phase": "Running" if status in ("Running", "OOMKilled") else "Pending",
                        "status": status,
                        "ready": ready,
                        "restarts": f.restarts if f else 0,
                        "image": d.revisions[d.current_revision],
                    }
                )
        return pods

    def _logged(
        self,
        tool: str,
        ns: str,
        target: str,
        args: dict[str, Any],
        key: str,
        *,
        changed: bool,
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
    ) -> None:
        self._log.append(
            {
                "seq": len(self._log) + 1,
                "tool": tool,
                "namespace": ns,
                "target": target,
                "args": args,
                "idempotency_key": key,
                "changed": changed,
                "before": before or {},
                "after": after or {},
            }
        )

    def _note(self, ns: str, dep: str, reason: str, message: str) -> None:
        self._state.events.append(
            {"type": "Normal", "reason": reason, "object": dep, "namespace": ns, "message": message}
        )

    # --- reads --------------------------------------------------------------------------------

    async def list_pods(self, namespace: str, label_selector: str | None) -> list[dict[str, Any]]:
        if namespace not in self._state.deps:
            raise BackendError("not_found", f"namespace {namespace} not found")
        return self._pods(namespace, label_selector)

    async def describe(self, namespace: str, kind: str, name: str) -> dict[str, Any]:
        if kind == "pod":
            for p in self._pods(namespace, None):
                if p["name"] == name:
                    return {"kind": "pod", **p}
            raise BackendError("not_found", f"pod {name} not found in {namespace}")
        d = self._dep(namespace, name)
        if kind == "service":
            octet = int(_digest(self.seed, namespace, name)[:2], 16)
            return {
                "kind": "service",
                "name": name,
                "namespace": namespace,
                "type": "ClusterIP",
                "cluster_ip": f"10.96.0.{octet}",
                "selector": {"app": name},
            }
        f = self._failure(d)
        pods = self._pods(namespace, f"app={name}")
        return {
            "kind": "deployment",
            "name": name,
            "namespace": namespace,
            "replicas": d.replicas,
            "ready_replicas": sum(1 for p in pods if p["ready"]),
            "revision": d.current_revision,
            "image": d.revisions[d.current_revision],
            "revisions": sorted(d.revisions),
            "restarted_at": d.restarted_at,
            "conditions": [f.condition] if f and f.condition else [],
        }

    async def get_logs(
        self, namespace: str, pod: str, container: str | None, tail_lines: int, previous: bool
    ) -> LogChunk:
        owner = next((p for p in self._pods(namespace, None) if p["name"] == pod), None)
        if owner is None:
            raise BackendError("not_found", f"pod {pod} not found in {namespace}")
        d = self._dep(namespace, owner["deployment"])
        f = self._failure(d)
        if d.log_override is not None:
            text = d.log_override
        elif f is not None:
            text = f.log
        else:
            text = "" if previous else _HEALTHY_LOG
        lines = text.splitlines()
        return LogChunk("\n".join(lines[-tail_lines:]), truncated=len(lines) > tail_lines)

    async def get_events(self, namespace: str, involved_object: str | None) -> list[dict[str, Any]]:
        if namespace not in self._state.deps:
            raise BackendError("not_found", f"namespace {namespace} not found")
        events: list[dict[str, Any]] = []
        for name, d in self._state.deps[namespace].items():
            f = self._failure(d)
            if f is not None and f.event:
                events.append(
                    {
                        "type": "Warning",
                        "reason": f.condition or _REASONS.get(f.pod_status, "Failed"),
                        "object": name,
                        "namespace": namespace,
                        "message": f.event,
                    }
                )
        events += [e for e in self._state.events if e["namespace"] == namespace]
        return [e for e in events if involved_object in (None, e["object"])]

    async def rollout_status(self, namespace: str, deployment: str) -> dict[str, Any]:
        d = self._dep(namespace, deployment)
        f = self._failure(d)
        base = {
            "namespace": namespace,
            "deployment": deployment,
            "replicas": d.replicas,
            "revision": d.current_revision,
        }
        if f is not None and f.condition:
            return {
                **base,
                "status": "failed",
                "reason": f.condition,
                "progress_deadline_s": f.after_s,
                "ready_replicas": 0,
            }
        if f is not None:
            return {
                **base,
                "status": "progressing",
                "reason": "ReplicasNotReady",
                "ready_replicas": 0,
            }
        return {
            **base,
            "status": "complete",
            "reason": "NewReplicaSetAvailable",
            "ready_replicas": d.replicas,
        }

    # --- writes (idempotent by construction) -------------------------------------------------

    async def restart(
        self, namespace: str, deployment: str, *, key: str, at: str
    ) -> dict[str, Any]:
        d = self._dep(namespace, deployment)
        self._fail_injected(key)
        before = {"restarted_at": d.restarted_at, "healthy": d.failure is None}
        changed = d.restarted_at != at
        d.restarted_at = at
        f = self._failure(d)
        if f is not None and f.fixed_by == "restart_deployment":
            d.failure = None
            changed = True
        self._logged(
            "restart_deployment",
            namespace,
            deployment,
            {"namespace": namespace, "deployment": deployment},
            key,
            changed=changed,
            before=before,
            after={"restarted_at": at, "healthy": d.failure is None},
        )
        if changed:
            self._note(namespace, deployment, "Restarted", f"Deployment {deployment} restarted")
        return {"restarted_at": at, "changed": changed, "healthy": d.failure is None}

    async def scale(
        self, namespace: str, deployment: str, replicas: int, *, key: str
    ) -> dict[str, Any]:
        d = self._dep(namespace, deployment)
        self._fail_injected(key)
        previous = d.replicas
        d.replicas = replicas
        self._logged(
            "scale_deployment",
            namespace,
            deployment,
            {"namespace": namespace, "deployment": deployment, "replicas": replicas},
            key,
            changed=previous != replicas,
            before={"replicas": previous},
            after={"replicas": replicas},
        )
        if previous != replicas:
            self._note(
                namespace,
                deployment,
                "ScalingReplicaSet",
                f"Scaled {deployment} from {previous} to {replicas}",
            )
        return {"replicas": replicas, "previous": previous, "changed": previous != replicas}

    async def rollback(
        self, namespace: str, deployment: str, to_revision: int, *, key: str
    ) -> dict[str, Any]:
        d = self._dep(namespace, deployment)
        if to_revision not in d.revisions:
            raise BackendError(
                "invalid_revision", f"revision {to_revision} does not exist for {deployment}"
            )
        self._fail_injected(key)
        previous = d.current_revision
        d.current_revision = to_revision
        f = self._failure(d)
        cleared = (
            f is not None
            and f.fixed_by == "rollback_deployment"
            and (to_revision in f.good_revisions)
        )
        if cleared:
            d.failure = None
        changed = previous != to_revision or cleared
        self._logged(
            "rollback_deployment",
            namespace,
            deployment,
            {"namespace": namespace, "deployment": deployment, "to_revision": to_revision},
            key,
            changed=changed,
            before={"revision": previous},
            after={"revision": to_revision, "healthy": d.failure is None},
        )
        if changed:
            self._note(
                namespace,
                deployment,
                "RolledBack",
                f"Rolled {deployment} back to revision {to_revision}",
            )
        return {
            "revision": to_revision,
            "previous_revision": previous,
            "image": d.revisions[to_revision],
            "changed": changed,
            "healthy": d.failure is None,
        }

    async def applied(self, tool: str, args: dict[str, Any], *, key: str, at: str) -> bool:
        return any(e["idempotency_key"] == key and e["tool"] == tool for e in self._log)

    async def record_escalation(self, tool: str, args: dict[str, Any], *, key: str) -> None:
        target = str(args.get("rotation") or args.get("title") or "")
        self._logged(tool, str(args.get("namespace") or ""), target, args, key, changed=True)
