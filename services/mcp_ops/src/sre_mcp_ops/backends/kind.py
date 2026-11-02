"""The kind backend (CI only): the same contract as the simulator, on a real API server.

The `kubernetes` client is synchronous, so every call runs in `asyncio.to_thread`. No kubectl
subprocess (the Windows selector loop cannot spawn one, and CI needs none). The package is the
`kind` extra; it is imported lazily so the lab, which uses the simulator, never needs it.

Idempotency by construction: restart stamps the pod template annotation `restartedAt` with the
operation's time; scale sets an absolute count; rollback copies the pod template of an explicit
revision and records the operation key on the deployment, because a real rollback renumbers the
revision it restores and a second "go to revision 1" would no longer find revision 1.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
from typing import Any

from sre_mcp_ops.backends.base import BackendError, LogChunk

KEY_ANNOTATION = "sre-copilot/operation-key"
RESTART_ANNOTATION = "kubectl.kubernetes.io/restartedAt"
REVISION_ANNOTATION = "deployment.kubernetes.io/revision"


class KindBackend:
    def __init__(self) -> None:
        k8s = importlib.import_module("kubernetes")
        try:
            k8s.config.load_kube_config()
        except Exception:
            k8s.config.load_incluster_config()
        self._exc: Any = importlib.import_module("kubernetes.client.exceptions")
        self._core = k8s.client.CoreV1Api()
        self._apps = k8s.client.AppsV1Api()
        self._client: Any = k8s.client

    async def _call(self, fn: Any, *a: Any, **kw: Any) -> Any:
        try:
            return await asyncio.to_thread(fn, *a, **kw)
        except self._exc.ApiException as exc:
            if exc.status == 404:
                raise BackendError("not_found", "object not found") from None
            raise BackendError("unavailable", f"cluster API error {exc.status}") from None

    @staticmethod
    def _pod(p: Any) -> dict[str, Any]:
        cs = p.status.container_statuses or []
        waiting = next((c.state.waiting.reason for c in cs if c.state and c.state.waiting), None)
        return {
            "name": p.metadata.name,
            "namespace": p.metadata.namespace,
            "deployment": (p.metadata.labels or {}).get("app"),
            "phase": p.status.phase,
            "status": waiting or p.status.phase,
            "ready": bool(cs) and all(c.ready for c in cs),
            "restarts": sum(c.restart_count for c in cs),
            "image": p.spec.containers[0].image,
        }

    async def list_pods(self, namespace: str, label_selector: str | None) -> list[dict[str, Any]]:
        res = await self._call(
            self._core.list_namespaced_pod, namespace, label_selector=label_selector or ""
        )
        return sorted((self._pod(p) for p in res.items), key=lambda p: str(p["name"]))

    async def _deployment(self, namespace: str, name: str) -> Any:
        return await self._call(self._apps.read_namespaced_deployment, name, namespace)

    @staticmethod
    def _dep_facts(d: Any) -> dict[str, Any]:
        tmpl = d.spec.template.metadata.annotations or {}
        return {
            "kind": "deployment",
            "name": d.metadata.name,
            "namespace": d.metadata.namespace,
            "replicas": d.spec.replicas,
            "ready_replicas": d.status.ready_replicas or 0,
            "revision": int((d.metadata.annotations or {}).get(REVISION_ANNOTATION, 0)),
            "image": d.spec.template.spec.containers[0].image,
            "restarted_at": tmpl.get(RESTART_ANNOTATION),
            "conditions": [c.reason for c in (d.status.conditions or []) if c.status != "True"],
        }

    async def describe(self, namespace: str, kind: str, name: str) -> dict[str, Any]:
        if kind == "deployment":
            return self._dep_facts(await self._deployment(namespace, name))
        if kind == "pod":
            return {"kind": "pod", **self._pod(await self._call(
                self._core.read_namespaced_pod, name, namespace))}  # fmt: skip
        s = await self._call(self._core.read_namespaced_service, name, namespace)
        return {
            "kind": "service",
            "name": name,
            "namespace": namespace,
            "type": s.spec.type,
            "cluster_ip": s.spec.cluster_ip,
            "selector": s.spec.selector or {},
        }

    async def get_logs(
        self, namespace: str, pod: str, container: str | None, tail_lines: int, previous: bool
    ) -> LogChunk:
        text = await self._call(
            self._core.read_namespaced_pod_log, pod, namespace,
            container=container, tail_lines=tail_lines, previous=previous,
        )  # fmt: skip
        return LogChunk(str(text), truncated=False)

    async def get_events(self, namespace: str, involved_object: str | None) -> list[dict[str, Any]]:
        sel = f"involvedObject.name={involved_object}" if involved_object else ""
        res = await self._call(self._core.list_namespaced_event, namespace, field_selector=sel)
        return [
            {
                "type": e.type,
                "reason": e.reason,
                "object": e.involved_object.name,
                "namespace": namespace,
                "message": e.message,
            }
            for e in res.items
        ]

    async def rollout_status(self, namespace: str, deployment: str) -> dict[str, Any]:
        d = await self._deployment(namespace, deployment)
        facts = self._dep_facts(d)
        failed = next(
            (c for c in (d.status.conditions or []) if c.reason == "ProgressDeadlineExceeded"), None
        )
        done = facts["ready_replicas"] == facts["replicas"] == (d.status.updated_replicas or 0)
        status = "failed" if failed else "complete" if done else "progressing"
        return {
            "namespace": namespace,
            "deployment": deployment,
            "replicas": facts["replicas"],
            "revision": facts["revision"],
            "ready_replicas": facts["ready_replicas"],
            "status": status,
            "reason": "ProgressDeadlineExceeded" if failed else "",
        }

    async def restart(
        self, namespace: str, deployment: str, *, key: str, at: str
    ) -> dict[str, Any]:
        d = await self._deployment(namespace, deployment)
        previous = (d.spec.template.metadata.annotations or {}).get(RESTART_ANNOTATION)
        if previous != at:
            body = {"spec": {"template": {"metadata": {"annotations": {RESTART_ANNOTATION: at}}}}}
            await self._call(self._apps.patch_namespaced_deployment, deployment, namespace, body)
        return {"restarted_at": at, "changed": previous != at}

    async def scale(
        self, namespace: str, deployment: str, replicas: int, *, key: str
    ) -> dict[str, Any]:
        d = await self._deployment(namespace, deployment)
        previous = int(d.spec.replicas)
        if previous != replicas:
            body = {"spec": {"replicas": replicas}}
            await self._call(
                self._apps.patch_namespaced_deployment_scale, deployment, namespace, body
            )
        return {"replicas": replicas, "previous": previous, "changed": previous != replicas}

    async def rollback(
        self, namespace: str, deployment: str, to_revision: int, *, key: str
    ) -> dict[str, Any]:
        d = await self._deployment(namespace, deployment)
        previous = int((d.metadata.annotations or {}).get(REVISION_ANNOTATION, 0))
        if (d.metadata.annotations or {}).get(KEY_ANNOTATION) == key:
            return {"revision": previous, "previous_revision": previous, "changed": False}
        selector = ",".join(f"{k}={v}" for k, v in (d.spec.selector.match_labels or {}).items())
        sets = await self._call(
            self._apps.list_namespaced_replica_set, namespace, label_selector=selector
        )
        target = next(
            (
                rs for rs in sets.items
                if (rs.metadata.annotations or {}).get(REVISION_ANNOTATION) == str(to_revision)
            ),
            None,
        )  # fmt: skip
        if target is None:
            raise BackendError(
                "invalid_revision", f"revision {to_revision} does not exist for {deployment}"
            )
        template = self._client.ApiClient().sanitize_for_serialization(target.spec.template)
        template = copy.deepcopy(template)
        template.get("metadata", {}).get("labels", {}).pop("pod-template-hash", None)
        body = {
            "metadata": {"annotations": {KEY_ANNOTATION: key}},
            "spec": {"template": template},
        }
        await self._call(self._apps.patch_namespaced_deployment, deployment, namespace, body)
        return {"revision": to_revision, "previous_revision": previous, "changed": True}

    async def applied(self, tool: str, args: dict[str, Any], *, key: str, at: str) -> bool:
        d = await self._deployment(args["namespace"], args["deployment"])
        if tool == "restart_deployment":
            stamp = (d.spec.template.metadata.annotations or {}).get(RESTART_ANNOTATION)
            return bool(stamp == at)
        if tool == "scale_deployment":
            return bool(int(d.spec.replicas) == args["replicas"])
        return bool((d.metadata.annotations or {}).get(KEY_ANNOTATION) == key)

    async def record_escalation(self, tool: str, args: dict[str, Any], *, key: str) -> None:
        return None
