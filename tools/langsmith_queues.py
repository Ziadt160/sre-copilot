"""Set up the human-review queue in LangSmith: dry run by default, changes only with --apply.

Creates, once each:

  * annotation queue  "sre-copilot-review"  (rubric below)
  * rule              "sre-copilot-thumbs-down"   every root run a user rated thumbs-down
  * rule              "sre-copilot-review-sample" a 10% random sample of root runs
    both rules add the run to the queue, with extended trace retention OFF (extended retention
    is billed; the free plan keeps traces 14 days, and the labels are mirrored into Postgres by
    `python -m sre_core.review.mirror` before then).

    uv run python tools/langsmith_queues.py            # prints the plan, sends nothing
    uv run python tools/langsmith_queues.py --apply    # creates what is missing (safe to repeat)

The dry run needs no key. With a key it also reads the workspace to say what already exists;
it never writes. The LangSmith SDK has no method for automation rules, so rules go through the
REST endpoint /runs/rules. If the response does not confirm that retention extension is off,
this tool says so and you must switch the toggle off by hand on each rule (LangSmith > project >
Automations > rule > "Extend retention").
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

QUEUE_NAME = "sre-copilot-review"
RULE_THUMBS_DOWN = "sre-copilot-thumbs-down"
RULE_SAMPLE = "sre-copilot-review-sample"
SAMPLE_RATE = 0.10

RUBRIC = """\
Grade each answer on three things, then each tool action it proposed.

1. faithfulness: faithful (every claim is in the cited sources) / partly (some claims are not) /
   not (the answer is not supported by its sources).
2. tool choice: right tool / wrong tool (for action turns; skip for plain answers).
3. tool arguments: right args / wrong args (namespace, name, replicas, revision).
4. end state: did the cluster end up as the user wanted (yes / no / not applicable).

Use the comment box for what was wrong. Add a correction when you know the right answer.\
"""

# Feedback written by the app is `user_rating` (1 = up, 0 = down); see sre_core.review.feedback_sync.
_FILTER_THUMBS_DOWN = (
    'and(eq(is_root, true), eq(feedback_key, "user_rating"), eq(feedback_score, 0))'
)
_FILTER_SAMPLE = "eq(is_root, true)"


@dataclass(frozen=True)
class RuleSpec:
    name: str
    sampling_rate: float
    filter: str


RULES = (
    RuleSpec(RULE_THUMBS_DOWN, 1.0, _FILTER_THUMBS_DOWN),
    RuleSpec(RULE_SAMPLE, SAMPLE_RATE, _FILTER_SAMPLE),
)


@dataclass
class Report:
    would: list[str] = field(default_factory=list)  # dry run
    created: list[str] = field(default_factory=list)
    existing: list[str] = field(default_factory=list)
    manual: list[str] = field(default_factory=list)  # steps the user must do by hand


def rule_body(spec: RuleSpec, *, project_id: str, queue_id: str) -> dict[str, Any]:
    return {
        "display_name": spec.name,
        "session_id": project_id,
        "sampling_rate": spec.sampling_rate,
        "filter": spec.filter,
        "add_to_annotation_queue_id": queue_id,
        "extend_only": False,
        "is_enabled": True,
        # Extended retention is billed. The field name is the one the API returns on a rule.
        "extend_annotation_queue_trace_retention": False,
    }


def _find_queue(client: Any) -> Any | None:
    return next(iter(client.list_annotation_queues(name=QUEUE_NAME)), None)


def _list_rules(client: Any, project_id: str) -> dict[str, dict[str, Any]]:
    resp = client.request_with_retries("GET", "/runs/rules", params={"session_id": project_id})
    return {r["display_name"]: r for r in resp.json()}


def _project_id(client: Any, project: str) -> str | None:
    try:
        return str(client.read_project(project_name=project).id)
    except Exception:  # SDK raises LangSmithNotFoundError; any failure means "not there yet"
        return None


def _retention_off(rule: dict[str, Any]) -> bool:
    return rule.get("extend_annotation_queue_trace_retention") is False


def _manual_retention_note(name: str) -> str:
    return (
        f"Rule {name!r}: the API did not confirm retention extension is off. Open LangSmith > "
        "project > Automations > the rule, and switch off extended data retention by hand."
    )


def plan(client: Any | None, project: str) -> Report:
    """Dry run: what would be created. Reads only (and only if there is a client)."""
    report = Report()
    queue = _find_queue(client) if client else None
    if queue:
        report.existing.append(f"queue {QUEUE_NAME}")
    else:
        report.would.append(f"create queue {QUEUE_NAME} with the review rubric")
    project_id = _project_id(client, project) if client else None
    rules = _list_rules(client, project_id) if client and project_id else {}
    for spec in RULES:
        if spec.name in rules:
            report.existing.append(f"rule {spec.name}")
            if not _retention_off(rules[spec.name]):
                report.manual.append(_manual_retention_note(spec.name))
        else:
            report.would.append(
                f"create rule {spec.name}: sampling {spec.sampling_rate:g}, filter "
                f"{spec.filter}, add to {QUEUE_NAME}, retention extension OFF"
            )
    if client and project_id is None:
        report.manual.append(
            f"project {project!r} does not exist yet: run the app once with tracing on, "
            "then --apply (rules attach to a project)"
        )
    return report


def apply(client: Any, project: str) -> Report:
    report = Report()
    queue = _find_queue(client)
    if queue is None:
        queue = client.create_annotation_queue(
            name=QUEUE_NAME,
            description="SRE Copilot answers and actions for human grading",
            rubric_instructions=RUBRIC,
        )
        report.created.append(f"queue {QUEUE_NAME}")
    else:
        report.existing.append(f"queue {QUEUE_NAME}")

    project_id = _project_id(client, project)
    if project_id is None:
        report.manual.append(
            f"project {project!r} does not exist yet: run the app once with tracing on, "
            "then run this again to create the rules"
        )
        return report

    rules = _list_rules(client, project_id)
    for spec in RULES:
        if spec.name in rules:
            report.existing.append(f"rule {spec.name}")
            if not _retention_off(rules[spec.name]):
                report.manual.append(_manual_retention_note(spec.name))
            continue
        resp = client.request_with_retries(
            "POST",
            "/runs/rules",
            json=rule_body(spec, project_id=project_id, queue_id=str(queue.id)),
        )
        report.created.append(f"rule {spec.name}")
        if not _retention_off(resp.json()):
            report.manual.append(_manual_retention_note(spec.name))
    return report


def _print(report: Report, *, applied: bool) -> None:
    if not applied:
        print("DRY RUN: nothing was changed. Pass --apply to create:")
        for line in report.would or ["(nothing: everything exists)"]:
            print(f"  would {line}")
    else:
        for line in report.created:
            print(f"  created {line}")
    for line in report.existing:
        print(f"  exists  {line}")
    for line in report.manual:
        print(f"  MANUAL  {line}")


def main(
    argv: list[str] | None = None, *, client_factory: Callable[[str], Any] | None = None
) -> int:
    parser = argparse.ArgumentParser(description="Set up the LangSmith review queue and rules.")
    parser.add_argument("--apply", action="store_true", help="create what is missing")
    parser.add_argument("--project", default=os.environ.get("LANGSMITH_PROJECT", "sre-copilot"))
    args = parser.parse_args(argv)

    key = os.environ.get("LANGSMITH_API_KEY")
    client: Any | None = None
    if key:
        if client_factory is None:
            from langsmith import Client

            client_factory = lambda k: Client(api_key=k)  # noqa: E731
        client = client_factory(key)
    if args.apply and client is None:
        print("LANGSMITH_API_KEY is not set; nothing to apply to.", file=sys.stderr)
        return 2
    if args.apply:
        _print(apply(client, args.project), applied=True)
    else:
        _print(plan(client, args.project), applied=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
