"""Shared fixtures for golden-tooling tests: fixture corpus, a fake OpenAI client, key scripting."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from sre_evals.golden.llm import ModelSpec, OpenAICompatLLM
from sre_shared.models import DocType
from sre_shared.testing.seed import FixtureDoc, create_corpus_version, insert_docs

PAD = " The control plane reconciles the observed state with the desired state continuously."


def _body(tag: str, text: str) -> str:
    return f"[T:{tag}] {text}{PAD * 2}"


DOCS = [
    FixtureDoc(
        source_path="content/en/docs/concepts/workloads/pods/_index.md",
        k8s_version="1.34",
        title="Pods",
        sections=[
            (
                (),
                _body("pod0", "A Pod is the smallest deployable unit of computing in Kubernetes."),
            ),
            (
                ("Restarts",),
                _body(
                    "pod1",
                    "A container that keeps failing enters CrashLoopBackOff and is restarted.",
                ),
            ),
        ],
    ),
    FixtureDoc(
        source_path="content/en/docs/concepts/workloads/controllers/deployment.md",
        k8s_version="1.34",
        title="Deployments",
        sections=[
            ((), _body("dep0", "A Deployment manages a set of replicas of an application.")),
            (
                ("Rolling update",),
                _body("dep1", "Use --max-surge to bound extra Pods during a rolling update."),
            ),
            (
                ("Template",),
                _body("dep2", "Set spec.template.spec.containers to define the Pod containers."),
            ),
        ],
    ),
    FixtureDoc(
        source_path="content/en/docs/concepts/workloads/controllers/deployment.md",
        k8s_version="1.33",
        title="Deployments",
        sections=[((), _body("dep33", "A Deployment manages a set of replicas in version 1.33."))],
    ),
    FixtureDoc(
        source_path="content/en/docs/tasks/debug/debug-application/debug-pods.md",
        k8s_version="1.34",
        title="Debug Pods",
        doc_type=DocType.TASK,
        sections=[
            (
                ("Images",),
                _body("dbg0", "An ImagePullBackOff status means the image cannot be pulled."),
            ),
            ((), _body("dbg1", "Use kubectl describe pod to see events for a failing Pod.")),
        ],
    ),
    FixtureDoc(
        source_path="content/en/docs/reference/kubectl/cheatsheet.md",
        k8s_version="1.34",
        title="kubectl cheatsheet",
        doc_type=DocType.REFERENCE,
        sections=[
            ((), _body("ref0", "Reference listing with CrashLoopBackOff and --dry-run=client."))
        ],
    ),
    FixtureDoc(
        source_path="internal/private.md",
        k8s_version="1.34",
        title="Private",
        access_group="team-a",
        sections=[((), _body("prv0", "Private note about CrashLoopBackOff handling."))],
    ),
]


@pytest.fixture
async def corpus(db: Any) -> int:
    """Seed the fixture corpus; returns the corpus version id."""
    cv = await create_corpus_version(db)
    await insert_docs(db, DOCS, corpus_version_id=cv)
    return cv


def _chat_reply(content: str) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeOpenAI:
    """Stands in for openai.OpenAI / AsyncOpenAI: `chat.completions.create(...)`.

    Questions are looked up by the `[T:tag]` marker found in the chunk text of the prompt.
    """

    def __init__(
        self,
        questions: dict[str, str] | None = None,
        critic_score: Callable[[str], float] | None = None,
        repair: Callable[[str], str] | None = None,
        drop_tags: set[str] | None = None,
    ) -> None:
        self.questions = questions or {}
        self.critic_score = critic_score or (lambda q: 0.9)
        self.repair = repair
        self.drop_tags = drop_tags or set()
        self.prompts: list[str] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.achat = SimpleNamespace(completions=SimpleNamespace(create=self._acreate))

    def question_for(self, tag: str) -> str:
        return self.questions.get(tag, f"What does the documentation say about {tag}?")

    def _answer(self, prompt: str) -> str:
        if '"title": "SyntheticDataList"' in prompt:
            tags = re.findall(r"\[T:(\w+)\]", prompt.split("Context:")[-1])
            if not tags or tags[0] in self.drop_tags:
                return json.dumps({"data": []})
            return json.dumps({"data": [{"input": self.question_for(tags[0])}]})
        if '"title": "InputFeedback"' in prompt:
            q = prompt.split("Query:")[-1] if "Query:" in prompt else prompt
            score = next(
                (self.critic_score(v) for v in self.questions.values() if v in q),
                self.critic_score(q),
            )
            return json.dumps({"score": score, "feedback": "fine"})
        if '"title": "RewrittenInput"' in prompt:
            return json.dumps({"rewritten_input": "What is rewritten?"})
        if "Rewrite the question so that it contains the exact text" in prompt:
            return self.repair(prompt) if self.repair else "no luck"
        if '"title": "Response"' in prompt:
            return json.dumps({"response": "A grounded answer."})
        return "A grounded answer."

    def _create(self, *, messages: list[dict[str, str]], **_: Any) -> Any:
        prompt = messages[-1]["content"] if len(messages) == 1 else messages[0]["content"]
        self.prompts.append(prompt)
        return _chat_reply(self._answer(prompt))

    async def _acreate(self, *, messages: list[dict[str, str]], **kw: Any) -> Any:
        return self._create(messages=messages, **kw)


@pytest.fixture
def make_llm() -> Callable[..., tuple[OpenAICompatLLM, FakeOpenAI]]:
    def make(**kw: Any) -> tuple[OpenAICompatLLM, FakeOpenAI]:
        fake = FakeOpenAI(**kw)
        spec = ModelSpec("ollama", "fake-model", "http://127.0.0.1:1/v1", "none")
        return OpenAICompatLLM(spec, client=fake, aclient=SimpleNamespace(chat=fake.achat)), fake

    return make
