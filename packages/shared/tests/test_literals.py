"""Literal detection shared by the golden picker and the query router."""

from __future__ import annotations

import pytest

from sre_shared.literals import find_exact_strings, has_literal


@pytest.mark.parametrize(
    "text",
    [
        "pod stuck in CrashLoopBackOff",
        "what does --max-surge do",
        "set spec.template.spec.containers",
        "ErrImagePull after a registry change",
    ],
)
def test_has_literal(text: str) -> None:
    assert has_literal(text)


@pytest.mark.parametrize(
    "text", ["how do I restart a pod?", "use --force", "what is a spec?", "kubectl get pods"]
)
def test_plain_questions_have_no_literal(text: str) -> None:
    assert not has_literal(text)


def test_find_exact_strings_orders_by_first_appearance() -> None:
    text = "spec.template.spec.containers then --max-surge then CrashLoopBackOff"
    assert [s for _, s in find_exact_strings(text)] == [
        "spec.template.spec.containers",
        "--max-surge",
        "CrashLoopBackOff",
    ]
